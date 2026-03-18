"""
YoloeDetector — YOLOE open-vocabulary detection with BoTSORT tracking.

Mirrors the YoloDetector flow exactly:
  • BoTSORT tracking (model.track) for persistent track IDs
  • Multi-frame confirmation (high-conf=1 frame, low-conf=2 frames)
  • ROI filtering, tracker_class_lock, spatial deduplication
  • Identical JSON output structure, progress reporting, perf timing
  • GPS matching, NDJSON streaming, DB saving

The only difference from YoloDetector is:
  • Uses YOLOE model with open-vocabulary text prompts
  • Post-tracking filtering: display label mapping, per-category confidence,
    5 pothole post-detection filters (shadow/size/texture/aspect/brightness)

Output classes: defected_sign_board, pothole
"""

import cv2
import json
import asyncio
import time
import logging
import torch
import os
import tempfile
from datetime import datetime
from collections import defaultdict, deque

import numpy as np

from app.detectors.base.base_detector import BaseDetector, executor
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.core.logging_config import PerfTimer, perf_logger
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.db.crud_hierarchy import find_location_by_gps
from app.helpers.yoloe_helper import (
    load_yoloe_model,
    get_display_label,
    get_conf_threshold,
    check_signboard_pole_tilt,
    _run_pothole_filters,
    ROAD_DAMAGE_CLASSES,
    ALL_CLASSES,
    YOLOE_CONF_THRESHOLD,
    YOLOE_CONF_SIGNBOARD,
    YOLOE_CONF_POTHOLE,
    NUM_TARGET_CLASSES,
)

logger = logging.getLogger(__name__)

# Performance tuning — matches YoloDetector
TRACKER = "botsort.yaml"
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
YOLOE_IMGSZ = int(os.getenv("YOLOE_IMGSZ", "640"))


class YoloeDetector(BaseDetector):
    """
    YOLOE open-vocabulary detector with BoTSORT tracking.

    Uses the same tracking + confirmation + dedup pipeline as YoloDetector.
    Detections are filtered through YOLOE-specific prompt mapping,
    per-category confidence thresholds, and pothole post-detection filters.
    """

    def __init__(self):
        # Skip BaseDetector.__init__ because it loads a YOLO model;
        # we use a YOLOE model instead, loaded lazily via yoloe_helper.
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.detection_mode = "yoloe"
        logger.info("YoloeDetector created — model will be loaded on first use")

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def is_duplicate_location(
        self,
        cx,
        cy,
        class_name,
        current_time,
        spatial_locations,
        time_threshold,
        min_distance_threshold,
    ):
        """Check if this location/class was already counted recently."""
        for existing in spatial_locations:
            prev_cx, prev_cy = existing["center"]
            prev_class = existing["class"]
            prev_time = existing["time"]

            distance = self.calculate_distance((cx, cy), (prev_cx, prev_cy))
            time_gap = current_time - prev_time

            if (
                prev_class == class_name
                and distance < min_distance_threshold
                and time_gap < time_threshold
            ):
                reason = f"{distance:.1f}px from existing, {time_gap:.2f}s ago"
                return True, reason
        return False, None

    def _load_model(self):
        """Load YOLOE model via helper (lazy singleton)."""
        self.model = load_yoloe_model()
        logger.info("YoloeDetector model loaded via yoloe_helper")

    def _ensure_model(self):
        """Ensure the model is loaded before processing."""
        if self.model is None:
            self._load_model()

    async def process_video(
        self, video_id: str, video_path: str, json_path: str, speed_kmh: int
    ):
        """Start video processing in background thread."""
        processing_status[video_id] = {"status": "processing", "progress": 0}
        self._ensure_model()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            self._process_video_blocking,
            video_id,
            video_path,
            json_path,
            speed_kmh,
            loop,
        )

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        cap = None
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id,
                    {"type": "status", "status": "processing", "progress": 0},
                ),
                loop,
            )

            gps_points = self._load_gps_data(json_path)
            process_start = time.time()

            # ── Performance accumulators ─────────────────────────────────────
            perf_timings = {
                "frame_decode": {"total": 0.0, "count": 0},
                "yoloe_inference": {"total": 0.0, "count": 0},
                "gps_coord": {"total": 0.0, "count": 0},
                "db_gps_match": {"total": 0.0, "count": 0},
                "db_bulk_write": {"total": 0.0, "count": 0},
            }

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = total_frames / fps

            logger.info(
                f"Processing [yoloe] for {video_id}: "
                f"{total_frames} frames @ {fps:.1f} FPS"
            )
            logger.info(
                f"YOLOE settings: FRAME_SKIP={FRAME_SKIP}, IMGSZ={YOLOE_IMGSZ}, "
                f"TRACKER={TRACKER}, "
                f"CONF_SIGN={YOLOE_CONF_SIGNBOARD}, CONF_POT={YOLOE_CONF_POTHOLE}"
            )

            # Adaptive parameters (same as YoloDetector)
            DETECTION_TIME_WINDOW = video_duration * 0.25
            TIME_THRESHOLD = video_duration * 0.30
            HIGH_CONFIDENCE_THRESHOLD = 0.75
            LOW_CONFIDENCE_MIN_FRAMES = 2
            MIN_DISTANCE_THRESHOLD = 120

            ROI_LEFT = 0
            ROI_RIGHT = width
            ROI_TOP = int(height * 0.05)
            ROI_BOTTOM = int(height * 0.95)

            # Tracking structures (same as YoloDetector)
            tracker_history = defaultdict(lambda: deque(maxlen=50))
            confirmed = {}
            counted_ids = {cls: set() for cls in ALL_CLASSES}
            spatial_locations = []
            tracker_class_lock = {}
            _tid_last_seen = {}

            def _confirm_detection(
                tid, class_name, frame_count, current_time, conf,
                x1, y1, x2, y2, cx, cy,
            ):
                """Helper to confirm a detection and update all tracking structures."""
                confirmed[tid] = {
                    "detection_id": tid,
                    "type": class_name,
                    "first_detected_frame": frame_count,
                    "first_detected_time": round(current_time, 2),
                    "confidence": round(float(conf), 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "vl_verified": None,
                    "vl_confidence": None,
                    "vl_category": None,
                }
                if class_name in counted_ids:
                    counted_ids[class_name].add(tid)
                spatial_locations.append(
                    {
                        "center": (cx, cy),
                        "time": current_time,
                        "class": class_name,
                    }
                )
                rejection_stats["multi_frame_pending"].discard(tid)

            def _evict_stale_trackers():
                """Remove stale entries to bound memory."""
                stale_tids = [
                    tid for tid, last_t in _tid_last_seen.items()
                    if current_time - last_t > TIME_THRESHOLD
                    and tid not in confirmed
                ]
                for tid in stale_tids:
                    tracker_class_lock.pop(tid, None)
                    tracker_history.pop(tid, None)
                    del _tid_last_seen[tid]
                if stale_tids:
                    logger.info(
                        f"Evicted {len(stale_tids)} stale tracker IDs | "
                        f"tracker_lock={len(tracker_class_lock)} "
                        f"last_seen={len(_tid_last_seen)}"
                    )

            rejection_stats = {
                "multi_frame_pending": set(),
                "spatial_duplicate": 0,
                "roi_outside": 0,
                "class_mismatch": 0,
                "prompt_filtered": 0,
                "conf_filtered": 0,
                "pothole_filtered": 0,
                "vl_mismatch": 0,
                "vl_errors": 0,
            }

            # NDJSON streaming to temp file (memory efficient)
            _ndjson_fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".ndjson",
                prefix=f"yoloe_frames_{video_id}_",
                dir=str(RESULTS_DIR),
                delete=False,
            )
            _ndjson_path = _ndjson_fd.name
            _frames_written = 0
            total_detections_count = 0
            frame_count = 0
            last_progress = 0

            while cap.isOpened():
                _t0_read = time.perf_counter()
                ret, frame = cap.read()
                _read_elapsed = time.perf_counter() - _t0_read
                if not ret:
                    break
                perf_timings["frame_decode"]["total"] += _read_elapsed
                perf_timings["frame_decode"]["count"] += 1

                frame_count += 1

                # Frame skipping — same as YoloDetector
                if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                    continue

                current_time = frame_count / fps

                # ── YOLOE inference with BoTSORT tracking ─────────────────────
                _t0_inf = time.perf_counter()
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=YOLOE_CONF_THRESHOLD,
                    tracker=TRACKER,
                    verbose=False,
                    device=self.device,
                    imgsz=YOLOE_IMGSZ,
                )
                _inf_elapsed = time.perf_counter() - _t0_inf
                perf_timings["yoloe_inference"]["total"] += _inf_elapsed
                perf_timings["yoloe_inference"]["count"] += 1

                frame_data = {"frame_id": frame_count, "detections": []}

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    names = results[0].names

                    # Extract segmentation masks for pole tilt analysis
                    masks_data = None
                    if results[0].masks is not None:
                        masks_data = results[0].masks.data.cpu().numpy()

                    for idx, (tid, cls_id, box, conf) in enumerate(
                        zip(track_ids, class_ids, boxes, confidences)
                    ):
                        tid, cls_id = int(tid), int(cls_id)
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        # Skip anything outside our target classes
                        if cls_id >= NUM_TARGET_CLASSES:
                            continue

                        prompt_name = names.get(cls_id, f"class_{cls_id}")

                        # ── YOLOE Filter 1: Map prompt → backend class ────────
                        backend_class = get_display_label(prompt_name)
                        if backend_class is None:
                            rejection_stats["prompt_filtered"] += 1
                            continue

                        # ── YOLOE Filter 2: Per-category confidence ───────────
                        min_conf = get_conf_threshold(prompt_name)
                        if conf < min_conf:
                            rejection_stats["conf_filtered"] += 1
                            continue

                        # ── YOLOE Filter 3: Pothole post-detection filters ────
                        if backend_class == "pothole":
                            skip, reason = _run_pothole_filters(
                                frame, x1, y1, x2, y2
                            )
                            if skip:
                                rejection_stats["pothole_filtered"] += 1
                                continue

                        # ── ROI filter (same as YoloDetector) ─────────────────
                        if not (
                            ROI_LEFT < cx < ROI_RIGHT
                            and ROI_TOP < cy < ROI_BOTTOM
                        ):
                            rejection_stats["roi_outside"] += 1
                            continue

                        # ── Tracker class lock (same as YoloDetector) ─────────
                        if tid in tracker_class_lock:
                            if tracker_class_lock[tid] != backend_class:
                                rejection_stats["class_mismatch"] += 1
                                continue
                        else:
                            tracker_class_lock[tid] = backend_class

                        # ── Multi-frame confirmation (same as YoloDetector) ───
                        tracker_history[tid].append(current_time)
                        _tid_last_seen[tid] = current_time
                        recent = [
                            t
                            for t in tracker_history[tid]
                            if current_time - t <= DETECTION_TIME_WINDOW
                        ]
                        min_needed = (
                            1
                            if conf >= HIGH_CONFIDENCE_THRESHOLD
                            else LOW_CONFIDENCE_MIN_FRAMES
                        )

                        if (
                            len(recent) >= min_needed
                            and tid not in confirmed
                        ):
                            # ── Spatial deduplication ─────────────────────────
                            is_dup, _ = self.is_duplicate_location(
                                cx,
                                cy,
                                backend_class,
                                current_time,
                                spatial_locations,
                                TIME_THRESHOLD,
                                MIN_DISTANCE_THRESHOLD,
                            )
                            if not is_dup:
                                _confirm_detection(
                                    tid,
                                    backend_class,
                                    frame_count,
                                    current_time,
                                    conf,
                                    x1, y1, x2, y2,
                                    cx, cy,
                                )
                                # ── Pole tilt for confirmed signboards ─────
                                if backend_class == "defected_sign_board":
                                    seg_mask = None
                                    if masks_data is not None and idx < len(masks_data):
                                        mask_raw = masks_data[idx]
                                        fh, fw = frame.shape[:2]
                                        if mask_raw.shape[:2] != (fh, fw):
                                            mask_raw = cv2.resize(
                                                mask_raw, (fw, fh),
                                                interpolation=cv2.INTER_NEAREST,
                                            )
                                        seg_mask = (mask_raw > 0.5).astype(np.uint8)
                                    tilt_angle, pole_status = check_signboard_pole_tilt(
                                        frame, (x1, y1, x2, y2), seg_mask
                                    )
                                    confirmed[tid]["pole_tilt_angle"] = round(tilt_angle, 1)
                                    confirmed[tid]["pole_status"] = pole_status
                            else:
                                rejection_stats["spatial_duplicate"] += 1
                        elif tid not in confirmed:
                            rejection_stats["multi_frame_pending"].add(tid)

                        # ── Append to frame data if confirmed ─────────────────
                        if tid in confirmed:
                            total_detections_count += 1
                            det_entry = {
                                "frame_id": frame_count,
                                "detection_id": tid,
                                "type": backend_class,
                                "prompt_name": prompt_name,
                                "confidence": round(float(conf), 3),
                                "count": {
                                    "defected_sign_board": len(
                                        counted_ids.get("defected_sign_board", set())
                                    ),
                                    "pothole": len(
                                        counted_ids.get("pothole", set())
                                    ),
                                    "road_crack": 0,
                                    "damaged_road_marking": 0,
                                    "good_sign_board": 0,
                                },
                                "bbox": {
                                    "x1": x1, "y1": y1,
                                    "x2": x2, "y2": y2,
                                },
                                "center": {"x": cx, "y": cy},
                                "area": (x2 - x1) * (y2 - y1),
                            }
                            # Include pole tilt data for signboards
                            if "pole_tilt_angle" in confirmed[tid]:
                                det_entry["pole_tilt_angle"] = confirmed[tid]["pole_tilt_angle"]
                                det_entry["pole_status"] = confirmed[tid]["pole_status"]
                            frame_data["detections"].append(det_entry)

                if frame_data["detections"]:
                    _ndjson_fd.write(json.dumps(frame_data) + "\n")
                    _frames_written += 1

                # Progress reporting
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id,
                        progress,
                        loop,
                        {
                            "unique_pothole": len(
                                counted_ids.get("pothole", set())
                            ),
                            "unique_defected_sign_board": len(
                                counted_ids.get("defected_sign_board", set())
                            ),
                            "unique_road_crack": 0,
                            "unique_damaged_road_marking": 0,
                            "unique_good_sign_board": 0,
                            "total_road_damage": sum(
                                len(counted_ids.get(c, set()))
                                for c in ROAD_DAMAGE_CLASSES
                            ),
                            "total_detections": total_detections_count,
                        },
                    )
                    last_progress = progress
                    # Periodically evict stale tracker entries
                    _evict_stale_trackers()

            process_end = time.time()

            # ── Build final results (same structure as YoloDetector) ──────────
            defected_sign_board_list = self._get_class_list(
                confirmed, "defected_sign_board", gps_points, perf_timings
            )
            pothole_list = self._get_class_list(
                confirmed, "pothole", gps_points, perf_timings
            )

            frames_with_detections = _frames_written
            detection_rate = (
                round((frames_with_detections / frame_count) * 100, 2)
                if frame_count > 0
                else 0
            )

            # Read back NDJSON frames
            frames = self._read_ndjson_frames(
                _ndjson_fd, _ndjson_path, _frames_written
            )

            results = {
                "video_id": video_id,
                "video_path": video_path,
                "detection_mode": self.detection_mode,
                "processed_at": datetime.now().isoformat(),
                "video_info": {
                    "total_frames": total_frames,
                    "fps": round(fps, 2),
                    "duration": round(video_duration, 2),
                    "width": width,
                    "height": height,
                    "resolution": f"{width}x{height}",
                },
                "summary": {
                    "total_frames": frame_count,
                    "unique_defected_sign_board": len(
                        counted_ids.get("defected_sign_board", set())
                    ),
                    "unique_pothole": len(
                        counted_ids.get("pothole", set())
                    ),
                    "unique_road_crack": 0,
                    "unique_damaged_road_marking": 0,
                    "unique_good_sign_board": 0,
                    "total_road_damage": sum(
                        len(counted_ids.get(c, set()))
                        for c in ROAD_DAMAGE_CLASSES
                    ),
                    "total_detections": total_detections_count,
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "rejection_stats": {
                    "multi_frame_pending": len(
                        rejection_stats["multi_frame_pending"]
                    ),
                    "spatial_duplicate": rejection_stats["spatial_duplicate"],
                    "class_mismatch": rejection_stats["class_mismatch"],
                    "roi_outside": rejection_stats["roi_outside"],
                    "prompt_filtered": rejection_stats["prompt_filtered"],
                    "conf_filtered": rejection_stats["conf_filtered"],
                    "pothole_filtered": rejection_stats["pothole_filtered"],
                    "vl_mismatch": 0,
                    "vl_errors": 0,
                },
                "vl_stats": None,
                "defected_sign_board_list": defected_sign_board_list,
                "pothole_list": pothole_list,
                "road_crack_list": [],
                "damaged_road_marking_list": [],
                "good_sign_board_list": [],
                "frames": frames,
            }

            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            # Save to DB
            all_detections_flat = (
                defected_sign_board_list + pothole_list
            )
            self._save_to_db(video_id, all_detections_flat, perf_timings)

            # ── Perf report ──────────────────────────────────────────────────
            total_time = process_end - process_start
            frames_processed = (
                frame_count // FRAME_SKIP if FRAME_SKIP > 1 else frame_count
            )

            def _fmt(stage_key):
                d = perf_timings[stage_key]
                total = d["total"]
                count = d["count"]
                avg_ms = (total / count * 1000) if count > 0 else 0.0
                return total, count, avg_ms

            fd_t, fd_c, fd_avg = _fmt("frame_decode")
            yi_t, yi_c, yi_avg = _fmt("yoloe_inference")
            gc_t, gc_c, gc_avg = _fmt("gps_coord")
            dg_t, dg_c, dg_avg = _fmt("db_gps_match")
            db_t, db_c, db_avg = _fmt("db_bulk_write")

            def _flag(t):
                return " ⚠️" if total_time > 0 and (t / total_time) > 0.20 else ""

            report_lines = [
                f"{'=' * 78}",
                f"  PERF REPORT — [{video_id}]  mode=yoloe",
                f"{'=' * 78}",
                f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'Frame decode (I/O)':<30} {fd_t:>10.3f} {fd_c:>7d} {fd_avg:>15.2f}{_flag(fd_t)}",
                f"  {'YOLOE inference+track':<30} {yi_t:>10.3f} {yi_c:>7d} {yi_avg:>15.2f}{_flag(yi_t)}",
                f"  {'GPS coord lookup':<30} {gc_t:>10.3f} {gc_c:>7d} {gc_avg:>15.2f}{_flag(gc_t)}",
                f"  {'DB GPS matching':<30} {dg_t:>10.3f} {dg_c:>7d} {dg_avg:>15.2f}{_flag(dg_t)}",
                f"  {'DB bulk write':<30} {db_t:>10.3f} {db_c:>7d} {db_avg:>15.2f}{_flag(db_t)}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'TOTAL pipeline time':<30} {total_time:>10.3f}s",
                f"  {'Video duration':<30} {video_duration:>10.1f}s  ({total_frames} frames @ {fps:.0f} FPS)",
                f"  {'Frames processed':<30} {frames_processed:>10d}  (FRAME_SKIP={FRAME_SKIP})",
                f"  {'Detections saved':<30} {len(all_detections_flat):>10d}",
                f"{'=' * 78}",
            ]
            report_str = "\n".join(report_lines)
            perf_logger.info(f"\n{report_str}")
            logger.info(f"\n{report_str}")

            # Mark complete
            processing_status[video_id] = {
                "status": "completed",
                "progress": 100,
            }
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id,
                    {
                        "type": "complete",
                        "status": "completed",
                        "summary": results["summary"],
                        "rejection_stats": results["rejection_stats"],
                    },
                ),
                loop,
            )
            return results

        except Exception as e:
            logger.error(f"Error processing {video_id}: {e}")
            processing_status[video_id] = {
                "status": "error",
                "message": str(e),
            }
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id, {"type": "error", "message": str(e)}
                ),
                loop,
            )
            raise
        finally:
            if cap:
                cap.release()
            try:
                if "_ndjson_fd" in dir() and not _ndjson_fd.closed:
                    _ndjson_fd.close()
                if "_ndjson_path" in dir() and os.path.exists(_ndjson_path):
                    os.remove(_ndjson_path)
                    logger.info(f"NDJSON temp file cleaned up: {_ndjson_path}")
            except OSError:
                pass
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Shared helpers (same contract as YoloDetector) ───────────────────────

    @staticmethod
    def _read_ndjson_frames(
        ndjson_fd, ndjson_path: str, expected_count: int = 0
    ) -> list:
        """Close the NDJSON temp file and read all frame lines back."""
        if not ndjson_fd.closed:
            ndjson_fd.close()
        frames = []
        try:
            with open(ndjson_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        frames.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read back NDJSON frames: {e}")
        logger.info(
            f"NDJSON read-back — {len(frames)} frames "
            f"(expected {expected_count})"
        )
        return frames

    def _get_class_list(self, confirmed, class_name, gps_points, perf_timings=None):
        """Build sorted list of detections for a given class."""
        lst = []
        for det_id, info in confirmed.items():
            if info["type"] == class_name:
                det_data = {
                    "detection_id": det_id,
                    "type": info["type"],
                    "first_detected_frame": info["first_detected_frame"],
                    "first_detected_time": info["first_detected_time"],
                    "confidence": info["confidence"],
                    "bbox": info.get("bbox", {}),
                    "vl_verified": info.get("vl_verified"),
                    "vl_confidence": info.get("vl_confidence"),
                    "vl_category": info.get("vl_category"),
                }
                # Include pole tilt data for signboard detections
                if "pole_tilt_angle" in info:
                    det_data["pole_tilt_angle"] = info["pole_tilt_angle"]
                    det_data["pole_status"] = info["pole_status"]
                if gps_points:
                    _t0 = time.perf_counter()
                    gps_coords = self.find_nearest_gps(
                        info["first_detected_time"], gps_points
                    )
                    _gps_elapsed = time.perf_counter() - _t0
                    if perf_timings is not None:
                        perf_timings["gps_coord"]["total"] += _gps_elapsed
                        perf_timings["gps_coord"]["count"] += 1
                    det_data.update(gps_coords)
                lst.append(det_data)
        return sorted(lst, key=lambda x: x["first_detected_frame"])

    def _save_to_db(self, video_id, all_detections, perf_timings=None):
        """Save detections to database."""
        db = SessionLocal()
        try:
            db_detections = []
            for det in all_detections:
                lat, lng = det.get("lat"), det.get("lng")
                project_id, package_id, location_id = None, None, None
                if lat and lng:
                    _t0 = time.perf_counter()
                    location = find_location_by_gps(db, lat, lng)
                    _gps_db_elapsed = time.perf_counter() - _t0
                    if perf_timings is not None:
                        perf_timings["db_gps_match"]["total"] += _gps_db_elapsed
                        perf_timings["db_gps_match"]["count"] += 1
                    perf_logger.info(
                        f"[{video_id}] DB GPS match | {_gps_db_elapsed:.4f}s"
                        f" | lat={lat:.5f} lng={lng:.5f}"
                        f" | {'HIT' if location else 'MISS'}"
                    )
                    if location:
                        location_id = location.id
                        package_id = location.package_id
                        if location.package:
                            project_id = location.package.project_id

                db_det = Detection(
                    video_id=video_id,
                    frame_number=det["first_detected_frame"],
                    timestamp_ms=int(det["first_detected_time"] * 1000),
                    confidence=det["confidence"],
                    detection_type=det["type"],
                    class_name=det["type"],
                    latitude=lat,
                    longitude=lng,
                    project_id=project_id,
                    package_id=package_id,
                    location_id=location_id,
                )
                db_det.set_bounding_box(det.get("bbox", {}))
                db_detections.append(db_det)

            if db_detections:
                _t0 = time.perf_counter()
                crud.create_detections_bulk(db, db_detections)
                _bulk_elapsed = time.perf_counter() - _t0
                if perf_timings is not None:
                    perf_timings["db_bulk_write"]["total"] += _bulk_elapsed
                    perf_timings["db_bulk_write"]["count"] += 1
                perf_logger.info(
                    f"[{video_id}] DB bulk write | {_bulk_elapsed:.4f}s"
                    f" | {len(db_detections)} rows"
                )
                logger.info(
                    f"Saved {len(db_detections)} detections to DB for {video_id}"
                )
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()