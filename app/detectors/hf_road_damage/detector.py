"""
HfRoadDamageDetector — HuggingFace RDD2022 model for pothole & road crack only.

Uses yolo12s_RDD2022_best.pt (from rezzzq/yolo12s-road-damage-rdd2022).
Standalone detection mode — does NOT modify any existing detectors.

RDD2022 class mapping:
    D00, D10, D20 → road_crack
    D40           → pothole
    Repair        → ignored

Output classes: pothole, road_crack
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
from app.models.video import Video
from app.db.crud_hierarchy import find_chainage_by_gps
from app.helpers.hf_road_damage_helper import (
    load_hf_road_damage_model,
    map_hf_class,
    HF_ROAD_DAMAGE_CONF,
    HF_ACCEPTED_CLASSES,
)

logger = logging.getLogger(__name__)

TRACKER = "botsort.yaml"
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
HF_IMGSZ = int(os.getenv("HF_IMGSZ", "640"))

ROAD_DAMAGE_CLASSES = {"pothole", "road_crack"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES


class HfRoadDamageDetector(BaseDetector):
    """
    Standalone detector using HuggingFace RDD2022 model for pothole + road_crack.

    Uses BoTSORT tracking, multi-frame confirmation, spatial deduplication,
    GPS matching, and DB saving — same pipeline as YoloDetector.
    """

    def __init__(self):
        # Skip BaseDetector.__init__ — we load our own model via helper
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.detection_mode = "hf_road_damage"
        logger.info("HfRoadDamageDetector created — model loaded on first use")

    def _load_model(self):
        self.model = load_hf_road_damage_model()
        logger.info("HfRoadDamageDetector model loaded via hf_road_damage_helper")

    def _ensure_model(self):
        if self.model is None:
            self._load_model()

    @staticmethod
    def calculate_distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def is_duplicate_location(
        self, cx, cy, class_name, current_time,
        spatial_locations, time_threshold, min_distance_threshold,
    ):
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
                return True, f"{distance:.1f}px from existing, {time_gap:.2f}s ago"
        return False, None

    async def process_video(
        self, video_id: str, video_path: str, json_path: str, speed_kmh: int
    ):
        processing_status[video_id] = {"status": "processing", "progress": 0}
        self._ensure_model()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            self._process_video_blocking,
            video_id, video_path, json_path, speed_kmh, loop,
        )

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        cap = None

        # Reset tracker state before each video
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor = None

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

            perf_timings = {
                "frame_decode":    {"total": 0.0, "count": 0},
                "hf_inference":    {"total": 0.0, "count": 0},
                "gps_coord":       {"total": 0.0, "count": 0},
                "db_gps_match":    {"total": 0.0, "count": 0},
                "db_bulk_write":   {"total": 0.0, "count": 0},
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
                f"Processing [hf_road_damage] for {video_id}: "
                f"{total_frames} frames @ {fps:.1f} FPS"
            )
            logger.info(
                f"HF RDD2022 settings: FRAME_SKIP={FRAME_SKIP}, IMGSZ={HF_IMGSZ}, "
                f"CONF={HF_ROAD_DAMAGE_CONF}, TRACKER={TRACKER}"
            )

            # Adaptive parameters
            DETECTION_TIME_WINDOW = video_duration * 0.25
            TIME_THRESHOLD = video_duration * 0.30
            HIGH_CONFIDENCE_THRESHOLD = 0.75
            LOW_CONFIDENCE_MIN_FRAMES = 2
            MIN_DISTANCE_THRESHOLD = 120

            ROI_LEFT = 0
            ROI_RIGHT = width
            ROI_TOP = int(height * 0.05)
            ROI_BOTTOM = int(height * 0.95)

            # Tracking structures
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
                    {"center": (cx, cy), "time": current_time, "class": class_name}
                )
                rejection_stats["multi_frame_pending"].discard(tid)

            def _evict_stale_trackers():
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
                "hf_class_filtered": 0,
                "conf_filtered": 0,
            }

            _ndjson_fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".ndjson",
                prefix=f"hf_frames_{video_id}_",
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
                if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                    continue

                current_time = frame_count / fps

                # ── HF RDD2022 inference with BoTSORT tracking ─────────────
                _t0_inf = time.perf_counter()
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=HF_ROAD_DAMAGE_CONF,
                    tracker=TRACKER,
                    verbose=False,
                    device=self.device,
                    imgsz=HF_IMGSZ,
                )
                _inf_elapsed = time.perf_counter() - _t0_inf
                perf_timings["hf_inference"]["total"] += _inf_elapsed
                perf_timings["hf_inference"]["count"] += 1

                frame_data = {"frame_id": frame_count, "detections": []}

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    names = results[0].names

                    for tid, cls_id, box, conf in zip(
                        track_ids, class_ids, boxes, confidences
                    ):
                        tid, cls_id = int(tid), int(cls_id)
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        hf_class_name = names.get(cls_id, f"class_{cls_id}")

                        # ── Map RDD2022 class → backend class ──────────────
                        backend_class = map_hf_class(hf_class_name)
                        if backend_class is None:
                            rejection_stats["hf_class_filtered"] += 1
                            continue

                        # ── ROI filter ─────────────────────────────────────
                        if not (
                            ROI_LEFT < cx < ROI_RIGHT
                            and ROI_TOP < cy < ROI_BOTTOM
                        ):
                            rejection_stats["roi_outside"] += 1
                            continue

                        # ── Tracker class lock ─────────────────────────────
                        if tid in tracker_class_lock:
                            if tracker_class_lock[tid] != backend_class:
                                rejection_stats["class_mismatch"] += 1
                                continue
                        else:
                            tracker_class_lock[tid] = backend_class

                        # ── Multi-frame confirmation ───────────────────────
                        tracker_history[tid].append(current_time)
                        _tid_last_seen[tid] = current_time
                        recent = [
                            t for t in tracker_history[tid]
                            if current_time - t <= DETECTION_TIME_WINDOW
                        ]
                        min_needed = (
                            1 if conf >= HIGH_CONFIDENCE_THRESHOLD
                            else LOW_CONFIDENCE_MIN_FRAMES
                        )

                        if (
                            len(recent) >= min_needed
                            and tid not in confirmed
                        ):
                            is_dup, _ = self.is_duplicate_location(
                                cx, cy, backend_class, current_time,
                                spatial_locations, TIME_THRESHOLD,
                                MIN_DISTANCE_THRESHOLD,
                            )
                            if not is_dup:
                                _confirm_detection(
                                    tid, backend_class, frame_count,
                                    current_time, conf,
                                    x1, y1, x2, y2, cx, cy,
                                )
                            else:
                                rejection_stats["spatial_duplicate"] += 1
                        elif tid not in confirmed:
                            rejection_stats["multi_frame_pending"].add(tid)

                        # ── Append to frame data if confirmed ──────────────
                        if tid in confirmed:
                            total_detections_count += 1
                            frame_data["detections"].append({
                                "frame_id": frame_count,
                                "detection_id": tid,
                                "type": backend_class,
                                "hf_class": hf_class_name,
                                "confidence": round(float(conf), 3),
                                "count": {
                                    "pothole": len(counted_ids.get("pothole", set())),
                                    "road_crack": len(counted_ids.get("road_crack", set())),
                                },
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "center": {"x": cx, "y": cy},
                                "area": (x2 - x1) * (y2 - y1),
                            })

                if frame_data["detections"]:
                    _ndjson_fd.write(json.dumps(frame_data) + "\n")
                    _frames_written += 1

                # Progress reporting
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id, progress, loop,
                        {
                            "unique_pothole": len(counted_ids.get("pothole", set())),
                            "unique_road_crack": len(counted_ids.get("road_crack", set())),
                            "total_road_damage": sum(
                                len(counted_ids.get(c, set())) for c in ROAD_DAMAGE_CLASSES
                            ),
                            "total_detections": total_detections_count,
                        },
                    )
                    last_progress = progress
                    _evict_stale_trackers()

            process_end = time.time()

            # ── Build final results ──────────────────────────────────────────
            pothole_list = self._get_class_list(
                confirmed, "pothole", gps_points, perf_timings
            )
            road_crack_list = self._get_class_list(
                confirmed, "road_crack", gps_points, perf_timings
            )

            frames_with_detections = _frames_written
            detection_rate = (
                round((frames_with_detections / frame_count) * 100, 2)
                if frame_count > 0 else 0
            )

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
                    "unique_pothole": len(counted_ids.get("pothole", set())),
                    "unique_road_crack": len(counted_ids.get("road_crack", set())),
                    "total_road_damage": sum(
                        len(counted_ids.get(c, set())) for c in ROAD_DAMAGE_CLASSES
                    ),
                    "total_detections": total_detections_count,
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "rejection_stats": {
                    "multi_frame_pending": len(rejection_stats["multi_frame_pending"]),
                    "spatial_duplicate": rejection_stats["spatial_duplicate"],
                    "class_mismatch": rejection_stats["class_mismatch"],
                    "roi_outside": rejection_stats["roi_outside"],
                    "hf_class_filtered": rejection_stats["hf_class_filtered"],
                },
                "pothole_list": pothole_list,
                "road_crack_list": road_crack_list,
                "frames": frames,
            }

            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            # Save to DB
            all_detections_flat = pothole_list + road_crack_list
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
            hi_t, hi_c, hi_avg = _fmt("hf_inference")
            gc_t, gc_c, gc_avg = _fmt("gps_coord")
            dg_t, dg_c, dg_avg = _fmt("db_gps_match")
            db_t, db_c, db_avg = _fmt("db_bulk_write")

            def _flag(t):
                return " ⚠️" if total_time > 0 and (t / total_time) > 0.20 else ""

            report_lines = [
                f"{'=' * 78}",
                f"  PERF REPORT — [{video_id}]  mode=hf_road_damage",
                f"{'=' * 78}",
                f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'Frame decode (I/O)':<30} {fd_t:>10.3f} {fd_c:>7d} {fd_avg:>15.2f}{_flag(fd_t)}",
                f"  {'HF RDD2022 inference+track':<30} {hi_t:>10.3f} {hi_c:>7d} {hi_avg:>15.2f}{_flag(hi_t)}",
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
            processing_status[video_id] = {"status": "completed", "progress": 100}
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
            processing_status[video_id] = {"status": "error", "message": str(e)}
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

    # ── Shared helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _read_ndjson_frames(ndjson_fd, ndjson_path: str, expected_count: int = 0) -> list:
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
            f"NDJSON read-back — {len(frames)} frames (expected {expected_count})"
        )
        return frames

    def _get_class_list(self, confirmed, class_name, gps_points, perf_timings=None):
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
        db = SessionLocal()
        try:
            video = db.query(Video).filter(Video.id == video_id).first()
            video_chainage_id = video.chainage_id if video else None
            video_package_id = None
            video_project_id = None
            video_direction = None

            if video_chainage_id and video.chainage:
                video_package_id = video.chainage.package_id
                video_direction = video.chainage.direction
                if video.chainage.package:
                    video_project_id = video.chainage.package.project_id

            db_detections = []
            for det in all_detections:
                lat, lng = det.get("lat"), det.get("lng")
                project_id = video_project_id
                package_id = video_package_id
                chainage_id = video_chainage_id
                if lat and lng:
                    _t0 = time.perf_counter()
                    chainage = find_chainage_by_gps(
                        db, lat, lng,
                        package_id=video_package_id,
                        direction=video_direction,
                    )
                    _gps_db_elapsed = time.perf_counter() - _t0
                    if perf_timings is not None:
                        perf_timings["db_gps_match"]["total"] += _gps_db_elapsed
                        perf_timings["db_gps_match"]["count"] += 1
                    if chainage:
                        chainage_id = chainage.id
                        package_id = chainage.package_id
                        if chainage.package:
                            project_id = chainage.package.project_id

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
                    chainage_id=chainage_id,
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
                logger.info(
                    f"Saved {len(db_detections)} detections to DB for {video_id}"
                )
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()
