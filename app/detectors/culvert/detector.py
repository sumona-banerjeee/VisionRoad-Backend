"""
CulvertDetector — YOLO + BotSORT culvert / defective-culvert detection.

Detection logic follows Test/Test_culvert.py:
  - model  : models/culvert_best.pt
  - tracker: botsort.yaml
  - conf   : 0.60
  - dedup  : BotSORT track IDs — each unique track ID is counted at most once.
             No spatial-dedup step (is_duplicate_location is intentionally omitted).

JSON / DB output structure mirrors app/detectors/yolo/detector.py.
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

from app.detectors.base.base_detector import BaseDetector
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.core.logging_config import PerfTimer, perf_logger
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.db.crud_hierarchy import find_chainage_by_gps

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = r"models\culvert_best.pt"
TRACKER = "botsort.yaml"
CONF_THRESHOLD = 0.60          # >60 % as per Test_culvert.py

# Performance tuning (inherits env-var conventions from YoloDetector)
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))

# Culvert classes produced by the model
CULVERT_CLASSES = {"good_culvert", "defective_culvert"}

# Class name → colour (BGR) — kept for potential future annotation use
CLASS_COLORS = {
    "good_culvert":      (0, 255, 0),   # Green
    "defective_culvert": (0, 0, 255),   # Red
}
# ──────────────────────────────────────────────────────────────────────────────


class CulvertDetector(BaseDetector):
    """
    Culvert-specific YOLO detector.

    Uses BotSORT tracking to assign stable IDs across frames.
    Each track ID is counted at most once (first time it crosses the confidence
    threshold).  No VL / SAM3 verification; no spatial deduplication.

    Args:
        detection_mode: passed through to result JSON for traceability.
    """

    def __init__(self, detection_mode: str = "culvert_detection"):
        super().__init__(model_path=MODEL_PATH)
        self.detection_mode = detection_mode
        logger.info(f"CulvertDetector ready — mode: {self.detection_mode}")

    # ── Main processing entry-point (called by BaseDetector.process_video) ───

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

            # ── Performance accumulators ──────────────────────────────────────
            perf_timings = {
                "frame_decode":  {"total": 0.0, "count": 0},
                "yolo_inference": {"total": 0.0, "count": 0},
                "gps_coord":     {"total": 0.0, "count": 0},
                "db_gps_match":  {"total": 0.0, "count": 0},
                "db_bulk_write": {"total": 0.0, "count": 0},
            }
            # ─────────────────────────────────────────────────────────────────

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = total_frames / fps

            logger.info(
                f"CulvertDetector [{self.detection_mode}] — {video_id}: "
                f"{total_frames} frames @ {fps:.1f} FPS | "
                f"FRAME_SKIP={FRAME_SKIP} IMGSZ={YOLO_IMGSZ}"
            )

            # ── ROI (trim top/bottom 5 %) ─────────────────────────────────────
            ROI_TOP    = int(height * 0.05)
            ROI_BOTTOM = int(height * 0.95)
            ROI_LEFT   = 0
            ROI_RIGHT  = width

            # ── Tracking state ────────────────────────────────────────────────
            # Track IDs seen so far — each ID counted at most once.
            confirmed: dict[int, dict] = {}         # tid → detection info
            counted_ids = {cls: set() for cls in CULVERT_CLASSES}

            # Per-tid: how many frames has it been seen (for multi-frame gate)?
            tracker_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
            tracker_class_lock: dict[int, str] = {}  # prevent tid class flip

            # Adaptive time windows
            DETECTION_TIME_WINDOW    = video_duration * 0.25
            HIGH_CONFIDENCE_THRESHOLD = 0.80  # single-frame confirm if ≥ this
            LOW_CONFIDENCE_MIN_FRAMES = 2     # else require ≥2 frames

            rejection_stats = {
                "roi_outside":        0,
                "class_mismatch":     0,
                "multi_frame_pending": set(),
            }

            # ── NDJSON streaming (same pattern as YoloDetector) ───────────────
            _ndjson_fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".ndjson",
                prefix=f"culvert_frames_{video_id}_",
                dir=str(RESULTS_DIR),
                delete=False,
            )
            _ndjson_path = _ndjson_fd.name
            _pending_frames: list[dict] = []
            _frames_written = 0
            logger.info(f"NDJSON temp file: {_ndjson_path}")

            total_detections_count = 0
            frame_count   = 0
            last_progress = 0

            # ── Frame loop ─────────────────────────────────────────────────────
            while cap.isOpened():
                _t0_read = time.perf_counter()
                ret, frame = cap.read()
                _read_elapsed = time.perf_counter() - _t0_read
                if not ret:
                    break

                perf_timings["frame_decode"]["total"] += _read_elapsed
                perf_timings["frame_decode"]["count"] += 1
                frame_count += 1

                # Frame skipping
                if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                    continue

                current_time = frame_count / fps

                with PerfTimer("YOLO inference", video_id) as _t_yolo:
                    results = self.model.track(
                        frame,
                        persist=True,
                        conf=CONF_THRESHOLD,
                        tracker=TRACKER,
                        verbose=False,
                        device=self.device,
                        imgsz=YOLO_IMGSZ,
                    )
                perf_timings["yolo_inference"]["total"] += _t_yolo.elapsed
                perf_timings["yolo_inference"]["count"] += 1

                frame_data = {"frame_id": frame_count, "detections": []}

                if results[0].boxes.id is not None:
                    track_ids   = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids   = results[0].boxes.cls.cpu().numpy().astype(int)
                    boxes_xyxy  = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for tid, cid, box, conf in zip(
                        track_ids, class_ids, boxes_xyxy, confidences
                    ):
                        tid, cid = int(tid), int(cid)
                        x1, y1, x2, y2 = map(int, box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        class_name = str(self.model.names[cid])

                        # ── ROI filter ─────────────────────────────────────
                        if not (
                            ROI_LEFT < cx < ROI_RIGHT
                            and ROI_TOP < cy < ROI_BOTTOM
                        ):
                            rejection_stats["roi_outside"] += 1
                            continue

                        # ── Class-lock: prevent tracker ID from flipping class
                        if tid in tracker_class_lock:
                            if tracker_class_lock[tid] != class_name:
                                rejection_stats["class_mismatch"] += 1
                                continue
                        else:
                            tracker_class_lock[tid] = class_name

                        # ── Multi-frame confirmation gate ──────────────────
                        tracker_history[tid].append(current_time)
                        recent = [
                            t for t in tracker_history[tid]
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
                            # ── Confirm the detection ──────────────────────
                            confirmed[tid] = {
                                "detection_id": tid,
                                "type": class_name,
                                "first_detected_frame": frame_count,
                                "first_detected_time": round(current_time, 2),
                                "confidence": round(float(conf), 3),
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            }
                            if class_name in counted_ids:
                                counted_ids[class_name].add(tid)

                        elif tid not in confirmed:
                            rejection_stats["multi_frame_pending"].add(tid)

                        # ── Append to frame_data if confirmed ──────────────
                        if tid in confirmed:
                            frame_data["detections"].append(
                                {
                                    "frame_id": frame_count,
                                    "detection_id": tid,
                                    "type": class_name,
                                    "confidence": round(float(conf), 3),
                                    "count": {
                                        "good_culvert":      len(counted_ids["good_culvert"]),
                                        "defective_culvert": len(counted_ids["defective_culvert"]),
                                    },
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "center": {"x": cx, "y": cy},
                                    "area": (x2 - x1) * (y2 - y1),
                                }
                            )

                if frame_data["detections"]:
                    _pending_frames.append(frame_data)

                # ── Progress update every 5 % ──────────────────────────────
                progress = int((frame_count / total_frames) * 100) if total_frames else 0
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id,
                        progress,
                        loop,
                        {
                            "unique_good_culvert":      len(counted_ids["good_culvert"]),
                            "unique_defective_culvert": len(counted_ids["defective_culvert"]),
                            "total_culvert_detections": (
                                len(counted_ids["good_culvert"])
                                + len(counted_ids["defective_culvert"])
                            ),
                        },
                    )
                    last_progress = progress

            yolo_end = time.time()

            # ── Post-loop: write surviving frames to NDJSON ────────────────────
            confirmed_ids = set(confirmed.keys())
            _frames_written = 0
            total_detections_count = 0
            for _fdata in _pending_frames:
                _surviving = [
                    d for d in _fdata["detections"]
                    if d["detection_id"] in confirmed_ids
                ]
                if _surviving:
                    _fdata["detections"] = _surviving
                    _ndjson_fd.write(json.dumps(_fdata) + "\n")
                    _frames_written += 1
                    total_detections_count += len(_surviving)
            del _pending_frames

            logger.info(
                f"NDJSON write — {_frames_written} frames, "
                f"{total_detections_count} detections"
            )

            # ── Build per-class lists (with GPS) ──────────────────────────────
            good_culvert_list      = self._get_class_list(confirmed, "good_culvert",      gps_points, perf_timings)
            defective_culvert_list = self._get_class_list(confirmed, "defective_culvert", gps_points, perf_timings)

            frames_with_detections = _frames_written
            detection_rate = (
                round((frames_with_detections / frame_count) * 100, 2)
                if frame_count > 0 else 0
            )

            # ── Assemble result JSON ───────────────────────────────────────────
            results_dict = {
                "video_id":        video_id,
                "video_path":      video_path,
                "detection_mode":  self.detection_mode,
                "processed_at":    datetime.now().isoformat(),
                "video_info": {
                    "total_frames": total_frames,
                    "fps":          round(fps, 2),
                    "duration":     round(video_duration, 2),
                    "width":        width,
                    "height":       height,
                    "resolution":   f"{width}x{height}",
                },
                "summary": {
                    "total_frames":              frame_count,
                    "unique_good_culvert":        len(counted_ids["good_culvert"]),
                    "unique_defective_culvert":   len(counted_ids["defective_culvert"]),
                    "total_culvert_detections":   (
                        len(counted_ids["good_culvert"])
                        + len(counted_ids["defective_culvert"])
                    ),
                    "total_detections":           total_detections_count,
                    "frames_with_detections":     frames_with_detections,
                    "detection_rate":             detection_rate,
                },
                "rejection_stats": {
                    "roi_outside":         rejection_stats["roi_outside"],
                    "class_mismatch":      rejection_stats["class_mismatch"],
                    "multi_frame_pending": len(rejection_stats["multi_frame_pending"]),
                },
                "good_culvert_list":      good_culvert_list,
                "defective_culvert_list": defective_culvert_list,
                "frames": self._read_ndjson_frames(
                    _ndjson_fd, _ndjson_path, _frames_written
                ),
            }

            detection_results[video_id] = results_dict
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results_dict, f, indent=2)

            # ── Save to DB ────────────────────────────────────────────────────
            all_detections_flat = good_culvert_list + defective_culvert_list
            self._save_to_db(video_id, all_detections_flat, perf_timings)

            process_end = time.time()
            total_time  = process_end - process_start
            yolo_time   = yolo_end - process_start

            # ── Perf report ───────────────────────────────────────────────────
            def _fmt(key):
                d = perf_timings[key]
                t, c = d["total"], d["count"]
                return t, c, (t / c * 1000) if c else 0.0

            fd_t, fd_c, fd_avg = _fmt("frame_decode")
            yi_t, yi_c, yi_avg = _fmt("yolo_inference")
            gc_t, gc_c, gc_avg = _fmt("gps_coord")
            dg_t, dg_c, dg_avg = _fmt("db_gps_match")
            db_t, db_c, db_avg = _fmt("db_bulk_write")

            def _flag(t):
                return " ⚠️" if total_time > 0 and (t / total_time) > 0.20 else ""

            report_lines = [
                f"{'=' * 74}",
                f"  PERF REPORT — [{video_id}]  mode={self.detection_mode}",
                f"{'=' * 74}",
                f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'Frame decode (I/O)':<30} {fd_t:>10.3f} {fd_c:>7d} {fd_avg:>15.2f}{_flag(fd_t)}",
                f"  {'YOLO inference':<30} {yi_t:>10.3f} {yi_c:>7d} {yi_avg:>15.2f}{_flag(yi_t)}",
                f"  {'GPS coord lookup':<30} {gc_t:>10.3f} {gc_c:>7d} {gc_avg:>15.2f}{_flag(gc_t)}",
                f"  {'DB GPS matching':<30} {dg_t:>10.3f} {dg_c:>7d} {dg_avg:>15.2f}{_flag(dg_t)}",
                f"  {'DB bulk write':<30} {db_t:>10.3f} {db_c:>7d} {db_avg:>15.2f}{_flag(db_t)}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'TOTAL pipeline time':<30} {total_time:>10.3f}s",
                f"  {'Video duration':<30} {video_duration:>10.1f}s  ({total_frames} frames @ {fps:.0f} FPS)",
                f"  {'Detections saved':<30} {len(all_detections_flat):>10d}",
                f"{'=' * 74}",
            ]
            perf_logger.info("\n" + "\n".join(report_lines))

            processing_status[video_id] = {"status": "completed", "progress": 100}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id,
                    {
                        "type":          "complete",
                        "status":        "completed",
                        "summary":       results_dict["summary"],
                        "rejection_stats": results_dict["rejection_stats"],
                    },
                ),
                loop,
            )
            return results_dict

        except Exception as e:
            logger.error(f"CulvertDetector error processing {video_id}: {e}")
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

    # ── Helpers inherited from BaseDetector ────────────────────────────────────
    #   find_nearest_gps(), _load_gps_data(), _send_progress(), _warmup()
    #
    # ── Local helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _read_ndjson_frames(ndjson_fd, ndjson_path: str, expected_count: int = 0) -> list:
        """Close the NDJSON temp file and read all frame lines back as a list."""
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
            f"NDJSON read-back — {len(frames)} frames loaded "
            f"(expected {expected_count}) from {ndjson_path}"
        )
        return frames

    def _get_class_list(self, confirmed, class_name, gps_points, perf_timings=None) -> list:
        """Build a list of confirmed detections for a given class, with GPS coords."""
        lst = []
        gps_timestamps = (
            [p.get("timestamp", 0) for p in gps_points] if gps_points else []
        )
        for tid, info in confirmed.items():
            if info["type"] != class_name:
                continue
            det_data = {
                "detection_id":         tid,
                "type":                 info["type"],
                "first_detected_frame": info["first_detected_frame"],
                "first_detected_time":  info["first_detected_time"],
                "confidence":           info["confidence"],
                "bbox":                 info.get("bbox", {}),
            }
            if gps_points:
                _t0 = time.perf_counter()
                gps_coords = self.find_nearest_gps(
                    info["first_detected_time"], gps_points, gps_timestamps
                )
                _gps_elapsed = time.perf_counter() - _t0
                if perf_timings is not None:
                    perf_timings["gps_coord"]["total"] += _gps_elapsed
                    perf_timings["gps_coord"]["count"] += 1
                det_data.update(gps_coords)
            lst.append(det_data)
        return sorted(lst, key=lambda x: x["first_detected_frame"])

    def _save_to_db(self, video_id: str, all_detections: list, perf_timings=None):
        """Bulk-insert detections into the database."""
        db = SessionLocal()
        try:
            db_detections = []
            for det in all_detections:
                lat, lng = det.get("lat"), det.get("lng")
                project_id = package_id = chainage_id = None
                if lat and lng:
                    _t0 = time.perf_counter()
                    chainage = find_chainage_by_gps(db, lat, lng)
                    _gps_db_elapsed = time.perf_counter() - _t0
                    if perf_timings is not None:
                        perf_timings["db_gps_match"]["total"] += _gps_db_elapsed
                        perf_timings["db_gps_match"]["count"] += 1
                    if chainage:
                        chainage_id = chainage.id
                        package_id  = chainage.package_id
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
                    f"Saved {len(db_detections)} culvert detections to DB for {video_id}"
                )
        except Exception as e:
            logger.error(f"CulvertDetector DB save error: {e}")
        finally:
            db.close()
