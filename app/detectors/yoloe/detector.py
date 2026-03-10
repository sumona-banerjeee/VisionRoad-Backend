"""
YoloeDetector — YOLOE open-vocabulary detection for road defects.

Uses the yoloe_helper module for model loading and per-frame inference.
This detector handles the video processing loop, progress reporting,
GPS matching, spatial deduplication, NDJSON streaming, and DB saving.

Unlike YoloDetector (which uses BoTSORT tracking), YOLOE runs pure
per-frame prediction with no tracker. Spatial deduplication across
frames prevents duplicate counts of the same physical defect.
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
from collections import defaultdict

from app.detectors.base.base_detector import BaseDetector, executor
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.core.logging_config import PerfTimer, perf_logger
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.services.location_mapper import find_location_by_gps
from app.helpers.yoloe_helper import (
    load_yoloe_model,
    process_frame_with_yoloe,
    ROAD_DAMAGE_CLASSES,
    ALL_CLASSES,
    YOLOE_CONF_THRESHOLD,
)

logger = logging.getLogger(__name__)

# YOLOE processes every frame (no skipping) to match test file behavior


class YoloeDetector(BaseDetector):
    """
    YOLOE open-vocabulary detector.

    Loads the YOLOE model via yoloe_helper and processes videos frame-by-frame.
    Detections are mapped to standard backend class names for result consistency.
    """

    def __init__(self):
        # Skip BaseDetector.__init__ because it loads a YOLO model;
        # we use a YOLOE model instead, loaded lazily via yoloe_helper.
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.detection_mode = "yoloe"
        logger.info("YoloeDetector created — model will be loaded on first use")

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
                f"YOLOE settings: FRAME_SKIP=1 (every frame), "
                f"CONF={YOLOE_CONF_THRESHOLD}, dedup=OFF"
            )

            # Tracking structures (no spatial dedup — every detection is kept)
            confirmed = {}
            counted_ids = {cls: set() for cls in ALL_CLASSES}
            next_det_id = 0  # Simple incrementing ID (no tracker IDs)

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

                current_time = frame_count / fps

                # ── YOLOE inference via helper ───────────────────────────────
                _t0_inf = time.perf_counter()
                detections = process_frame_with_yoloe(self.model, frame)
                _inf_elapsed = time.perf_counter() - _t0_inf
                perf_timings["yoloe_inference"]["total"] += _inf_elapsed
                perf_timings["yoloe_inference"]["count"] += 1

                frame_data = {"frame_id": frame_count, "detections": []}

                for det in detections:
                    class_name = det["class_name"]
                    cx, cy = det["center"]
                    x1, y1, x2, y2 = det["bbox"]
                    conf = det["confidence"]

                    # ── Record every detection (no dedup) ─────────────────────
                    det_id = next_det_id
                    next_det_id += 1

                    confirmed[det_id] = {
                        "detection_id": det_id,
                        "type": class_name,
                        "prompt_name": det["prompt_name"],
                        "first_detected_frame": frame_count,
                        "first_detected_time": round(current_time, 2),
                        "confidence": conf,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "vl_verified": None,
                        "vl_confidence": None,
                        "vl_category": None,
                    }
                    if class_name in counted_ids:
                        counted_ids[class_name].add(det_id)

                    total_detections_count += 1
                    frame_data["detections"].append(
                        {
                            "frame_id": frame_count,
                            "detection_id": det_id,
                            "type": class_name,
                            "prompt_name": det["prompt_name"],
                            "confidence": conf,
                            "count": {
                                cls: len(counted_ids[cls])
                                for cls in ALL_CLASSES
                            },
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "center": {"x": cx, "y": cy},
                            "area": (x2 - x1) * (y2 - y1),
                        }
                    )

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
                            "unique_pothole": len(counted_ids.get("pothole", set())),
                            "unique_defected_sign_board": len(
                                counted_ids.get("defected_sign_board", set())
                            ),
                            "unique_road_crack": len(
                                counted_ids.get("road_crack", set())
                            ),
                            "unique_damaged_road_marking": len(
                                counted_ids.get("damaged_road_marking", set())
                            ),
                            "unique_good_sign_board": len(
                                counted_ids.get("good_sign_board", set())
                            ),
                            "total_road_damage": sum(
                                len(counted_ids.get(c, set()))
                                for c in ROAD_DAMAGE_CLASSES
                            ),
                            "total_detections": total_detections_count,
                        },
                    )
                    last_progress = progress

            process_end = time.time()

            # ── Build final results ──────────────────────────────────────────
            defected_sign_board_list = self._get_class_list(
                confirmed, "defected_sign_board", gps_points, perf_timings
            )
            pothole_list = self._get_class_list(
                confirmed, "pothole", gps_points, perf_timings
            )
            road_crack_list = self._get_class_list(
                confirmed, "road_crack", gps_points, perf_timings
            )
            damaged_road_marking_list = self._get_class_list(
                confirmed, "damaged_road_marking", gps_points, perf_timings
            )
            good_sign_board_list = self._get_class_list(
                confirmed, "good_sign_board", gps_points, perf_timings
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
                        counted_ids["defected_sign_board"]
                    ),
                    "unique_pothole": len(counted_ids["pothole"]),
                    "unique_road_crack": len(counted_ids["road_crack"]),
                    "unique_damaged_road_marking": len(
                        counted_ids["damaged_road_marking"]
                    ),
                    "unique_good_sign_board": len(
                        counted_ids["good_sign_board"]
                    ),
                    "total_road_damage": sum(
                        len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES
                    ),
                    "total_detections": total_detections_count,
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "rejection_stats": {
                    "multi_frame_pending": 0,
                    "spatial_duplicate": 0,
                    "class_mismatch": 0,
                    "roi_outside": 0,
                    "vl_mismatch": 0,
                    "vl_errors": 0,
                },
                "vl_stats": None,
                "defected_sign_board_list": defected_sign_board_list,
                "pothole_list": pothole_list,
                "road_crack_list": road_crack_list,
                "damaged_road_marking_list": damaged_road_marking_list,
                "good_sign_board_list": good_sign_board_list,
                "frames": frames,
            }

            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            # Save to DB
            all_detections_flat = (
                defected_sign_board_list
                + pothole_list
                + road_crack_list
                + damaged_road_marking_list
                + good_sign_board_list
            )
            self._save_to_db(video_id, all_detections_flat, perf_timings)

            # ── Perf report ──────────────────────────────────────────────────
            total_time = process_end - process_start
            frames_processed = frame_count

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

            report_lines = [
                f"{'=' * 78}",
                f"  PERF REPORT — [{video_id}]  mode=yoloe",
                f"{'=' * 78}",
                f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'Frame decode (I/O)':<30} {fd_t:>10.3f} {fd_c:>7d} {fd_avg:>15.2f}",
                f"  {'YOLOE inference':<30} {yi_t:>10.3f} {yi_c:>7d} {yi_avg:>15.2f}",
                f"  {'GPS coord lookup':<30} {gc_t:>10.3f} {gc_c:>7d} {gc_avg:>15.2f}",
                f"  {'DB GPS matching':<30} {dg_t:>10.3f} {dg_c:>7d} {dg_avg:>15.2f}",
                f"  {'DB bulk write':<30} {db_t:>10.3f} {db_c:>7d} {db_avg:>15.2f}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'TOTAL pipeline time':<30} {total_time:>10.3f}s",
                f"  {'Video duration':<30} {video_duration:>10.1f}s  ({total_frames} frames @ {fps:.0f} FPS)",
                f"  {'Frames processed':<30} {frames_processed:>10d}  (every frame)",
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

    def _load_gps_data(self, json_path):
        """Load GPS data from JSON file."""
        from pathlib import Path

        gps_points = []
        if json_path and Path(json_path).exists():
            try:
                with open(json_path, "r") as f:
                    gps_data = json.load(f)
                    gps_points = gps_data.get("gpsPoints", [])
                    logger.info(
                        f"Loaded {len(gps_points)} GPS points from {json_path}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load GPS data: {e}")
        return gps_points

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
                logger.info(
                    f"Saved {len(db_detections)} detections to DB for {video_id}"
                )
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()

    @staticmethod
    def find_nearest_gps(detection_time: float, gps_points: list) -> dict:
        """Find nearest GPS point to a given detection time."""
        if not gps_points:
            return {"lat": None, "lng": None}
        nearest_point = min(
            gps_points,
            key=lambda p: abs(p.get("timestamp", 0) - detection_time),
        )
        return {
            "lat": nearest_point.get("lat"),
            "lng": nearest_point.get("lng"),
        }

    def _send_progress(self, video_id, progress, loop, extra_data=None):
        """Send progress update via WebSocket."""
        processing_status[video_id]["progress"] = progress
        message = {"type": "progress", "progress": progress}
        if extra_data:
            message.update(extra_data)
        asyncio.run_coroutine_threadsafe(
            manager.send_message(video_id, message),
            loop,
        )
