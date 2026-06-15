"""
YoloeSegDetector — YOLOE 26x Seg model for open-vocabulary road features.
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
from app.helpers.yoloe_seg_helper import (
    load_yoloe_seg_model,
    map_yoloe_seg_class,
    YOLOE_SEG_CONF,
    ALL_CLASSES,
)

logger = logging.getLogger(__name__)

TRACKER = "botsort.yaml"
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
YOLOE_IMGSZ = int(os.getenv("YOLOE_IMGSZ", "640"))


class YoloeSegDetector(BaseDetector):
    """
    Standalone detector using YOLOE-26x-seg for pothole, asphalt crack, manhole cover,
    traffic sign, street light pole, water puddle.
    """

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.detection_mode = "yoloe_seg"
        logger.info("YoloeSegDetector created")

    def _load_model(self):
        self.model = load_yoloe_seg_model()

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
                "yoloe_inference": {"total": 0.0, "count": 0},
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

            DETECTION_TIME_WINDOW = video_duration * 0.25
            TIME_THRESHOLD = video_duration * 0.30
            HIGH_CONFIDENCE_THRESHOLD = 0.50
            LOW_CONFIDENCE_MIN_FRAMES = 2
            MIN_DISTANCE_THRESHOLD = 120

            ROI_LEFT = 0
            ROI_RIGHT = width
            ROI_TOP = int(height * 0.05)
            ROI_BOTTOM = int(height * 0.95)

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

            rejection_stats = {
                "multi_frame_pending": set(),
                "spatial_duplicate": 0,
                "roi_outside": 0,
                "class_mismatch": 0,
                "class_filtered": 0,
            }

            _ndjson_fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".ndjson",
                prefix=f"yoloe_seg_{video_id}_",
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

                _t0_inf = time.perf_counter()
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=YOLOE_SEG_CONF,
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

                    for tid, cls_id, box, conf in zip(
                        track_ids, class_ids, boxes, confidences
                    ):
                        tid, cls_id = int(tid), int(cls_id)
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        raw_class_name = names.get(cls_id, f"class_{cls_id}")

                        backend_class = map_yoloe_seg_class(raw_class_name)
                        if backend_class is None:
                            rejection_stats["class_filtered"] += 1
                            continue

                        if not (
                            ROI_LEFT < cx < ROI_RIGHT
                            and ROI_TOP < cy < ROI_BOTTOM
                        ):
                            rejection_stats["roi_outside"] += 1
                            continue

                        if tid in tracker_class_lock:
                            if tracker_class_lock[tid] != backend_class:
                                rejection_stats["class_mismatch"] += 1
                                continue
                        else:
                            tracker_class_lock[tid] = backend_class

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

                        if tid in confirmed:
                            total_detections_count += 1
                            frame_data["detections"].append({
                                "frame_id": frame_count,
                                "detection_id": tid,
                                "type": backend_class,
                                "raw_class": raw_class_name,
                                "confidence": round(float(conf), 3),
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "center": {"x": cx, "y": cy},
                            })

                if frame_data["detections"]:
                    _ndjson_fd.write(json.dumps(frame_data) + "\n")
                    _frames_written += 1

                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id, progress, loop,
                        {"total_detections": total_detections_count},
                    )
                    last_progress = progress
                    _evict_stale_trackers()

            process_end = time.time()

            all_detections_flat = []
            final_lists = {}
            for cls in ALL_CLASSES:
                final_lists[f"{cls}_list"] = self._get_class_list(
                    confirmed, cls, gps_points, perf_timings
                )
                all_detections_flat.extend(final_lists[f"{cls}_list"])

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
                },
                "summary": {
                    "total_frames": frame_count,
                    "total_detections": total_detections_count,
                    "detection_rate": detection_rate,
                },
                "rejection_stats": {
                    "multi_frame_pending": len(rejection_stats["multi_frame_pending"]),
                    "spatial_duplicate": rejection_stats["spatial_duplicate"],
                    "class_mismatch": rejection_stats["class_mismatch"],
                },
                "frames": frames,
            }
            results.update(final_lists)

            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            self._save_to_db(video_id, all_detections_flat, perf_timings)

            processing_status[video_id] = {"status": "completed", "progress": 100}
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
            except OSError:
                pass
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
            pass
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
                }
                if gps_points:
                    gps_coords = self.find_nearest_gps(
                        info["first_detected_time"], gps_points
                    )
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
                    chainage = find_chainage_by_gps(
                        db, lat, lng,
                        package_id=video_package_id,
                        direction=video_direction,
                    )
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
                crud.create_detections_bulk(db, db_detections)
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()
