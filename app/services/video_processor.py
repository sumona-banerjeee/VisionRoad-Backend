"""
This module processes videos to detect potholes
"""

import cv2
import json
import asyncio
import logging
import torch
from datetime import datetime
from collections import defaultdict, deque

from app.services.base_detector import BaseDetector
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.services.location_mapper import find_location_by_gps
from app.ws.websocket_manager import manager

logger = logging.getLogger(__name__)

# Configuration constants
MODEL_PATH = "models/best.pt"
TRACKER = "bytetrack.yaml"
MIN_DETECTION_FRAMES = 3
DETECTION_TIME_WINDOW = 1.0
CONFIDENCE_THRESHOLD = 0.80


class VideoProcessor(BaseDetector):
    def __init__(self):
        """Initialize video processor with YOLO model"""
        super().__init__(model_path=MODEL_PATH)

    @staticmethod
    def get_adaptive_params(speed):
        """Get adaptive parameters based on speed"""
        if speed < 30:
            return {"roi_ratio": 0.50, "conf": 0.70}
        elif speed < 60:
            return {"roi_ratio": 0.65, "conf": 0.70}
        else:
            return {"roi_ratio": 0.75, "conf": 0.22}

    def detect_frame(
        self, frame, frame_id, results_log, tracker, confirmed, current_time, speed
    ):
        """Detect potholes in a single frame with tracking"""
        h, w = frame.shape[:2]
        params = self.get_adaptive_params(speed)

        # ROI extraction
        roi_y = int(h * (1 - params["roi_ratio"]))
        roi = frame[roi_y:h, :]

        detections_in_frame = []
        count = 0
        new_confirmed = 0

        try:
            results = self.model.track(
                roi,
                conf=params["conf"],
                tracker=TRACKER,
                persist=True,
                verbose=False,
                device=self.device,
                imgsz=640,
            )

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None

                if ids is None:
                    continue

                for box, track_id, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)

                    y1_full, y2_full = y1 + roi_y, y2 + roi_y
                    tracker[track_id].append(current_time)

                    recent = [
                        t
                        for t in tracker[track_id]
                        if current_time - t <= DETECTION_TIME_WINDOW
                    ]

                    if (
                        len(recent) >= MIN_DETECTION_FRAMES
                        and track_id not in confirmed
                    ):
                        confirmed[track_id] = {
                            "frame": frame_id,
                            "time": current_time,
                            "conf": conf,
                            "bbox": {"x1": x1, "y1": y1_full, "x2": x2, "y2": y2_full},
                        }
                        new_confirmed = 1

                    if track_id in confirmed:
                        count += 1
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1_full + y2_full) / 2)
                        area = (x2 - x1) * (y2_full - y1_full)

                        detections_in_frame.append(
                            {
                                "frame_id": frame_id,
                                "pothole_id": track_id,
                                "type": "pothole",
                                "confidence": round(float(conf), 3),
                                "pothole_count":len(confirmed),
                                "bbox": {
                                    "x1": x1,
                                    "y1": y1_full,
                                    "x2": x2,
                                    "y2": y2_full,
                                },
                                "center": {"x": center_x, "y": center_y},
                                "area": area,
                            }
                        )

            if detections_in_frame:
                results_log["frames"].append(
                    {
                        "frame_id": frame_id,
                        "speed_kmh": speed,
                        "roi_ratio": params["roi_ratio"],
                        "potholes": detections_in_frame,
                    }
                )

        except Exception as e:
            logger.error(f"Detection error: {e}")

        return count, new_confirmed

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        cap = None
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id, {"type": "status", "status": "processing", "progress": 0}
                ),
                loop,
            )

            gps_points = self._load_gps_data(json_path)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Processing {video_id}: {total_frames} frames @ {fps:.1f} FPS")

            results_log = {"frames": []}
            tracker = defaultdict(lambda: deque(maxlen=20))
            confirmed = {}
            total_detections_count = 0
            frame_count = 0
            last_progress = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / fps

                n, _ = self.detect_frame(
                    frame,
                    frame_count,
                    results_log,
                    tracker,
                    confirmed,
                    current_time,
                    speed,
                )
                total_detections_count += n

                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id,
                        progress,
                        loop,
                        {
                            "unique_potholes": len(confirmed),
                            "total_detections": total_detections_count,
                        },
                    )
                    last_progress = progress

            # Build results and save
            pothole_list = []
            for pid, info in confirmed.items():
                pothole_data = {
                    "pothole_id": int(pid),
                    "first_detected_frame": info["frame"],
                    "first_detected_time": round(info["time"], 2),
                    "confidence": round(float(info["conf"]), 3),
                    "bbox": info.get("bbox", {}),
                }
                if gps_points:
                    gps_coords = self.find_nearest_gps(info["time"], gps_points)
                    pothole_data["lat"] = gps_coords["lat"]
                    pothole_data["lng"] = gps_coords["lng"]
                pothole_list.append(pothole_data)

            pothole_list = sorted(pothole_list, key=lambda x: x["first_detected_frame"])
            frames_with_detections = len(results_log["frames"])
            detection_rate = (
                round((frames_with_detections / frame_count) * 100, 2)
                if frame_count > 0
                else 0
            )

            results = {
                "video_id": video_id,
                "video_path": video_path,
                "speed_kmh": speed,
                "processed_at": datetime.now().isoformat(),
                "video_info": {
                    "total_frames": total_frames,
                    "fps": round(fps, 2),
                    "duration": round(total_frames / fps, 2),
                    "width": width,
                    "height": height,
                    "resolution": f"{width}x{height}",
                },
                "summary": {
                    "total_frames": frame_count,
                    "unique_potholes": len(confirmed),
                    "total_detections": total_detections_count,
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "pothole_list": pothole_list,
                "frames": results_log["frames"],
            }

            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            self._save_to_db(video_id, pothole_list)

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
                manager.send_message(video_id, {"type": "error", "message": str(e)}),
                loop,
            )
            raise
        finally:
            if cap:
                cap.release()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _save_to_db(self, video_id, pothole_list):
        db = SessionLocal()
        try:
            db_detections = []
            for pothole in pothole_list:
                lat = pothole.get("lat")
                lng = pothole.get("lng")
                project_id = None
                package_id = None
                location_id = None

                if lat and lng:
                    location = find_location_by_gps(db, lat, lng)
                    if location:
                        location_id = location.id
                        package_id = location.package_id
                        if location.package:
                            project_id = location.package.project_id

                det = Detection(
                    video_id=video_id,
                    frame_number=pothole["first_detected_frame"],
                    timestamp_ms=int(pothole["first_detected_time"] * 1000),
                    confidence=pothole["confidence"],
                    detection_type="pothole",
                    class_name="pothole",
                    latitude=lat,
                    longitude=lng,
                    project_id=project_id,
                    package_id=package_id,
                    location_id=location_id,
                )
                det.set_bounding_box(pothole.get("bbox", {}))
                db_detections.append(det)

            if db_detections:
                crud.create_detections_bulk(db, db_detections)
                logger.info(
                    f"Saved {len(db_detections)} pothole detections to database for {video_id}"
                )
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()
