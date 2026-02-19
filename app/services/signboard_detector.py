"""
This module processes videos to detect signboards
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

# Configuration
TRACKER = "bytetrack.yaml"
CONFIDENCE_THRESHOLD = 0.6
MODEL_PATH = "models/best-board-v2.pt"

# ROI Configuration (Ratios)
ROI_TOP_RATIO = 0.10
ROI_BOTTOM_RATIO = 0.70
ROI_LEFT_RATIO = 0.0
ROI_RIGHT_RATIO = 1.0


class SignBoardDetector(BaseDetector):
    def __init__(self):
        """Initialize sign board detector with YOLO model"""
        super().__init__(model_path=MODEL_PATH)

    def detect_frame(
        self, frame, frame_id, results_log, tracker, confirmed, current_time, fps
    ):
        """Detect sign boards in a single frame with tracking"""
        h, w = frame.shape[:2]

        roi_left = int(w * ROI_LEFT_RATIO)
        roi_right = int(w * ROI_RIGHT_RATIO)
        roi_top = int(h * ROI_TOP_RATIO)
        roi_bottom = int(h * ROI_BOTTOM_RATIO)

        detections_in_frame = []

        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=CONFIDENCE_THRESHOLD,
                device=self.device,
            )

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None

                if ids is None:
                    continue

                for box, track_id, conf, class_id in zip(boxes, ids, confs, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    if not (roi_left < cx < roi_right and roi_top < cy < roi_bottom):
                        continue

                    class_name = str(self.model.names[class_id])

                    if track_id not in confirmed:
                        confirmed[track_id] = {
                            "signboard_id": track_id,
                            "type": class_name,
                            "first_detected_frame": frame_id,
                            "first_detected_time": round(current_time, 2),
                            "confidence": round(float(conf), 3),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "center": {"x": cx, "y": cy},
                        }

                    detections_in_frame.append(
                        {
                            "frame_id": frame_id,
                            "signboard_id": track_id,
                            "type": class_name,
                            "confidence": round(float(conf), 3),
                            "signboard_count":len(confirmed),  
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "center": {"x": cx, "y": cy},
                            "area": (x2 - x1) * (y2 - y1),
                        }
                    )

            if detections_in_frame:
                results_log["frames"].append(
                    {"frame_id": frame_id, "signboards": detections_in_frame}
                )

        except Exception as e:
            logger.error(f"Sign board detection error: {e}")

        return len(detections_in_frame)

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

            logger.info(
                f"Processing sign board detection for {video_id}: {total_frames} frames @ {fps:.1f} FPS"
            )

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

                n = self.detect_frame(
                    frame,
                    frame_count,
                    results_log,
                    tracker,
                    confirmed,
                    current_time,
                    fps,
                )
                total_detections_count += n

                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id,
                        progress,
                        loop,
                        {
                            "unique_signboards": len(confirmed),
                            "total_detections": total_detections_count,
                        },
                    )
                    last_progress = progress

            # Build results
            signboard_list = []
            for info in confirmed.values():
                signboard_data = {
                    "signboard_id": info["signboard_id"],
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
                    signboard_data["lat"] = gps_coords["lat"]
                    signboard_data["lng"] = gps_coords["lng"]
                signboard_list.append(signboard_data)

            signboard_list = sorted(
                signboard_list, key=lambda x: x["first_detected_frame"]
            )
            frames_with_detections = len(results_log["frames"])
            detection_rate = (
                round((frames_with_detections / frame_count) * 100, 2)
                if frame_count > 0
                else 0
            )

            results = {
                "video_id": video_id,
                "video_path": video_path,
                "detection_type": "sign-board-detection",
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
                    "unique_signboards": len(confirmed),
                    "total_detections": total_detections_count,
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "signboard_list": signboard_list,
                "frames": results_log["frames"],
            }

            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            self._save_to_db(video_id, signboard_list)

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
            logger.error(f"Error processing sign board detection for {video_id}: {e}")
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

    def _save_to_db(self, video_id, signboard_list):
        db = SessionLocal()
        try:
            db_detections = []
            for signboard in signboard_list:
                lat = signboard.get("lat")
                lng = signboard.get("lng")
                project_id, package_id, location_id = None, None, None

                if lat and lng:
                    location = find_location_by_gps(db, lat, lng)
                    if location:
                        location_id = location.id
                        package_id = location.package_id
                        if location.package:
                            project_id = location.package.project_id

                det = Detection(
                    video_id=video_id,
                    frame_number=signboard["first_detected_frame"],
                    timestamp_ms=int(signboard["first_detected_time"] * 1000),
                    confidence=signboard["confidence"],
                    detection_type="signboard",
                    class_name=signboard["type"],
                    latitude=lat,
                    longitude=lng,
                    project_id=project_id,
                    package_id=package_id,
                    location_id=location_id,
                )
                det.set_bounding_box(signboard.get("bbox", {}))
                db_detections.append(det)

            if db_detections:
                crud.create_detections_bulk(db, db_detections)
                logger.info(
                    f"Saved {len(db_detections)} signboard detections to database for {video_id}"
                )
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()
