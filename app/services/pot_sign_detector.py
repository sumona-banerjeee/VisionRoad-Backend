"""
This module processes videos to detect both potholes and signboards using combined model
"""

import cv2
import json
import asyncio
import logging
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

from app.services.base_detector import BaseDetector
from app.ws.websocket_manager import manager
from app.core.storage import processing_status, detection_results, RESULTS_DIR
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.services.location_mapper import find_location_by_gps

logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "models/final-v1.pt"
TRACKER = "botsort.yaml"
CONF_THRESHOLD = 0.50

# Road damage classes
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}
EXCLUDED_CLASSES = {"good_sign_board"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES

class PotSignDetector(BaseDetector):
    def __init__(self):
        """Initialize combined pot-sign detector with YOLO model"""
        super().__init__(model_path=MODEL_PATH)

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def is_duplicate_location(self, cx, cy, class_name, current_time, spatial_locations, time_threshold, min_distance_threshold):
        """Check if this location/class was already counted recently."""
        for existing in spatial_locations:
            prev_cx, prev_cy = existing["center"]
            prev_class = existing["class"]
            prev_time = existing["time"]

            distance = self.calculate_distance((cx, cy), (prev_cx, prev_cy))
            time_gap = current_time - prev_time

            if prev_class == class_name and distance < min_distance_threshold and time_gap < time_threshold:
                reason = f"{distance:.1f}px from existing, {time_gap:.2f}s ago"
                return True, reason
        return False, None

    def _process_video_blocking(self, video_id: str, video_path: str, json_path: str, speed: int, loop):
        cap = None
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(video_id, {"type": "status", "status": "processing", "progress": 0}),
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
            video_duration = total_frames / fps

            logger.info(f"Processing combined pot-sign detection for {video_id}: {total_frames} frames @ {fps:.1f} FPS")

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
            
            rejection_stats = {
                "multi_frame_pending": set(),
                "spatial_duplicate": 0,
                "roi_outside": 0,
                "class_mismatch": 0,
            }

            results_log = {"frames": []}
            total_detections_count = 0
            frame_count = 0
            last_progress = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / fps

                results = self.model.track(
                    frame, persist=True, conf=CONF_THRESHOLD, tracker=TRACKER, verbose=False, device=self.device
                )

                frame_data = {"frame_id": frame_count, "detections": []}

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for tid, cid, box, conf in zip(track_ids, class_ids, boxes, confidences):
                        tid, cid = int(tid), int(cid)
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        class_name = str(self.model.names[cid])

                        if not (ROI_LEFT < cx < ROI_RIGHT and ROI_TOP < cy < ROI_BOTTOM):
                            rejection_stats["roi_outside"] += 1
                            continue

                        if tid in tracker_class_lock:
                            if tracker_class_lock[tid] != class_name:
                                rejection_stats["class_mismatch"] += 1
                                continue
                        else:
                            tracker_class_lock[tid] = class_name

                        tracker_history[tid].append(current_time)
                        recent = [t for t in tracker_history[tid] if current_time - t <= DETECTION_TIME_WINDOW]
                        min_needed = 1 if conf >= HIGH_CONFIDENCE_THRESHOLD else LOW_CONFIDENCE_MIN_FRAMES

                        if len(recent) >= min_needed and tid not in confirmed:
                            is_dup, _ = self.is_duplicate_location(cx, cy, class_name, current_time, spatial_locations, TIME_THRESHOLD, MIN_DISTANCE_THRESHOLD)
                            if not is_dup:
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
                                spatial_locations.append({"center": (cx, cy), "time": current_time, "class": class_name})
                                rejection_stats["multi_frame_pending"].discard(tid)
                            else:
                                rejection_stats["spatial_duplicate"] += 1
                        elif tid not in confirmed:
                            rejection_stats["multi_frame_pending"].add(tid)

                        if tid in confirmed:
                            total_detections_count += 1
                            frame_data["detections"].append({
                                "frame_id": frame_count,
                                "detection_id": tid,
                                "type": class_name,
                                "confidence": round(float(conf), 3),
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "center": {"x": cx, "y": cy},
                                "area": (x2 - x1) * (y2 - y1),
                            })

                if frame_data["detections"]:
                    results_log["frames"].append(frame_data)

                # Progress
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(video_id, progress, loop, {
                        "unique_pothole": len(counted_ids["pothole"]),
                        "unique_defected_sign_board": len(counted_ids["defected_sign_board"]),
                        "unique_road_crack": len(counted_ids["road_crack"]),
                        "unique_damaged_road_marking": len(counted_ids["damaged_road_marking"]),
                        "unique_good_sign_board": len(counted_ids["good_sign_board"]),
                        "total_road_damage": sum(len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES),
                        "total_detections": total_detections_count,
                    })
                    last_progress = progress

            # Build final results
            defected_sign_board_list = self._get_class_list(confirmed, "defected_sign_board", gps_points)
            pothole_list = self._get_class_list(confirmed, "pothole", gps_points)
            road_crack_list = self._get_class_list(confirmed, "road_crack", gps_points)
            damaged_road_marking_list = self._get_class_list(confirmed, "damaged_road_marking", gps_points)
            good_sign_board_list = self._get_class_list(confirmed, "good_sign_board", gps_points)

            frames_with_detections = len(results_log["frames"])
            detection_rate = round((frames_with_detections / frame_count) * 100, 2) if frame_count > 0 else 0

            results = {
                "video_id": video_id,
                "video_path": video_path,
                "detection_type": "pot-sign-detection",
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
                    "unique_defected_sign_board": len(counted_ids["defected_sign_board"]),
                    "unique_pothole": len(counted_ids["pothole"]),
                    "unique_road_crack": len(counted_ids["road_crack"]),
                    "unique_damaged_road_marking": len(counted_ids["damaged_road_marking"]),
                    "unique_good_sign_board": len(counted_ids["good_sign_board"]),
                    "total_road_damage": sum(len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES),
                    "total_detections": total_detections_count,
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "rejection_stats": {
                    "multi_frame_pending": len(rejection_stats["multi_frame_pending"]),
                    "spatial_duplicate": rejection_stats["spatial_duplicate"],
                    "class_mismatch": rejection_stats["class_mismatch"],
                    "roi_outside": rejection_stats["roi_outside"],
                },
                "defected_sign_board_list": defected_sign_board_list,
                "pothole_list": pothole_list,
                "road_crack_list": road_crack_list,
                "damaged_road_marking_list": damaged_road_marking_list,
                "good_sign_board_list": good_sign_board_list,
                "frames": results_log["frames"],
            }

            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            all_detections_flat = defected_sign_board_list + pothole_list + road_crack_list + damaged_road_marking_list + good_sign_board_list
            self._save_to_db(video_id, all_detections_flat)

            processing_status[video_id] = {"status": "completed", "progress": 100}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(video_id, {"type": "complete", "status": "completed", "summary": results["summary"], "rejection_stats": results["rejection_stats"]}),
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

    def _get_class_list(self, confirmed, class_name, gps_points):
        lst = []
        for tid, info in confirmed.items():
            if info["type"] == class_name:
                det_data = {
                    "detection_id": tid,
                    "type": info["type"],
                    "first_detected_frame": info["first_detected_frame"],
                    "first_detected_time": info["first_detected_time"],
                    "confidence": info["confidence"],
                    "bbox": info.get("bbox", {}),
                }
                if gps_points:
                    gps_coords = self.find_nearest_gps(info["first_detected_time"], gps_points)
                    det_data.update(gps_coords)
                lst.append(det_data)
        return sorted(lst, key=lambda x: x["first_detected_frame"])

    def _save_to_db(self, video_id, all_detections):
        db = SessionLocal()
        try:
            db_detections = []
            for det in all_detections:
                lat, lng = det.get("lat"), det.get("lng")
                project_id, package_id, location_id = None, None, None
                if lat and lng:
                    location = find_location_by_gps(db, lat, lng)
                    if location:
                        location_id, package_id = location.id, location.package_id
                        if location.package: project_id = location.package.project_id

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
                crud.create_detections_bulk(db, db_detections)
                logger.info(f"Saved {len(db_detections)} detections to database for {video_id}")
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()
