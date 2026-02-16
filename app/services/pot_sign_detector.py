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
from fastapi import HTTPException
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

from app.ws.websocket_manager import manager
from app.core.storage import processing_status, detection_results, RESULTS_DIR
from app.db.database import SessionLocal
from app.db import crud
from app.services.location_mapper import find_location_by_gps
from app.models.processing import ProcessingStatusEnum

logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "models/final-v1.pt"
TRACKER = "botsort.yaml"  # Using BoT-SORT for better tracking consistency
CONF_THRESHOLD = 0.50
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)

# ===================== CLASS DEFINITIONS =====================
# Road damage classes that will be counted
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}

# Classes detected but not counted as damage
EXCLUDED_CLASSES = {"good_sign_board"}

ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES


class PotSignDetector:
    def __init__(self):
        """Initialize combined pot-sign detector with YOLO model"""
        try:
            logger.info(f"Loading combined pot-sign model on device: {DEVICE}")
            self.model = YOLO(MODEL_PATH)

            if DEVICE == "cuda:0":
                self.model.to(DEVICE)
                logger.info(
                    f"Combined model loaded on GPU: {torch.cuda.get_device_name(0)}"
                )
            else:
                logger.info("Combined model loaded on CPU")

            # Warmup
            self._warmup()
            logger.info("Combined pot-sign detector ready")

        except Exception as e:
            logger.error(f"Failed to load combined pot-sign model: {e}")
            raise

    def _warmup(self):
        """Warmup model for optimal performance"""
        try:
            import numpy as np

            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, device=DEVICE)
        except:
            pass

    @staticmethod
    def find_nearest_gps(detection_time: float, gps_points: list) -> dict:
        """Find the nearest GPS coordinates for a given timestamp"""
        if not gps_points:
            return {"lat": None, "lng": None}

        # Find the GPS point with the closest timestamp
        nearest_point = min(
            gps_points, key=lambda p: abs(p.get("timestamp", 0) - detection_time)
        )
        return {"lat": nearest_point.get("lat"), "lng": nearest_point.get("lng")}

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two points"""
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
        """
        Check if this location/class was already counted recently.
        Returns (is_duplicate: bool, reason: str)
        """
        for existing in spatial_locations:
            prev_cx, prev_cy = existing["center"]
            prev_class = existing["class"]
            prev_time = existing["time"]

            # Calculate spatial distance
            distance = self.calculate_distance((cx, cy), (prev_cx, prev_cy))
            time_gap = current_time - prev_time

            # If same class, close in space, and within time window = duplicate
            if (
                prev_class == class_name
                and distance < min_distance_threshold
                and time_gap < time_threshold
            ):
                reason = f"{distance:.1f}px from existing, {time_gap:.2f}s ago"
                return True, reason

        return False, None

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        """Process video in blocking thread"""
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id, {"type": "status", "status": "processing", "progress": 0}
                ),
                loop,
            )

            # Load GPS data from JSON file
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

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = total_frames / fps

            logger.info(
                f"Processing combined pot-sign detection for {video_id}: {total_frames} frames @ {fps:.1f} FPS, duration: {video_duration:.2f}s"
            )

            # ===== ADAPTIVE PARAMETERS =====
            DETECTION_TIME_WINDOW_PERCENT = 0.25  # 25% of video duration
            TIME_THRESHOLD_PERCENT = 0.30  # 30% of video duration
            HIGH_CONFIDENCE_THRESHOLD = 0.75  # High confidence = immediate confirmation
            LOW_CONFIDENCE_MIN_FRAMES = 2  # Low confidence needs multiple frames
            MIN_DISTANCE_THRESHOLD = 120  # pixels

            DETECTION_TIME_WINDOW = video_duration * DETECTION_TIME_WINDOW_PERCENT
            TIME_THRESHOLD = video_duration * TIME_THRESHOLD_PERCENT

            # ROI: Top 5% to bottom 95% (to capture both signboards and potholes)
            ROI_LEFT = int(width * 0.0)
            ROI_RIGHT = int(width * 1.0)
            ROI_TOP = int(height * 0.05)
            ROI_BOTTOM = int(height * 0.95)

            logger.info(
                f"Adaptive params: detection_window={DETECTION_TIME_WINDOW:.2f}s, "
                f"dedup_window={TIME_THRESHOLD:.2f}s, min_distance={MIN_DISTANCE_THRESHOLD}px"
            )

            # Tracking structures
            tracker_history = defaultdict(lambda: deque(maxlen=50))
            confirmed = {}
            counted_ids = set()

            # Individual class counters
            counted_defected_sign_board = set()
            counted_pothole = set()
            counted_road_crack = set()
            counted_damaged_road_marking = set()
            counted_good_sign_board = set()

            spatial_locations = []
            tracker_class_lock = {}  # ✨ CLASS LOCKING

            # Rejection stats
            rejection_stats = {
                "multi_frame_pending": set(),
                "spatial_duplicate": 0,
                "roi_outside": 0,
                "class_mismatch": 0,  # ✨ Track class switching
            }

            results_log = {"frames": []}
            total_detections = 0
            frame_count = 0
            last_progress = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / fps

                # Run detection with tracking
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=CONF_THRESHOLD,
                    tracker=TRACKER,
                    verbose=False,
                    device=DEVICE,
                )

                frame_data = {"frame_id": frame_count, "detections": []}

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for tid, cid, box, conf in zip(
                        track_ids, class_ids, boxes, confidences
                    ):
                        tid = int(tid)
                        cid = int(cid)

                        x1, y1, x2, y2 = map(int, box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        class_name = str(self.model.names[cid])

                        # ROI check
                        in_roi = ROI_LEFT < cx < ROI_RIGHT and ROI_TOP < cy < ROI_BOTTOM
                        if not in_roi:
                            rejection_stats["roi_outside"] += 1
                            continue

                        # ✨ CLASS LOCKING: Ensure each tracker ID maintains consistent class
                        if tid in tracker_class_lock:
                            locked_class = tracker_class_lock[tid]
                            if locked_class != class_name:
                                logger.debug(
                                    f"Frame {frame_count}: ID {tid} CLASS MISMATCH - "
                                    f"Expected '{locked_class}', got '{class_name}' (REJECTED)"
                                )
                                rejection_stats["class_mismatch"] += 1
                                continue  # Skip this detection
                        else:
                            # First time seeing this ID - lock it to this class
                            tracker_class_lock[tid] = class_name

                        # Update temporal tracker
                        tracker_history[tid].append(current_time)

                        # Check how many recent detections this track has
                        recent_detections = [
                            t
                            for t in tracker_history[tid]
                            if current_time - t <= DETECTION_TIME_WINDOW
                        ]

                        # Confidence-based multi-frame requirement
                        if conf >= HIGH_CONFIDENCE_THRESHOLD:
                            MIN_FRAMES_NEEDED = 1  # High confidence = single frame OK
                        else:
                            MIN_FRAMES_NEEDED = LOW_CONFIDENCE_MIN_FRAMES

                        # Multi-frame confirmation
                        if (
                            len(recent_detections) >= MIN_FRAMES_NEEDED
                            and tid not in confirmed
                        ):
                            # Spatial deduplication check
                            is_dup, reason = self.is_duplicate_location(
                                cx,
                                cy,
                                class_name,
                                current_time,
                                spatial_locations,
                                TIME_THRESHOLD,
                                MIN_DISTANCE_THRESHOLD,
                            )

                            if not is_dup:
                                # This is a NEW confirmed detection
                                confirmed[tid] = {
                                    "detection_id": tid,
                                    "type": class_name,
                                    "first_detected_frame": frame_count,
                                    "first_detected_time": round(current_time, 2),
                                    "confidence": round(float(conf), 3),
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                }

                                # Add to appropriate list based on class type
                                if class_name == "defected_sign_board":
                                    counted_defected_sign_board.add(tid)
                                elif class_name == "pothole":
                                    counted_pothole.add(tid)
                                elif class_name == "road_crack":
                                    counted_road_crack.add(tid)
                                elif class_name == "damaged_road_marking":
                                    counted_damaged_road_marking.add(tid)
                                elif class_name == "good_sign_board":
                                    counted_good_sign_board.add(tid)

                                counted_ids.add(tid)

                                # Add to spatial locations for future deduplication
                                spatial_locations.append(
                                    {
                                        "center": (cx, cy),
                                        "time": current_time,
                                        "class": class_name,
                                    }
                                )

                                rejection_stats["multi_frame_pending"].discard(tid)

                            else:
                                # Spatial duplicate
                                rejection_stats["spatial_duplicate"] += 1

                        elif tid not in confirmed:
                            # Still waiting for multi-frame confirmation
                            rejection_stats["multi_frame_pending"].add(tid)

                        # Add to frame data (only confirmed ones)
                        if tid in confirmed:
                            total_detections += 1
                            center_x = cx
                            center_y = cy
                            area = (x2 - x1) * (y2 - y1)

                            frame_data["detections"].append(
                                {
                                    "frame_id": frame_count,
                                    "detection_id": tid,
                                    "type": class_name,
                                    "confidence": round(float(conf), 3),
                                    "count":{
                                        "defected_sign_board":len(counted_defected_sign_board),
                                        "pothole":len(counted_pothole),
                                        "road_crack":len(counted_road_crack),
                                        "damaged_road_marking":len(counted_damaged_road_marking),
                                        "good_sign_board":len(counted_good_sign_board),
                                    },
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "center": {"x": center_x, "y": center_y},
                                    "area": area,
                                }
                            )

                if frame_data["detections"]:
                    results_log["frames"].append(frame_data)

                # Progress update every 5%
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    processing_status[video_id]["progress"] = progress
                    asyncio.run_coroutine_threadsafe(
                        manager.send_message(
                            video_id,
                            {
                                "type": "progress",
                                "progress": progress,
                                "unique_defected_sign_board": len(
                                    counted_defected_sign_board
                                ),
                                "unique_pothole": len(counted_pothole),
                                "unique_road_crack": len(counted_road_crack),
                                "unique_damaged_road_marking": len(
                                    counted_damaged_road_marking
                                ),
                                "unique_good_sign_board": len(counted_good_sign_board),
                                "total_road_damage": (
                                    len(counted_defected_sign_board)
                                    + len(counted_pothole)
                                    + len(counted_road_crack)
                                    + len(counted_damaged_road_marking)
                                ),
                                "total_detections": total_detections,
                            },
                        ),
                        loop,
                    )
                    last_progress = progress

            cap.release()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Build results with GPS coordinates
            defected_sign_board_list = []
            pothole_list = []
            road_crack_list = []
            damaged_road_marking_list = []
            good_sign_board_list = []

            for tid, info in confirmed.items():
                detection_data = {
                    "detection_id": tid,
                    "type": info["type"],
                    "first_detected_frame": info["first_detected_frame"],
                    "first_detected_time": info["first_detected_time"],
                    "confidence": info["confidence"],
                    "bbox": info.get("bbox", {}),
                }

                # Add GPS coordinates if available
                if gps_points:
                    gps_coords = self.find_nearest_gps(
                        info["first_detected_time"], gps_points
                    )
                    detection_data["lat"] = gps_coords["lat"]
                    detection_data["lng"] = gps_coords["lng"]

                # Separate into individual class lists
                class_type = info["type"]
                if class_type == "defected_sign_board":
                    defected_sign_board_list.append(detection_data)
                elif class_type == "pothole":
                    pothole_list.append(detection_data)
                elif class_type == "road_crack":
                    road_crack_list.append(detection_data)
                elif class_type == "damaged_road_marking":
                    damaged_road_marking_list.append(detection_data)
                elif class_type == "good_sign_board":
                    good_sign_board_list.append(detection_data)

            # Sort all lists by first detected frame
            defected_sign_board_list = sorted(
                defected_sign_board_list, key=lambda x: x["first_detected_frame"]
            )
            pothole_list = sorted(pothole_list, key=lambda x: x["first_detected_frame"])
            road_crack_list = sorted(
                road_crack_list, key=lambda x: x["first_detected_frame"]
            )
            damaged_road_marking_list = sorted(
                damaged_road_marking_list, key=lambda x: x["first_detected_frame"]
            )
            good_sign_board_list = sorted(
                good_sign_board_list, key=lambda x: x["first_detected_frame"]
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
                "detection_config": {
                    "detection_time_window": round(DETECTION_TIME_WINDOW, 2),
                    "time_threshold": round(TIME_THRESHOLD, 2),
                    "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
                    "low_confidence_min_frames": LOW_CONFIDENCE_MIN_FRAMES,
                    "min_distance_threshold": MIN_DISTANCE_THRESHOLD,
                    "tracker": TRACKER,
                    "road_damage_classes": sorted(list(ROAD_DAMAGE_CLASSES)),
                    "excluded_classes": sorted(list(EXCLUDED_CLASSES)),
                },
                "summary": {
                    "total_frames": frame_count,
                    "unique_defected_sign_board": len(counted_defected_sign_board),
                    "unique_pothole": len(counted_pothole),
                    "unique_road_crack": len(counted_road_crack),
                    "unique_damaged_road_marking": len(counted_damaged_road_marking),
                    "unique_good_sign_board": len(counted_good_sign_board),
                    "total_road_damage": (
                        len(counted_defected_sign_board)
                        + len(counted_pothole)
                        + len(counted_road_crack)
                        + len(counted_damaged_road_marking)
                    ),
                    "total_detections": total_detections,
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

            # === Save detections to database ===
            try:
                db = SessionLocal()
                saved_counts = {
                    "defected_sign_board": 0,
                    "pothole": 0,
                    "road_crack": 0,
                    "damaged_road_marking": 0,
                    "good_sign_board": 0,
                }

                # Helper function to save a detection
                def save_detection(detection, class_name):
                    lat = detection.get("lat")
                    lng = detection.get("lng")

                    # Map GPS to location hierarchy
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

                    crud.create_detection(
                        db=db,
                        video_id=video_id,
                        frame_number=detection["first_detected_frame"],
                        timestamp_ms=int(detection["first_detected_time"] * 1000),
                        confidence=detection["confidence"],
                        detection_type=class_name,
                        class_name=class_name,
                        bounding_box=detection.get("bbox", {}),
                        latitude=lat,
                        longitude=lng,
                        project_id=project_id,
                        package_id=package_id,
                        location_id=location_id,
                    )

                # Save defected sign boards
                for detection in defected_sign_board_list:
                    save_detection(detection, "defected_sign_board")
                    saved_counts["defected_sign_board"] += 1

                # Save potholes
                for detection in pothole_list:
                    save_detection(detection, "pothole")
                    saved_counts["pothole"] += 1

                # Save road cracks
                for detection in road_crack_list:
                    save_detection(detection, "road_crack")
                    saved_counts["road_crack"] += 1

                # Save damaged road markings
                for detection in damaged_road_marking_list:
                    save_detection(detection, "damaged_road_marking")
                    saved_counts["damaged_road_marking"] += 1

                # Save good sign boards
                for detection in good_sign_board_list:
                    save_detection(detection, "good_sign_board")
                    saved_counts["good_sign_board"] += 1

                db.commit()

                total_road_damage_saved = (
                    saved_counts["defected_sign_board"]
                    + saved_counts["pothole"]
                    + saved_counts["road_crack"]
                    + saved_counts["damaged_road_marking"]
                )

                logger.info(
                    f"Saved {total_road_damage_saved} road damage detections to database for {video_id}"
                )
                logger.info(
                    f"  - Defected sign boards: {saved_counts['defected_sign_board']}"
                )
                logger.info(f"  - Potholes: {saved_counts['pothole']}")
                logger.info(f"  - Road cracks: {saved_counts['road_crack']}")
                logger.info(
                    f"  - Damaged road markings: {saved_counts['damaged_road_marking']}"
                )
                logger.info(
                    f"  - Good sign boards (info): {saved_counts['good_sign_board']}"
                )

            except Exception as db_error:
                logger.error(f"Database save error: {db_error}")
            finally:
                db.close()

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

            # Detailed logging
            logger.info("=" * 60)
            logger.info(f"5-CLASS ROAD DAMAGE DETECTION COMPLETE: {video_id}")
            logger.info(f"Total frames: {frame_count}")
            logger.info(f"Total detections: {total_detections}")
            logger.info("=== ROAD DAMAGE DETECTIONS ===")
            logger.info(f"  Defected Sign Boards: {len(counted_defected_sign_board)}")
            logger.info(f"  Potholes: {len(counted_pothole)}")
            logger.info(f"  Road Cracks: {len(counted_road_crack)}")
            logger.info(f"  Damaged Road Markings: {len(counted_damaged_road_marking)}")
            logger.info(
                f">>> TOTAL ROAD DAMAGE: {len(counted_defected_sign_board) + len(counted_pothole) + len(counted_road_crack) + len(counted_damaged_road_marking)} <<<"
            )
            logger.info("=== OTHER DETECTIONS ===")
            logger.info(
                f"  Good Sign Boards (info only): {len(counted_good_sign_board)}"
            )
            logger.info(
                f"Class mismatches rejected: {rejection_stats['class_mismatch']}"
            )
            logger.info("=" * 60)

            print(f"\n{'='*60}")
            print(f"5-CLASS ROAD DAMAGE DETECTION COMPLETE")
            print(f"{'='*60}")
            print(f"Video ID: {video_id}")
            print(f"Frames: {frame_count} | Device: {DEVICE}")
            print(f"Total detections: {total_detections}")
            print("=== ROAD DAMAGE DETECTIONS ===")
            print(f"  Defected Sign Boards: {len(counted_defected_sign_board)}")
            print(f"  Potholes: {len(counted_pothole)}")
            print(f"  Road Cracks: {len(counted_road_crack)}")
            print(f"  Damaged Road Markings: {len(counted_damaged_road_marking)}")
            print(
                f">>> TOTAL ROAD DAMAGE: {len(counted_defected_sign_board) + len(counted_pothole) + len(counted_road_crack) + len(counted_damaged_road_marking)} <<<"
            )
            print("=== OTHER DETECTIONS ===")
            print(f"  Good Sign Boards (info only): {len(counted_good_sign_board)}")
            print(
                f"Class mismatches (ID switching): {rejection_stats['class_mismatch']}"
            )
            print(f"{'='*60}\n")

            return results

        except Exception as e:
            logger.error(
                f"Error processing combined pot-sign detection for {video_id}: {e}"
            )
            processing_status[video_id] = {"status": "error", "message": str(e)}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(video_id, {"type": "error", "message": str(e)}),
                loop,
            )
            raise

    async def process_video(
        self, video_id: str, video_path: str, json_path: str, speed_kmh: int
    ):
        """Async video processing"""
        processing_status[video_id] = {"status": "processing", "progress": 0}
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

    async def get_status(self, video_id: str):
        """Get processing status"""
        if video_id not in processing_status:
            raise HTTPException(status_code=404, detail="Video ID not found")
        return processing_status[video_id]

    async def get_results(self, video_id: str):
        """Get detection results"""
        if video_id not in detection_results:
            result_file = RESULTS_DIR / f"{video_id}.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    detection_results[video_id] = json.load(f)
            else:
                raise HTTPException(status_code=404, detail="Results not found")
        return detection_results[video_id]
