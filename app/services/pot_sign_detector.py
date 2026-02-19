"""
This module processes videos to detect both potholes and signboards using combined model
"""

import cv2
import json
import asyncio
import time
import logging
import torch
import ollama
import base64
import os
from datetime import datetime
from collections import defaultdict, deque
from dotenv import load_dotenv

load_dotenv()

from app.services.base_detector import BaseDetector
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
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

# VL Verification Configuration
ENABLE_VL_VERIFICATION = os.getenv("ENABLE_VL_VERIFICATION", "true").lower() == "true"
VL_MODEL = "qwen3-vl:235b-instruct-cloud"
VL_IMAGE_MAX_SIZE = 512  # Max dimension for VL input images
VL_MIN_BBOX_SIZE = 30  # Skip VL for bboxes smaller than this


def load_api_keys():
    """Load all available API keys from environment variables"""
    api_keys = []
    i = 1

    # Try to load numbered API keys (OLLAMA_API_KEY_1, OLLAMA_API_KEY_2, etc.)
    while True:
        key = os.getenv(f"OLLAMA_API_KEY_{i}")
        if not key:
            break
        api_keys.append(key)
        i += 1

    # Fallback to single key if numbered keys not found
    if not api_keys:
        single_key = os.getenv("OLLAMA_API_KEY")
        if single_key:
            api_keys.append(single_key)

    if api_keys:
        logger.info(f"Loaded {len(api_keys)} API key(s) for VL verification")
    else:
        logger.warning("No API keys found. VL verification will be disabled.")

    return api_keys


def create_ollama_client(api_key):
    """Create Ollama client with given API key"""
    try:
        client = ollama.Client(
            host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"}
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create Ollama client: {e}")
        return None


class PotSignDetector(BaseDetector):
    def __init__(self):
        """Initialize combined pot-sign detector with YOLO model"""
        super().__init__(model_path=MODEL_PATH)

        # Load multiple API keys for rotation
        self.api_keys = []
        self.current_key_index = 0
        self.key_lock = None  # Thread-safe rotation

        if ENABLE_VL_VERIFICATION:
            self.api_keys = load_api_keys()
            if self.api_keys:
                # Import threading only if we have keys
                import threading

                self.key_lock = threading.Lock()

    def get_next_api_key(self):
        """Get next API key in rotation (thread-safe)"""
        if not self.api_keys:
            return None

        with self.key_lock:
            key = self.api_keys[self.current_key_index]
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            return key

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

    def verify_detection_with_vl(self, frame, bbox, predicted_class):
        """
        Verify a detection using VL model.

        Args:
            frame: Original video frame
            bbox: Tuple (x1, y1, x2, y2) of bounding box
            predicted_class: YOLO's predicted class name

        Returns:
            dict with verification results or None if verification fails/skipped
        """
        if not self.api_keys or not ENABLE_VL_VERIFICATION:
            return None

        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Skip if bbox is too small
        if bbox_width < VL_MIN_BBOX_SIZE or bbox_height < VL_MIN_BBOX_SIZE:
            logger.debug(f"Skipping VL for small bbox: {bbox_width}x{bbox_height}")
            return None

        try:
            # Get next API key in rotation
            api_key = self.get_next_api_key()
            if not api_key:
                return None

            # Crop detection region
            cropped = frame[y1:y2, x1:x2]

            # Resize to optimize tokens (maintain aspect ratio)
            max_dim = max(bbox_width, bbox_height)
            if max_dim > VL_IMAGE_MAX_SIZE:
                scale = VL_IMAGE_MAX_SIZE / max_dim
                new_width = int(bbox_width * scale)
                new_height = int(bbox_height * scale)
                cropped = cv2.resize(cropped, (new_width, new_height))

            # Encode to base64
            _, buffer = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_image = base64.b64encode(buffer).decode("utf-8")

            # Optimized prompt for token efficiency
            prompt = (
                "Look at this image and classify it as exactly ONE of: "
                "defected_sign_board, good_sign_board, pothole, road_crack, damaged_road_marking. "
                'If the image does NOT match any of these categories, set category to "null" '
                "and belongs_to_category to false. "
                "If it DOES match a category, set belongs_to_category to true. "
                'Respond with JSON: {"category": "name_or_null", "confidence": "high/medium/low", '
                '"belongs_to_category": true/false}'
            )

            # Create client with rotated API key
            vl_client = create_ollama_client(api_key)
            if not vl_client:
                return None

            # Call VL model (with timeout to prevent hanging)
            vl_start = time.time()
            response = vl_client.chat(
                model=VL_MODEL,
                format="json",
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\n\nRespond ONLY with valid JSON.",
                        "images": [base64_image],
                    }
                ],
                options={"temperature": 0.1}
            )
            vl_elapsed = time.time() - vl_start

            # Parse response
            vl_result = json.loads(response["message"]["content"])
            vl_result["yolo_prediction"] = predicted_class

            # Log VL result with timing
            vl_cat = vl_result.get("category")
            vl_conf = vl_result.get("confidence")
            match = "✓" if vl_cat == predicted_class else "✗"
            logger.info(
                f"VL call {match} [{vl_elapsed:.2f}s]: YOLO={predicted_class}, VL={vl_cat} "
                f"(confidence={vl_conf}, belongs={vl_result.get('belongs_to_category')})"
            )

            return vl_result

        except json.JSONDecodeError as e:
            logger.error(f"VL JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"VL verification error: {e}")
            return None

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
            video_duration = total_frames / fps

            logger.info(
                f"Processing combined pot-sign detection for {video_id}: {total_frames} frames @ {fps:.1f} FPS"
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
            vl_cache = {}  # Cache VL results by detection_id to avoid redundant calls

            rejection_stats = {
                "multi_frame_pending": set(),
                "spatial_duplicate": 0,
                "roi_outside": 0,
                "class_mismatch": 0,
                "vl_mismatch": 0,
                "vl_errors": 0,
            }

            vl_stats = {
                "total_verified": 0,
                "verified_success": 0,
                "verified_failed": 0,
                "skipped": 0,
                "vl_overrides": 0,
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
                    frame,
                    persist=True,
                    conf=CONF_THRESHOLD,
                    tracker=TRACKER,
                    verbose=False,
                    device=self.device,
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
                        tid, cid = int(tid), int(cid)
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        class_name = str(self.model.names[cid])

                        if not (
                            ROI_LEFT < cx < ROI_RIGHT and ROI_TOP < cy < ROI_BOTTOM
                        ):
                            rejection_stats["roi_outside"] += 1
                            continue

                        if tid in tracker_class_lock:
                            if tracker_class_lock[tid] != class_name:
                                rejection_stats["class_mismatch"] += 1
                                continue
                        else:
                            tracker_class_lock[tid] = class_name

                        tracker_history[tid].append(current_time)
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

                        if len(recent) >= min_needed and tid not in confirmed:
                            is_dup, _ = self.is_duplicate_location(
                                cx,
                                cy,
                                class_name,
                                current_time,
                                spatial_locations,
                                TIME_THRESHOLD,
                                MIN_DISTANCE_THRESHOLD,
                            )
                            if not is_dup:
                                vl_result = None
                                vl_verified = False
                                vl_category = None
                                vl_confidence = None
                                vl_verified = False
                                if self.api_keys and ENABLE_VL_VERIFICATION:
                                    # Check cache first
                                    if tid in vl_cache:
                                        vl_result = vl_cache[tid]
                                        vl_stats["skipped"] += 1
                                    else:
                                        # Perform VL verification
                                        vl_result = self.verify_detection_with_vl(
                                            frame, (x1, y1, x2, y2), class_name
                                        )
                                        vl_cache[tid] = vl_result

                                        if vl_result:
                                            vl_stats["total_verified"] += 1

                                    # Process VL result
                                    if vl_result:
                                        vl_category = vl_result.get("category")
                                        vl_confidence = vl_result.get("confidence")
                                        belongs_to_category = vl_result.get(
                                            "belongs_to_category", False
                                        )

                                        if (
                                            vl_category == class_name
                                            and belongs_to_category
                                        ):
                                            # Tier 1: VL agrees with YOLO — accept as-is
                                            vl_verified = True
                                            vl_stats["verified_success"] += 1
                                        elif (
                                            vl_category
                                            and vl_category != "null"
                                            and vl_category in ALL_CLASSES
                                            and belongs_to_category
                                            and vl_confidence in ("high", "medium")
                                        ):
                                            # Tier 2: VL disagrees but returns valid category with good confidence
                                            # Override YOLO's class with VL's prediction
                                            logger.info(
                                                f"VL override tid={tid}: YOLO={class_name} → VL={vl_category} "
                                                f"(conf={vl_confidence})"
                                            )
                                            class_name = vl_category
                                            tracker_class_lock[tid] = class_name
                                            vl_verified = True
                                            vl_stats["verified_success"] += 1
                                            vl_stats["vl_overrides"] += 1
                                        else:
                                            # Tier 3: VL returns null, low confidence, or belongs=false — reject
                                            vl_verified = False
                                            rejection_stats["vl_mismatch"] += 1
                                            vl_stats["verified_failed"] += 1
                                            logger.info(
                                                f"VL rejected detection tid={tid}: YOLO={class_name}, "
                                                f"VL={vl_category} (conf={vl_confidence}, belongs={belongs_to_category})"
                                            )
                                            continue  # Skip this detection
                                    else:
                                        # VL failed or was skipped, use fallback (accept YOLO prediction)
                                        if self.api_keys and ENABLE_VL_VERIFICATION:
                                            rejection_stats["vl_errors"] += 1
                                        vl_verified = True  # Fallback: trust YOLO
                                else:
                                    # VL disabled, accept all YOLO detections
                                    vl_verified = True

                                # Confirm detection (only if VL verified or VL disabled)
                                if vl_verified:
                                    confirmed[tid] = {
                                        "detection_id": tid,
                                        "type": class_name,
                                        "first_detected_frame": frame_count,
                                        "first_detected_time": round(current_time, 2),
                                        "confidence": round(float(conf), 3),
                                        "bbox": {
                                            "x1": x1,
                                            "y1": y1,
                                            "x2": x2,
                                            "y2": y2,
                                        },
                                        "vl_verified": (
                                            vl_verified
                                            if ENABLE_VL_VERIFICATION
                                            else None
                                        ),
                                        "vl_confidence": vl_confidence,
                                        "vl_category": vl_category,
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
                            else:
                                rejection_stats["spatial_duplicate"] += 1
                        elif tid not in confirmed:
                            rejection_stats["multi_frame_pending"].add(tid)

                        if tid in confirmed:
                            total_detections_count += 1
                            frame_data["detections"].append(
                                {
                                    "frame_id": frame_count,
                                    "detection_id": tid,
                                    "type": class_name,
                                    "confidence": round(float(conf), 3),
                                    "count":{
                                        "defected_sign_board":len(counted_ids["defected_sign_board"]),
                                        "pothole":len(counted_ids["pothole"]),
                                        "road_crack":len(counted_ids["road_crack"]),
                                        "damaged_road_marking":len(counted_ids["damaged_road_marking"]),
                                        "good_sign_board":len(counted_ids["good_sign_board"]),
                                    },
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "center": {"x": cx, "y": cy},
                                    "area": (x2 - x1) * (y2 - y1),
                                }
                            )

                if frame_data["detections"]:
                    results_log["frames"].append(frame_data)

                # Progress
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id,
                        progress,
                        loop,
                        {
                            "unique_pothole": len(counted_ids["pothole"]),
                            "unique_defected_sign_board": len(
                                counted_ids["defected_sign_board"]
                            ),
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
                        },
                    )
                    last_progress = progress

            # Build final results
            defected_sign_board_list = self._get_class_list(
                confirmed, "defected_sign_board", gps_points
            )
            pothole_list = self._get_class_list(confirmed, "pothole", gps_points)
            road_crack_list = self._get_class_list(confirmed, "road_crack", gps_points)
            damaged_road_marking_list = self._get_class_list(
                confirmed, "damaged_road_marking", gps_points
            )
            good_sign_board_list = self._get_class_list(
                confirmed, "good_sign_board", gps_points
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
                    "unique_good_sign_board": len(counted_ids["good_sign_board"]),
                    "total_road_damage": sum(
                        len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES
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
                    "vl_mismatch": rejection_stats["vl_mismatch"],
                    "vl_errors": rejection_stats["vl_errors"],
                },
                "vl_stats": (
                    {
                        "enabled": ENABLE_VL_VERIFICATION,
                        "total_verified": vl_stats["total_verified"],
                        "verified_success": vl_stats["verified_success"],
                        "verified_failed": vl_stats["verified_failed"],
                        "vl_overrides": vl_stats["vl_overrides"],
                        "cache_hits": vl_stats["skipped"],
                    }
                    if ENABLE_VL_VERIFICATION
                    else None
                ),
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

            all_detections_flat = (
                defected_sign_board_list
                + pothole_list
                + road_crack_list
                + damaged_road_marking_list
                + good_sign_board_list
            )
            self._save_to_db(video_id, all_detections_flat)

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
                    "vl_verified": info.get("vl_verified"),
                    "vl_confidence": info.get("vl_confidence"),
                    "vl_category": info.get("vl_category"),
                }
                if gps_points:
                    gps_coords = self.find_nearest_gps(
                        info["first_detected_time"], gps_points
                    )
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
                crud.create_detections_bulk(db, db_detections)
                logger.info(
                    f"Saved {len(db_detections)} detections to database for {video_id}"
                )
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()
