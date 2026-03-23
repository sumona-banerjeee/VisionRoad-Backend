import cv2
import json
import asyncio
import time
import logging
import os
import re
from datetime import datetime
from collections import defaultdict
import google.generativeai as genai
import tempfile

from app.detectors.base.base_detector import BaseDetector
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.models.video import Video
from app.db.crud_hierarchy import find_chainage_by_gps

logger = logging.getLogger(__name__)

genai.configure(api_key="AIzaSyBsVBv6yyqftOFYa-apHZeWIFWswECYpP8")

GEMINI_TO_DB_CLASS = {
    "defective signboard": "defected_sign_board",
    "pothole": "pothole",
    "faded road marking": "damaged_road_marking",
    "road crack": "road_crack",
    "defective culvert": "defective_culvert",
    "bad drainage issue": "drain_issue",
}

ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
    "drain_issue",
    "defective_culvert"
}

ALL_CLASSES = ROAD_DAMAGE_CLASSES | {"good_sign_board", "good_culvert"}

class GeminiDetector(BaseDetector):
    def __init__(self, detection_mode="gemini"):
        self.detection_mode = detection_mode
        BaseDetector.__init__(self, model_path="")

    def _load_model(self):
        logger.info("Initializing Gemini API client")
        self.model = genai.GenerativeModel("gemini-robotics-er-1.5-preview")

    def _process_video_blocking(self, video_id: str, video_path: str, json_path: str, speed: int, loop):
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(video_id, {"type": "status", "status": "processing", "progress": 0}),
                loop,
            )

            gps_points = self._load_gps_data(json_path)
            process_start = time.time()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = total_frames / fps
            
            FRAME_SKIP = 1  # User specifically requested frame by frame
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            BATCH_SIZE = 15
            batch_frames = []
            batch_frame_indices = []

            frame_count = 0
            
            def compute_iou(boxA, boxB):
                xA = max(boxA['x1'], boxB['x1'])
                yA = max(boxA['y1'], boxB['y1'])
                xB = min(boxA['x2'], boxB['x2'])
                yB = min(boxA['y2'], boxB['y2'])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
                boxBArea = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])
                return interArea / float(boxAArea + boxBArea - interArea + 1e-5)
            
            confirmed = {}
            counted_ids = {cls: set() for cls in ALL_CLASSES}
            _pending_frames = [] # type: list[dict]
            
            last_seen_time = {}
            total_detections_count = [0]
            
            prompt = """
You are a precision road inspection AI.

Analyze the images to find instances of ONLY these defective categories:
- defective signboard
- pothole
- faded road marking
- road crack
- defective culvert
- bad drainage issue

Do NOT return any good/healthy items. Return ONLY defects.

For EVERY defect found, return its bounding box in normalized format [ymin, xmin, ymax, xmax] scaled strictly from 0 to 1000.

Return ONLY a JSON array with exactly this format per defect:
[
  {
    "image_index": number (which input image this defect was found in, 1-indexed),
    "label": "one of the defective categories",
    "box": [ymin, xmin, ymax, xmax]
  }
]

If an image has no defects, do not output anything for that image.
Do not explain anything. Return raw JSON array only.
"""
            def _process_batch(b_frames, b_indices):
                if not b_frames:
                    return

                contents = [prompt]
                for f in b_frames:
                    _, buffer = cv2.imencode('.jpg', f)
                    contents.append({
                        "mime_type": "image/jpeg",
                        "data": buffer.tobytes()
                    })

                try:
                    response = self.model.generate_content(contents)
                    match = re.search(r'\[.*\]', response.text, re.DOTALL)
                    if match:
                        data = json.loads(match.group())
                        for item in data:
                            idx = item.get("image_index", 1) - 1
                            if 0 <= idx < len(b_indices):
                                actual_frame_count = b_indices[idx]
                                current_time = actual_frame_count / fps
                                label = item.get("label", "")
                                if label in GEMINI_TO_DB_CLASS:
                                    db_class = GEMINI_TO_DB_CLASS[label]
                                    
                                    box = item.get("box", [0, 0, 1000, 1000])
                                    try:
                                        ymin, xmin, ymax, xmax = box
                                    except ValueError:
                                        ymin, xmin, ymax, xmax = [0, 0, 1000, 1000]

                                    y1 = int((ymin / 1000) * height)
                                    x1 = int((xmin / 1000) * width)
                                    y2 = int((ymax / 1000) * height)
                                    x2 = int((xmax / 1000) * width)
                                    
                                    # Ensure coordinates are valid and bounded correctly
                                    y1, y2 = min(y1, y2), max(y1, y2)
                                    x1, x2 = min(x1, x2), max(x1, x2)
                                    
                                    bbox_coords = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                                    matched_tid = None
                                    
                                    # Simple tracker assignment based on IoU across frames
                                    for tid, info in confirmed.items():
                                        if info["type"] == db_class:
                                            # Check that the object hasn't been lost for too long
                                            if current_time - info.get("last_seen_time", -999) < 2.0:
                                                iou = compute_iou(info["bbox"], bbox_coords)
                                                if iou > 0.2:
                                                    matched_tid = tid
                                                    break

                                    if matched_tid is None:
                                        # New object logic
                                        matched_tid = len(confirmed) + 1
                                        counted_ids[db_class].add(matched_tid)
                                        confirmed[matched_tid] = {
                                            "detection_id": matched_tid,
                                            "type": db_class,
                                            "first_detected_frame": actual_frame_count,
                                            "first_detected_time": current_time,
                                            "last_seen_time": current_time,
                                            "confidence": 1.0,
                                            "bbox": bbox_coords,
                                        }
                                    else:
                                        # Update existing object logic
                                        confirmed[matched_tid]["last_seen_time"] = current_time
                                        confirmed[matched_tid]["bbox"] = bbox_coords
                                        
                                    det_entry = {
                                        "frame_id": actual_frame_count,
                                        "detection_id": matched_tid,
                                        "type": db_class,
                                        "confidence": 1.0,
                                        "count": {c: len(counted_ids.get(c, set())) for c in ALL_CLASSES},
                                        "bbox": bbox_coords,
                                        "center": {"x": (x1+x2)//2, "y": (y1+y2)//2},
                                        "area": max(0, x2-x1) * max(0, y2-y1),
                                    }
                                    
                                    # Merge into the pending frames cleanly
                                    if _pending_frames and _pending_frames[-1]["frame_id"] == actual_frame_count:
                                        _pending_frames[-1]["detections"].append(det_entry)
                                    else:
                                        _pending_frames.append({
                                            "frame_id": actual_frame_count,
                                            "detections": [det_entry]
                                        })
                                    total_detections_count[0] += 1
                except Exception as e:
                    logger.error(f"Gemini API error: {e}")
                    
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                if frame_count % FRAME_SKIP != 0:
                    continue
                    
                batch_frames.append(frame)
                batch_frame_indices.append(frame_count)
                
                if len(batch_frames) >= BATCH_SIZE:
                    _process_batch(batch_frames, batch_frame_indices)
                    batch_frames = []
                    batch_frame_indices = []
                    
                    progress = int((frame_count / total_frames) * 100)
                    self._send_progress(video_id, progress, loop, {
                        f"unique_{k}": len(counted_ids[k]) for k in ALL_CLASSES
                    })

            # process remaining
            if batch_frames:
                _process_batch(batch_frames, batch_frame_indices)

            cap.release()
            
            def build_list(cls_name):
                return self._get_class_list(confirmed, cls_name, gps_points)

            defected_sign_board_list = build_list("defected_sign_board")
            pothole_list = build_list("pothole")
            road_crack_list = build_list("road_crack")
            damaged_road_marking_list = build_list("damaged_road_marking")
            good_sign_board_list = build_list("good_sign_board")
            drain_issue_list = build_list("drain_issue")
            defective_culvert_list = build_list("defective_culvert")
            good_culvert_list = build_list("good_culvert")

            frames_with_detections = len(_pending_frames)
            detection_rate = round((frames_with_detections / frame_count) * 100, 2) if frame_count > 0 else 0

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
                    "unique_defected_sign_board": len(counted_ids["defected_sign_board"]),
                    "unique_pothole": len(counted_ids["pothole"]),
                    "unique_road_crack": len(counted_ids["road_crack"]),
                    "unique_damaged_road_marking": len(counted_ids["damaged_road_marking"]),
                    "unique_good_sign_board": len(counted_ids["good_sign_board"]),
                    "unique_drain_issue": len(counted_ids["drain_issue"]),
                    "unique_defective_culvert": len(counted_ids["defective_culvert"]),
                    "unique_good_culvert": len(counted_ids["good_culvert"]),
                    "total_road_damage": sum(len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES),
                    "total_detections": total_detections_count[0],
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "rejection_stats": {},
                "vl_stats": None,
                "defected_sign_board_list": defected_sign_board_list,
                "pothole_list": pothole_list,
                "road_crack_list": road_crack_list,
                "damaged_road_marking_list": damaged_road_marking_list,
                "good_sign_board_list": good_sign_board_list,
                "drain_issue_list": drain_issue_list,
                "defective_culvert_list": defective_culvert_list,
                "good_culvert_list": good_culvert_list,
                "frames": _pending_frames,
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
                + drain_issue_list
                + defective_culvert_list
                + good_culvert_list
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
                        "rejection_stats": {},
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

    def _get_class_list(self, confirmed, class_name, gps_points):
        lst = []
        gps_timestamps = [p.get("timestamp", 0) for p in gps_points] if gps_points else []
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
                    gps_coords = self.find_nearest_gps(
                        info["first_detected_time"], gps_points, gps_timestamps
                    )
                    det_data.update(gps_coords)
                lst.append(det_data)
        return sorted(lst, key=lambda x: x["first_detected_frame"])

    def _save_to_db(self, video_id, all_detections):
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
                project_id, package_id, chainage_id = video_project_id, video_package_id, video_chainage_id
                if lat and lng:
                    chainage = find_chainage_by_gps(db, lat, lng, package_id=video_package_id, direction=video_direction)
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
                logger.info(f"Saved {len(db_detections)} Gemini detections to database for {video_id}")
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()
