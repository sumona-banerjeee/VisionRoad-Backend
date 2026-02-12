'''
This module processes videos to detect signboards
'''
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
from queue import Queue
from threading import Thread, Event
import time

from app.ws.websocket_manager import manager
from app.core.storage import processing_status, detection_results, RESULTS_DIR
from app.db.database import SessionLocal
from app.db import crud
from app.services.location_mapper import find_location_by_gps
from app.models.processing import ProcessingStatusEnum
from app.models.detection import Detection
from app.core.model_loader import load_yolo_model

logger = logging.getLogger(__name__)

# Configuration
TRACKER = "bytetrack.yaml"
CONFIDENCE_THRESHOLD = 0.6
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best-board-v2.pt"

# ROI Configuration
# Top 10% to 70% of frame height
ROI_TOP_RATIO = 0.10
ROI_BOTTOM_RATIO = 0.70
ROI_LEFT_RATIO = 0.0
ROI_RIGHT_RATIO = 1.0

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)


class SignBoardDetector:
    def __init__(self):
        """Initialize sign board detector"""
        self._model = None
        logger.info("Sign board detector initialized (model loading deferred)")

    def _get_model(self):
        """Lazy load and warmup the model"""
        if self._model is not None:
            return self._model
            
        try:
            # Check for TensorRT engine first
            engine_path = Path(MODEL_PATH).with_suffix(".engine")
            final_model_path = str(engine_path) if engine_path.exists() else MODEL_PATH
            
            logger.info(f"Loading sign board model from {final_model_path} on device: {DEVICE}")
            self._model = load_yolo_model(final_model_path)

            if DEVICE == "cuda:0":
                self._model.to(DEVICE)
                logger.info(
                    f"Sign board model loaded on GPU: {torch.cuda.get_device_name(0)}"
                )
            else:
                logger.info("Sign board model loaded on CPU")

            # Warmup
            self._warmup()
            logger.info("Sign board detector model ready")
            return self._model

        except Exception as e:
            logger.error(f"Failed to load sign board model: {e}")
            raise

    def _warmup(self):
        """Warmup model for optimal performance"""
        if self._model is None:
            return
        try:
            import numpy as np

            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False, device=DEVICE)
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

    # === Main detection function for a single frame ===
    def detect_frame(
        self, frame, frame_id, tracker, confirmed, current_time, fps
    ):
        """Detect sign boards in a single frame with tracking"""
        h, w = frame.shape[:2]

        # Calculate ROI
        roi_left = int(w * ROI_LEFT_RATIO)
        roi_right = int(w * ROI_RIGHT_RATIO)
        roi_top = int(h * ROI_TOP_RATIO)
        roi_bottom = int(h * ROI_BOTTOM_RATIO)

        detections = []
        new_confirmed = []
        # Crop to ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        detections = []
        new_confirmed = []

        try:
            model = self._get_model()
            results = model.track(
                roi,
                persist=True,
                conf=CONFIDENCE_THRESHOLD,
                # tracker=TRACKER, # affect detection
                # verbose=False,
                device=DEVICE,  # use GPU if available
                imgsz=640,
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
                    rx1, ry1, rx2, ry2 = map(int, box)
                    track_id = int(track_id)

                    # Adjust coordinates back to full frame
                    x1, y1 = rx1 + roi_left, ry1 + roi_top
                    x2, y2 = rx2 + roi_left, ry2 + roi_top

                    # Calculate center in full frame
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Get class name
                    class_name = str(self.model.names[class_id])

                    # Track first detection
                    if track_id not in confirmed:
                        signboard_data = {
                            "signboard_id": track_id,
                            "type": class_name,
                            "first_detected_frame": frame_id,
                            "first_detected_time": round(current_time, 2),
                            "confidence": round(float(conf), 3),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "center": {"x": cx, "y": cy},
                        }
                        confirmed[track_id] = signboard_data
                        new_confirmed.append((track_id, signboard_data))

                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)

                    detections.append(
                        {
                            "frame_id": frame_id,
                            "signboard_id": track_id,
                            "type": class_name,
                            "confidence": round(float(conf), 3),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "center": {"x": cx, "y": cy},
                            "area": area,
                        }
                    )

            frame_result = None
            if detections:
                frame_result = {"frame_id": frame_id, "signboards": detections}

        except Exception as e:
            logger.error(f"Sign board detection error: {e}")
            frame_result = None

        return len(detections), new_confirmed, frame_result

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        """Process video in blocking thread with pipeline pattern"""
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id, {"type": "status", "status": "processing", "progress": 0}
                ),
                loop,
            )

            # Load GPS data
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

            logger.info(
                f"Processing sign board detection for {video_id}: {total_frames} frames @ {fps:.1f} FPS"
            )

            # Shared state
            results_log = {"frames": []}
            signboard_list = []
            
            # Tracking state (Only accessed by Inference Thread)
            tracker = defaultdict(lambda: deque(maxlen=20))
            confirmed = {}

            # Queues
            inference_queue = Queue(maxsize=30)
            post_proc_queue = Queue(maxsize=100)
            
            # Counters
            total_detections = 0 
            frames_processed_count = 0

            # --- Worker Functions ---
            
            def inference_worker():
                """Consumes frames, runs inference, produces results"""
                nonlocal total_detections
                while True:
                    item = inference_queue.get()
                    if item is None:
                        post_proc_queue.put(None) 
                        inference_queue.task_done()
                        break
                    
                    frame, frame_id, current_time = item
                    
                    try:
                        n, new_confirmed, frame_result = self.detect_frame(
                            frame, frame_id, tracker, confirmed, current_time, fps
                        )
                        total_detections += n
                        
                        # Send result to post-processing
                        post_proc_queue.put((frame_result, new_confirmed))
                        
                    except Exception as e:
                        logger.error(f"Inference error frame {frame_id}: {e}")
                    
                    inference_queue.task_done()

            def post_proc_worker():
                """Consumes results, handles DB batching, logging"""
                nonlocal frames_processed_count
                
                db = SessionLocal()
                db_buffer = [] 
                last_progress = 0
                
                try:
                    while True:
                        item = post_proc_queue.get()
                        if item is None:
                            break
                        
                        frame_result, new_confirmed = item
                        frames_processed_count += 1
                        
                        # 1. Update results log
                        if frame_result:
                            results_log["frames"].append(frame_result)
                        
                        # 2. Process new confirmed signboards
                        for sid, info in new_confirmed:
                            signboard_data = {
                                "signboard_id": int(sid),
                                "type": info["type"],
                                "first_detected_frame": info["first_detected_frame"],
                                "first_detected_time": round(info["first_detected_time"], 2),
                                "confidence": round(float(info["confidence"]), 3),
                                "bbox": info.get("bbox", {}),
                            }
                            
                            # Map GPS
                            lat, lng = None, None
                            if gps_points:
                                gps_coords = self.find_nearest_gps(
                                    info["first_detected_time"], gps_points
                                )
                                lat = gps_coords["lat"]
                                lng = gps_coords["lng"]
                                signboard_data["lat"] = lat
                                signboard_data["lng"] = lng
                            
                            signboard_list.append(signboard_data)
                            
                            # Prepare for DB Batch
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
                            
                            # Create Detection object manually
                            detection = Detection(
                                video_id=video_id,
                                frame_number=signboard_data["first_detected_frame"],
                                timestamp_ms=int(signboard_data["first_detected_time"] * 1000),
                                confidence=signboard_data["confidence"],
                                detection_type="signboard",
                                class_name=signboard_data["type"],
                                latitude=lat,
                                longitude=lng,
                                project_id=project_id,
                                package_id=package_id,
                                location_id=location_id,
                            )
                            detection.set_bounding_box(signboard_data.get("bbox", {}))
                            db_buffer.append(detection)

                        # 3. Batch DB Insert
                        if len(db_buffer) >= 50:
                            try:
                                crud.create_detections_bulk(db, db_buffer)
                                logger.info(f"Bulk saved {len(db_buffer)} signboard detections")
                                db_buffer = []
                            except Exception as e:
                                logger.error(f"Bulk save error: {e}")
                                db.rollback()

                        # 4. Progress Update
                        progress = int((frames_processed_count / total_frames) * 100)
                        if progress - last_progress >= 5:
                            processing_status[video_id]["progress"] = progress
                            asyncio.run_coroutine_threadsafe(
                                manager.send_message(
                                    video_id,
                                    {
                                        "type": "progress",
                                        "progress": progress,
                                        "unique_signboards": len(signboard_list),
                                        "total_detections": total_detections,
                                    },
                                ),
                                loop,
                            )
                            last_progress = progress
                        
                        post_proc_queue.task_done()
                    
                    # Flush remaining
                    if db_buffer:
                        try:
                            crud.create_detections_bulk(db, db_buffer)
                            logger.info(f"Bulk saved final {len(db_buffer)} signboards")
                        except Exception as e:
                            logger.error(f"Final bulk save error: {e}")
                            db.rollback()
                            
                finally:
                    db.close()
                    post_proc_queue.task_done()

            # --- Start Pipeline ---
            inf_thread = Thread(target=inference_worker, daemon=True)
            pp_thread = Thread(target=post_proc_worker, daemon=True)
            inf_thread.start()
            pp_thread.start()

            # --- Main Loop (Producer) ---
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / fps
                
                inference_queue.put((frame, frame_count, current_time))

            cap.release()
            
            # Signal end
            inference_queue.put(None)
            
            inf_thread.join()
            pp_thread.join()
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # --- Finalize Results ---
            signboard_list = sorted(
                signboard_list, key=lambda x: x["first_detected_frame"]
            )

            frames_with_detections = len(results_log["frames"])
            detection_rate = (
                round((frames_with_detections / total_frames) * 100, 2)
                if total_frames > 0
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
                    "total_frames": total_frames,
                    "unique_signboards": len(signboard_list),
                    "total_detections": total_detections,
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "signboard_list": signboard_list,
                "frames": results_log["frames"],
            }

            detection_results[video_id] = results

            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            # Update final status
            db = SessionLocal()
            try:
                crud.update_processing_status(
                    db=db,
                    video_id=video_id,
                    status=ProcessingStatusEnum.COMPLETED,
                    progress=100,
                    result_summary=results["summary"],
                )
                db.commit()
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
                    },
                ),
                loop,
            )

            # Log summary
            logger.info("=" * 60)
            logger.info(f"SIGN BOARD DETECTION COMPLETE: {video_id}")
            logger.info(f"Frames: {total_frames} | Signboards: {len(signboard_list)}")
            logger.info("=" * 60)
            
            return results

        except Exception as e:
            logger.error(f"Error processing sign board detection for {video_id}: {e}")
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
