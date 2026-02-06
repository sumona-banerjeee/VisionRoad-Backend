'''
This module processes videos to detect potholes
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

logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH ="models/best.pt"
TRACKER = "bytetrack.yaml"
MIN_DETECTION_FRAMES = 3
DETECTION_TIME_WINDOW = 1.0
CONFIDENCE_THRESHOLD = 0.80
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)


class VideoProcessor:
    def __init__(self):
        """Initialize video processor with YOLO model on GPU"""
        try:
            # Check for TensorRT engine first
            engine_path = Path(MODEL_PATH).with_suffix(".engine")
            final_model_path = str(engine_path) if engine_path.exists() else MODEL_PATH
            
            logger.info(f"Loading model from {final_model_path} on device: {DEVICE}")
            self.model = YOLO(final_model_path)

            if DEVICE == "cuda:0":
                self.model.to(DEVICE)
                logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Model loaded on CPU")

            # Warmup
            self._warmup()
            logger.info("Video processor ready")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
    def get_adaptive_params(speed):
        """Get adaptive parameters based on speed"""
        if speed < 30:
            return {"roi_ratio": 0.50, "conf": 0.70}
        elif speed < 60:
            return {"roi_ratio": 0.65, "conf": 0.70}
        else:
            return {"roi_ratio": 0.75, "conf": 0.22}

    # === Main detection function for a single frame ===
    def detect_frame(
        self, frame, frame_id, tracker, confirmed, current_time, speed
    ):
        """Detect potholes in a single frame with tracking"""
        h, w = frame.shape[:2]
        params = self.get_adaptive_params(speed)

        # ROI extraction
        roi_y = int(h * (1 - params["roi_ratio"]))
        roi = frame[roi_y:h, :]

        detections = []
        count = 0
        new_confirmed = []  # List of newly confirmed potholes in this frame

        try:
            results = self.model.track(
                roi,
                conf=params["conf"],
                tracker=TRACKER,
                persist=True,
                verbose=False,
                device=DEVICE,
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

                    # Adjust coordinates
                    y1_full, y2_full = y1 + roi_y, y2 + roi_y

                    # Update tracker
                    tracker[track_id].append(current_time)

                    # Check confirmation
                    recent = [
                        t
                        for t in tracker[track_id]
                        if current_time - t <= DETECTION_TIME_WINDOW
                    ]

                    if (
                        len(recent) >= MIN_DETECTION_FRAMES
                        and track_id not in confirmed
                    ):
                        pothole_data = {
                            "frame": frame_id,
                            "time": current_time,
                            "conf": conf,
                            "bbox": {"x1": x1, "y1": y1_full, "x2": x2, "y2": y2_full},
                        }
                        confirmed[track_id] = pothole_data
                        new_confirmed.append((track_id, pothole_data))

                    if track_id in confirmed:
                        count += 1
                        # Calculate center and area
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1_full + y2_full) / 2)
                        area = (x2 - x1) * (y2_full - y1_full)

                        detections.append(
                            {
                                "frame_id": frame_id,
                                "pothole_id": track_id,
                                "type": "pothole",
                                "confidence": round(float(conf), 3),
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

            frame_result = None
            if detections:
                frame_result = {
                    "frame_id": frame_id,
                    "speed_kmh": speed,
                    "roi_ratio": params["roi_ratio"],
                    "potholes": detections,
                }

        except Exception as e:
            logger.error(f"Detection error: {e}")
            frame_result = None

        return count, new_confirmed, frame_result

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

            logger.info(f"Processing {video_id}: {total_frames} frames @ {fps:.1f} FPS")

            # Shared state
            results_log = {"frames": []}
            pothole_list = []  # Final list of pothole data
            
            # Tracking state (Only accessed by Inference Thread)
            tracker = defaultdict(lambda: deque(maxlen=20))
            confirmed = {}

            # Queues
            inference_queue = Queue(maxsize=30)
            post_proc_queue = Queue(maxsize=100)
            
            # Events
            processing_complete = Event()

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
                        post_proc_queue.put(None) # Signal post-proc to stop
                        inference_queue.task_done()
                        break
                    
                    frame, frame_id, current_time = item
                    
                    try:
                        n, new_confirmed, frame_result = self.detect_frame(
                            frame, frame_id, tracker, confirmed, current_time, speed
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
                db_buffer = [] # Buffer for bulk inserts
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
                        
                        # 2. Process new confirmed potholes
                        for pid, info in new_confirmed:
                            # Create pothole data dict for final JSON
                            pothole_data = {
                                "pothole_id": int(pid),
                                "first_detected_frame": info["frame"],
                                "first_detected_time": round(info["time"], 2),
                                "confidence": round(float(info["conf"]), 3),
                                "bbox": info.get("bbox", {}),
                            }
                            
                            # Map GPS
                            lat, lng = None, None
                            if gps_points:
                                gps_coords = self.find_nearest_gps(info["time"], gps_points)
                                lat = gps_coords["lat"]
                                lng = gps_coords["lng"]
                                pothole_data["lat"] = lat
                                pothole_data["lng"] = lng
                            
                            pothole_list.append(pothole_data)
                            
                            # Prepare for DB Batch
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
                            
                            # Create Detection object manually for bulk insert
                            detection = Detection(
                                video_id=video_id,
                                frame_number=pothole_data["first_detected_frame"],
                                timestamp_ms=int(pothole_data["first_detected_time"] * 1000),
                                confidence=pothole_data["confidence"],
                                detection_type="pothole",
                                class_name="pothole",
                                latitude=lat,
                                longitude=lng,
                                project_id=project_id,
                                package_id=package_id,
                                location_id=location_id,
                            )
                            detection.set_bounding_box(pothole_data.get("bbox", {}))
                            db_buffer.append(detection)

                        # 3. Batch DB Insert
                        if len(db_buffer) >= 50:
                            try:
                                crud.create_detections_bulk(db, db_buffer)
                                logger.info(f"Bulk saved {len(db_buffer)} detections")
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
                                        "unique_potholes": len(pothole_list),
                                        "total_detections": total_detections,
                                    },
                                ),
                                loop,
                            )
                            last_progress = progress
                        
                        post_proc_queue.task_done()
                    
                    # Flush remaining DB buffer
                    if db_buffer:
                        try:
                            crud.create_detections_bulk(db, db_buffer)
                            logger.info(f"Bulk saved final {len(db_buffer)} detections")
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
                
                # Push to queue (blocks if full, providing backpressure)
                inference_queue.put((frame, frame_count, current_time))

            cap.release()
            
            # Signal end of stream
            inference_queue.put(None)
            
            # Wait for pipeline to finish
            inf_thread.join()
            pp_thread.join()
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # --- Finalize Results ---
            # Sort pothole list
            pothole_list = sorted(pothole_list, key=lambda x: x["first_detected_frame"])

            frames_with_detections = len(results_log["frames"])
            detection_rate = (
                round((frames_with_detections / total_frames) * 100, 2)
                if total_frames > 0
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
                    "total_frames": total_frames,
                    "unique_potholes": len(pothole_list),
                    "total_detections": total_detections,
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "pothole_list": pothole_list,
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
            logger.info(f"VIDEO PROCESSING COMPLETE: {video_id}")
            logger.info(f"Frames: {total_frames} | Potholes: {len(pothole_list)}")
            logger.info("=" * 60)
            
            return results

        except Exception as e:
            logger.error(f"Error processing {video_id}: {e}")
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
