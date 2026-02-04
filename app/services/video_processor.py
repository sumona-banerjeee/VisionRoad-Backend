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
            logger.info(f"Loading model on device: {DEVICE}")
            self.model = YOLO("models/best.pt")

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
        self, frame, frame_id, results_log, tracker, confirmed, current_time, speed
    ):
        """Detect potholes in a single frame with tracking"""
        h, w = frame.shape[:2]
        params = self.get_adaptive_params(speed)

        # ROI extraction
        roi_y = int(h * (1 - params["roi_ratio"]))
        roi = frame[roi_y:h, :]

        detections = []
        count = 0
        new_count = 0

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
                        confirmed[track_id] = {
                            "frame": frame_id,
                            "time": current_time,
                            "conf": conf,
                            "bbox": {"x1": x1, "y1": y1_full, "x2": x2, "y2": y2_full},
                        }
                        new_count = 1

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

            if detections:
                results_log["frames"].append(
                    {
                        "frame_id": frame_id,
                        "speed_kmh": speed,
                        "roi_ratio": params["roi_ratio"],
                        "potholes": detections,
                    }
                )

        except Exception as e:
            logger.error(f"Detection error: {e}")

        return count, new_count

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

            logger.info(f"Processing {video_id}: {total_frames} frames @ {fps:.1f} FPS")

            results_log = {"frames": []}
            tracker = defaultdict(lambda: deque(maxlen=20))
            confirmed = {}
            total_detections = 0
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
                total_detections += n

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
                                "unique_potholes": len(confirmed),
                                "total_detections": total_detections,
                            },
                        ),
                        loop,
                    )
                    last_progress = progress

            cap.release()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Build results with GPS coordinates
            pothole_list = []
            for pid, info in confirmed.items():
                pothole_data = {
                    "pothole_id": int(pid),
                    "first_detected_frame": info["frame"],
                    "first_detected_time": round(info["time"], 2),
                    "confidence": round(float(info["conf"]), 3),
                    "bbox": info.get("bbox", {}),
                }

                # Add GPS coordinates if available
                if gps_points:
                    gps_coords = self.find_nearest_gps(info["time"], gps_points)
                    pothole_data["lat"] = gps_coords["lat"]
                    pothole_data["lng"] = gps_coords["lng"]

                pothole_list.append(pothole_data)

            # Sort by first detected frame
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

            # === Save detections to database ===
            try:
                db = SessionLocal()
                saved_count = 0

                for pothole in pothole_list:
                    lat = pothole.get("lat")
                    lng = pothole.get("lng")

                    # Map GPS to location hierarchy
                    project_id = None
                    package_id = None
                    location_id = None

                    if lat and lng:
                        location = find_location_by_gps(db, lat, lng)
                        if location:
                            location_id = location.id
                            package_id = location.package_id
                            # Get project_id from package
                            if location.package:
                                project_id = location.package.project_id

                    # Save detection to database
                    crud.create_detection(
                        db=db,
                        video_id=video_id,
                        frame_number=pothole["first_detected_frame"],
                        timestamp_ms=int(pothole["first_detected_time"] * 1000),
                        confidence=pothole["confidence"],
                        detection_type="pothole",
                        class_name="pothole",
                        bounding_box=pothole.get("bbox", {}),
                        latitude=lat,
                        longitude=lng,
                        project_id=project_id,
                        package_id=package_id,
                        location_id=location_id,
                    )
                    saved_count += 1

                # Update processing status in database
                crud.update_processing_status(
                    db=db,
                    video_id=video_id,
                    status=ProcessingStatusEnum.COMPLETED,
                    progress=100,
                    result_summary=results["summary"],
                )

                db.commit()
                logger.info(
                    f"Saved {saved_count} pothole detections to database for {video_id}"
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
                    },
                ),
                loop,
            )

            # Detailed logging
            unique_ids = sorted([p["pothole_id"] for p in pothole_list])
            logger.info("=" * 60)
            logger.info(f"VIDEO PROCESSING COMPLETE: {video_id}")
            logger.info(f"Total frames: {frame_count}")
            logger.info(f"Total detections: {total_detections}")
            logger.info(f">>> UNIQUE POTHOLES: {len(confirmed)} <<<")
            logger.info(f"Pothole IDs: {unique_ids}")
            logger.info("=" * 60)
            print(f"\n{'='*60}")
            print(f"VIDEO PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Video ID: {video_id}")
            print(f"Frames: {frame_count} | Device: {DEVICE}")
            print(f"Total detections: {total_detections}")
            print(f">>> UNIQUE POTHOLES: {len(confirmed)} <<<")
            print(f"Pothole IDs: {unique_ids}")
            print(f"{'='*60}\n")
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
