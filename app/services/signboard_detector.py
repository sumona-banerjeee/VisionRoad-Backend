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

logger = logging.getLogger(__name__)

# Configuration
TRACKER = "bytetrack.yaml"
CONFIDENCE_THRESHOLD = 0.6
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best-board-v2.pt"

# ROI Configuration (from testv5.py)
# Top 10% to 70% of frame height
ROI_TOP_RATIO = 0.10
ROI_BOTTOM_RATIO = 0.70
ROI_LEFT_RATIO = 0.0
ROI_RIGHT_RATIO = 1.0

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)


class SignBoardDetector:
    def __init__(self):
        """Initialize sign board detector with YOLO model on GPU"""
        try:
            logger.info(f"Loading sign board model on device: {DEVICE}")
            self.model = YOLO(MODEL_PATH)

            if DEVICE == "cuda:0":
                self.model.to(DEVICE)
                logger.info(
                    f"Sign board model loaded on GPU: {torch.cuda.get_device_name(0)}"
                )
            else:
                logger.info("Sign board model loaded on CPU")

            # Warmup
            self._warmup()
            logger.info("Sign board detector ready")

        except Exception as e:
            logger.error(f"Failed to load sign board model: {e}")
            raise

    def _warmup(self):
        """Warmup model for optimal performance"""
        try:
            import numpy as np

            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, device=DEVICE)
        except:
            pass

    def detect_frame(
        self, frame, frame_id, results_log, tracker, confirmed, current_time, fps
    ):
        """Detect sign boards in a single frame with tracking"""
        h, w = frame.shape[:2]

        # Calculate ROI
        roi_left = int(w * ROI_LEFT_RATIO)
        roi_right = int(w * ROI_RIGHT_RATIO)
        roi_top = int(h * ROI_TOP_RATIO)
        roi_bottom = int(h * ROI_BOTTOM_RATIO)

        detections = []

        try:
            results = self.model.track(
                frame,
                conf=CONFIDENCE_THRESHOLD,
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
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None

                if ids is None:
                    continue

                for box, track_id, conf, class_id in zip(boxes, ids, confs, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)

                    # Calculate center
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Check if detection is in ROI
                    in_roi = roi_left < cx < roi_right and roi_top < cy < roi_bottom
                    if not in_roi:
                        continue

                    # Get class name
                    class_name = str(self.model.names[class_id])

                    # Track first detection
                    if track_id not in confirmed:
                        confirmed[track_id] = {
                            "signboard_id": track_id,
                            "type": class_name,
                            "first_detected_frame": frame_id,
                            "first_detected_time": round(current_time, 2),
                            "confidence": round(float(conf), 3),
                        }

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

            if detections:
                results_log["frames"].append(
                    {"frame_id": frame_id, "signboards": detections}
                )

        except Exception as e:
            logger.error(f"Sign board detection error: {e}")

        return len(detections)

    def _process_video_blocking(self, video_id: str, video_path: str, speed: int, loop):
        """Process video in blocking thread"""
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id, {"type": "status", "status": "processing", "progress": 0}
                ),
                loop,
            )

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
            total_detections = 0
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
                                "unique_signboards": len(confirmed),
                                "total_detections": total_detections,
                            },
                        ),
                        loop,
                    )
                    last_progress = progress

            cap.release()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Build results
            signboard_list = sorted(
                [
                    {
                        "signboard_id": info["signboard_id"],
                        "type": info["type"],
                        "first_detected_frame": info["first_detected_frame"],
                        "first_detected_time": info["first_detected_time"],
                        "confidence": info["confidence"],
                    }
                    for info in confirmed.values()
                ],
                key=lambda x: x["first_detected_frame"],
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
            unique_ids = sorted([s["signboard_id"] for s in signboard_list])
            logger.info("=" * 60)
            logger.info(f"SIGN BOARD DETECTION COMPLETE: {video_id}")
            logger.info(f"Total frames: {frame_count}")
            logger.info(f"Total detections: {total_detections}")
            logger.info(f">>> UNIQUE SIGNBOARDS: {len(confirmed)} <<<")
            logger.info(f"Signboard IDs: {unique_ids}")
            logger.info("=" * 60)
            print(f"\n{'='*60}")
            print(f"SIGN BOARD DETECTION COMPLETE")
            print(f"{'='*60}")
            print(f"Video ID: {video_id}")
            print(f"Frames: {frame_count} | Device: {DEVICE}")
            print(f"Total detections: {total_detections}")
            print(f">>> UNIQUE SIGNBOARDS: {len(confirmed)} <<<")
            print(f"Signboard IDs: {unique_ids}")
            print(f"{'='*60}\n")
            return results

        except Exception as e:
            logger.error(f"Error processing sign board detection for {video_id}: {e}")
            processing_status[video_id] = {"status": "error", "message": str(e)}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(video_id, {"type": "error", "message": str(e)}),
                loop,
            )
            raise

    async def process_video(self, video_id: str, video_path: str, speed_kmh: int):
        """Async video processing"""
        processing_status[video_id] = {"status": "processing", "progress": 0}
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            self._process_video_blocking,
            video_id,
            video_path,
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
