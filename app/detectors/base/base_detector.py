import json
import asyncio
import logging
import torch
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

from app.ws.websocket_manager import manager
from app.core.config import processing_status

logger = logging.getLogger(__name__)

# Shared thread pool for all detectors
executor = ThreadPoolExecutor(max_workers=4)


class BaseDetector:
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading model {self.model_path} on device: {self.device}")
            self.model = YOLO(self.model_path)
            # NOTE: do NOT call model.to(device) + model.half() here.
            # Ultralytics runs Conv+BN fusion (fuse()) on the first predict/track call
            # and requires FP32 weights at that point. FP16 is applied correctly via
            # half=USE_HALF in every model.track() call inside _process_video_blocking.
            if self.device.startswith("cuda"):
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(
                    f"Model will run on GPU: {gpu_name} ({vram:.1f} GB VRAM) — FP16 via track(half=True)"
                )
            else:
                logger.info("Model loaded on CPU (FP32)")
            self._warmup()
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            raise

    def _warmup(self):
        try:
            import numpy as np

            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, device=self.device)
        except:
            pass

    @staticmethod
    def find_nearest_gps(detection_time: float, gps_points: list) -> dict:
        if not gps_points:
            return {"lat": None, "lng": None}
        nearest_point = min(
            gps_points, key=lambda p: abs(p.get("timestamp", 0) - detection_time)
        )
        return {"lat": nearest_point.get("lat"), "lng": nearest_point.get("lng")}

    async def process_video(
        self, video_id: str, video_path: str, json_path: str, speed_kmh: int
    ):
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

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        """To be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _process_video_blocking")

    def _send_progress(self, video_id, progress, loop, extra_data=None):
        processing_status[video_id]["progress"] = progress
        message = {"type": "progress", "progress": progress}
        if extra_data:
            message.update(extra_data)

        asyncio.run_coroutine_threadsafe(
            manager.send_message(video_id, message),
            loop,
        )

    def _load_gps_data(self, json_path):
        gps_points = []
        if json_path and Path(json_path).exists():
            try:
                with open(json_path, "r") as f:
                    gps_data = json.load(f)
                    gps_points = gps_data.get("gpsPoints", [])
                    logger.info(f"Loaded {len(gps_points)} GPS points from {json_path}")
            except Exception as e:
                logger.warning(f"Failed to load GPS data: {e}")
        return gps_points
