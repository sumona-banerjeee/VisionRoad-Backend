"""
HuggingFace Road Damage Helper — RDD2022 model for pothole & road crack detection.

Loads the yolo12s_RDD2022_best.pt model (downloaded from HuggingFace:
rezzzq/yolo12s-road-damage-rdd2022) and provides class mapping from
RDD2022 codes to VisionRoad backend classes.

RDD2022 Classes:
    D00 → Longitudinal Crack → road_crack
    D10 → Transverse Crack   → road_crack
    D20 → Alligator Crack    → road_crack
    D40 → Pothole             → pothole
    Repair → (ignored)

Model: YOLOv12-small with A2C2f attention modules, fine-tuned on RDD2022.
Input Size: 640x640
"""

import logging
import threading
import time

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ── Model Configuration ─────────────────────────────────────────────────────
HF_ROAD_DAMAGE_MODEL_PATH = r"models\yolo12s_RDD2022_best.pt"
HF_ROAD_DAMAGE_CONF = 0.30  # confidence threshold for road damage detections

# ── RDD2022 → VisionRoad Class Mapping ───────────────────────────────────────
# Maps the model's output class names to the backend class names used in
# VisionRoad's pipeline.
HF_CLASS_MAP = {
    "D00": "road_crack",       # Longitudinal Crack
    "D10": "road_crack",       # Transverse Crack
    "D20": "road_crack",       # Alligator Crack
    "D40": "pothole",          # Pothole
    "Repair": None,            # Repaired Area — ignored
}

HF_ACCEPTED_CLASSES = {"pothole", "road_crack"}


def map_hf_class(hf_class_name: str) -> str | None:
    return HF_CLASS_MAP.get(hf_class_name)


# ── Lazy Singleton for HuggingFace Road Damage Model ────────────────────────
_hf_model_instance = None
_hf_model_lock = threading.Lock()


def load_hf_road_damage_model() -> YOLO:
    global _hf_model_instance
    if _hf_model_instance is None:
        with _hf_model_lock:
            if _hf_model_instance is None:
                logger.info(f"⏳ Loading HF road damage model: {HF_ROAD_DAMAGE_MODEL_PATH}")
                t0 = time.time()

                from app.detectors.base.base_detector import _yolo_load_ctx
                with _yolo_load_ctx():
                    model = YOLO(HF_ROAD_DAMAGE_MODEL_PATH)

                logger.info(
                    f"✅ HF road damage model loaded in {time.time() - t0:.1f}s — "
                    f"classes: {model.names}"
                )

                try:
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.predict(dummy, conf=HF_ROAD_DAMAGE_CONF, verbose=False, device=device)
                    logger.info("HF road damage model warmup done.")
                except Exception as e:
                    logger.warning(f"HF warmup failed: {e}")

                _hf_model_instance = model
    return _hf_model_instance
