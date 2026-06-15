"""
YOLOE Seg Helper — Open-vocabulary detection for specific road features.

Loads the yoloe-26x-seg.pt model and provides mapping for the requested
vocabulary: pothole, asphalt crack, manhole cover, traffic sign,
street light pole, water puddle.
"""

import logging
import threading
import time

import numpy as np
import torch
from ultralytics import YOLOE

logger = logging.getLogger(__name__)

# ── Model Configuration ─────────────────────────────────────────────────────
YOLOE_SEG_MODEL_PATH = r"models\yoloe-26x-seg.pt"
YOLOE_SEG_CONF = 0.25  # confidence threshold

# Text prompts to use with YOLOE
TARGET_PROMPTS = [
    "pothole",
    "asphalt crack",
    "manhole cover",
    "traffic sign",
    "street light pole",
    "water puddle",
]

# Map prompts to backend class names
PROMPT_TO_CLASS = {
    "pothole": "pothole",
    "asphalt crack": "asphalt_crack",
    "manhole cover": "manhole_cover",
    "traffic sign": "traffic_sign",
    "street light pole": "street_light_pole",
    "water puddle": "water_puddle",
}

ALL_CLASSES = set(PROMPT_TO_CLASS.values())

def map_yoloe_seg_class(prompt_name: str) -> str | None:
    """Map a raw YOLOE prompt to a VisionRoad backend class."""
    return PROMPT_TO_CLASS.get(prompt_name)

# ── Lazy Singleton ──────────────────────────────────────────────────────────
_yoloe_seg_instance = None
_yoloe_seg_lock = threading.Lock()

def load_yoloe_seg_model() -> YOLOE:
    global _yoloe_seg_instance
    if _yoloe_seg_instance is None:
        with _yoloe_seg_lock:
            if _yoloe_seg_instance is None:
                logger.info(f"⏳ Loading YOLOE Seg model: {YOLOE_SEG_MODEL_PATH}")
                t0 = time.time()

                from app.detectors.base.base_detector import _yolo_load_ctx
                with _yolo_load_ctx():
                    model = YOLOE(YOLOE_SEG_MODEL_PATH)
                
                # Set open-vocabulary prompts
                model.set_classes(TARGET_PROMPTS)

                logger.info(
                    f"✅ YOLOE Seg model loaded in {time.time() - t0:.1f}s — "
                    f"classes: {model.names}"
                )

                # Warmup
                try:
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.predict(dummy, conf=YOLOE_SEG_CONF, verbose=False, device=device)
                    logger.info("YOLOE Seg model warmup done.")
                except Exception as e:
                    logger.warning(f"YOLOE Seg warmup failed: {e}")

                _yoloe_seg_instance = model
    return _yoloe_seg_instance
