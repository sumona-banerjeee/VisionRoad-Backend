"""
YOLOE Seg Helper — Dual-model loader for road feature detection.

Model 1: yoloe-26x-seg.pt  — Open-vocabulary detection for pothole, asphalt crack,
         manhole cover, traffic sign, street light pole, water puddle.
Model 2: yolo12s_RDD2022_best.pt — Trained model for road crack detection only.
"""

import logging
import threading
import time

import numpy as np
import torch
from ultralytics import YOLOE, YOLO

logger = logging.getLogger(__name__)

# ── Model Configuration ─────────────────────────────────────────────────────
YOLOE_SEG_MODEL_PATH = r"models\yoloe-26x-seg.pt"
RDD_CRACK_MODEL_PATH = r"models\yolo12s_RDD2022_best.pt"

YOLOE_SEG_CONF = 0.25  # confidence threshold for YOLOE seg
RDD_CRACK_CONF = 0.30  # confidence threshold for RDD2022 road crack model

# ── YOLOE open-vocab text prompts ────────────────────────────────────────────
TARGET_PROMPTS = [
    "pothole",
    "asphalt crack",
    "manhole cover",
    "traffic sign",
    "street light pole",
    "water puddle",
]

# Map YOLOE prompts → backend class names
PROMPT_TO_CLASS = {
    "pothole": "pothole",
    "asphalt crack": "asphalt_crack",
    "manhole cover": "manhole_cover",
    "traffic sign": "traffic_sign",
    "street light pole": "street_light_pole",
    "water puddle": "water_puddle",
}

# RDD2022 class mapping (D00/D10/D20 → road_crack, D40 → pothole ignored here
# because YOLOE handles pothole; Repair → ignored)
RDD_CLASS_MAP = {
    "D00": "road_crack",       # Longitudinal Crack
    "D10": "road_crack",       # Transverse Crack
    "D20": "road_crack",       # Alligator Crack
    "D40": "pothole",          # Pothole — detected by both YOLOE and RDD2022
    "Repair": None,            # Repaired Area — ignored
}

# All output classes produced by the dual-model pipeline
ALL_CLASSES = set(PROMPT_TO_CLASS.values()) | {"road_crack"}


def map_yoloe_seg_class(prompt_name: str) -> str | None:
    """Map a raw YOLOE prompt to a VisionRoad backend class."""
    return PROMPT_TO_CLASS.get(prompt_name)


def map_rdd_crack_class(rdd_class_name: str) -> str | None:
    """Map an RDD2022 class to backend class (road_crack only)."""
    return RDD_CLASS_MAP.get(rdd_class_name)


# ── Lazy Singletons ─────────────────────────────────────────────────────────
_yoloe_seg_instance = None
_yoloe_seg_lock = threading.Lock()

_rdd_crack_instance = None
_rdd_crack_lock = threading.Lock()


def load_yoloe_seg_model() -> YOLOE:
    """Load and cache the YOLOE-26x-seg open-vocabulary model."""
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


def load_rdd_crack_model() -> YOLO:
    """Load and cache the RDD2022 road crack model."""
    global _rdd_crack_instance
    if _rdd_crack_instance is None:
        with _rdd_crack_lock:
            if _rdd_crack_instance is None:
                logger.info(f"⏳ Loading RDD crack model: {RDD_CRACK_MODEL_PATH}")
                t0 = time.time()

                from app.detectors.base.base_detector import _yolo_load_ctx
                with _yolo_load_ctx():
                    model = YOLO(RDD_CRACK_MODEL_PATH)

                logger.info(
                    f"✅ RDD crack model loaded in {time.time() - t0:.1f}s — "
                    f"classes: {model.names}"
                )

                # Warmup
                try:
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.predict(dummy, conf=RDD_CRACK_CONF, verbose=False, device=device)
                    logger.info("RDD crack model warmup done.")
                except Exception as e:
                    logger.warning(f"RDD crack warmup failed: {e}")

                _rdd_crack_instance = model
    return _rdd_crack_instance
