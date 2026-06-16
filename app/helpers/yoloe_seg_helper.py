"""
YOLOE Seg Helper — Dual-model loader for road feature detection.

Model 1: yoloe-26x-seg-pf.pt  — Prompt-FREE open-vocabulary detection (1200+ built-in classes).
         Detects: pothole, manhole_cover, defected_sign_board, street_light_pole, water_puddle
Model 2: yolo12s_RDD2022_best.pt — Trained model for road crack detection only.

KEY FIX: Use the -pf (prompt-free) checkpoint instead of -seg.pt + set_classes().
         The -pf model uses its own internal 1200+ class vocabulary and does NOT need
         set_classes() — calling it would be a no-op or error on the -pf checkpoint.
"""

import logging
import threading
import time

import numpy as np
import torch
from ultralytics import YOLOE, YOLO

logger = logging.getLogger(__name__)

# ── Model Configuration ─────────────────────────────────────────────────────
# IMPORTANT: Use the -pf (prompt-free) checkpoint — this is what the notebook uses.
# yoloe-26x-seg.pt  → requires set_classes() with text prompts (worse recall)
# yoloe-26x-seg-pf.pt → built-in 1200+ class vocab, no set_classes() needed (matches notebook)
YOLOE_SEG_MODEL_PATH = r"models\yoloe-26x-seg-pf.pt"
RDD_CRACK_MODEL_PATH = r"models\yolo12s_RDD2022_best.pt"

YOLOE_SEG_CONF = 0.25  # confidence threshold for YOLOE seg
RDD_CRACK_CONF = 0.30  # confidence threshold for RDD2022 road crack model

# ── YOLOE prompt-free class name mapping ─────────────────────────────────────
# The -pf model uses its own internal class names from the 1200-class LVIS vocabulary.
# Map those names → your backend class names.
# The -pf model detects these as standard LVIS labels — map the ones you care about.
YOLOE_PF_CLASS_MAP = {
    # Pothole / road damage
    "pothole": "pothole",

    # Manhole / drain covers
    "manhole": "manhole_cover",
    "manhole cover": "manhole_cover",
    "drain": "manhole_cover",

    # Traffic signs / signboards
    "traffic sign": "defected_sign_board",
    "traffic signboard": "defected_sign_board",
    "signboard": "defected_sign_board",
    "sign": "defected_sign_board",
    "stop sign": "defected_sign_board",
    "road sign": "defected_sign_board",

    # Street light poles
    "street light": "street_light_pole",
    "street light pole": "street_light_pole",
    "pole": "street_light_pole",
    "utility pole": "street_light_pole",
    "light pole": "street_light_pole",

    # Water puddles / flooding
    "puddle": "water_puddle",
    "water puddle": "water_puddle",
    "flood": "water_puddle",
}

# RDD2022 class mapping
# D00/D10/D20 → road_crack, D40 → pothole, Repair → ignored
RDD_CLASS_MAP = {
    "D00": "road_crack",   # Longitudinal Crack
    "D10": "road_crack",   # Transverse Crack
    "D20": "road_crack",   # Alligator Crack
    "D40": "pothole",      # Pothole — dual-detected with YOLOE
    "Repair": None,        # Repaired Area — ignored
}

# All output classes produced by the dual-model pipeline
ALL_CLASSES = set(YOLOE_PF_CLASS_MAP.values()) | {"road_crack"}


def map_yoloe_seg_class(raw_class_name: str) -> str | None:
    """
    Map a raw YOLOE prompt-free class name to a backend class.
    Case-insensitive lookup — the -pf model may return mixed-case names.
    Returns None if the class is not one we care about (caller should skip it).
    """
    return YOLOE_PF_CLASS_MAP.get(raw_class_name.lower())


def map_rdd_crack_class(rdd_class_name: str) -> str | None:
    """Map an RDD2022 class to backend class (road_crack or pothole; None = ignore)."""
    return RDD_CLASS_MAP.get(rdd_class_name)


# ── Lazy Singletons ─────────────────────────────────────────────────────────
_yoloe_seg_instance = None
_yoloe_seg_lock = threading.Lock()

_rdd_crack_instance = None
_rdd_crack_lock = threading.Lock()


def load_yoloe_seg_model() -> YOLOE:
    """
    Load and cache the YOLOE-26x-seg-pf (prompt-free) model.

    The -pf checkpoint has its own internal 1200+ class vocabulary — do NOT
    call model.set_classes() on it. Doing so is incorrect for this checkpoint
    and will reduce recall to match the notebook behavior.
    """
    global _yoloe_seg_instance
    if _yoloe_seg_instance is None:
        with _yoloe_seg_lock:
            if _yoloe_seg_instance is None:
                logger.info(f"⏳ Loading YOLOE Seg (prompt-free) model: {YOLOE_SEG_MODEL_PATH}")
                t0 = time.time()

                from app.detectors.base.base_detector import _yolo_load_ctx
                with _yolo_load_ctx():
                    model = YOLOE(YOLOE_SEG_MODEL_PATH)

                # ✅ DO NOT call model.set_classes() — the -pf checkpoint uses
                # its own built-in 1200-class vocabulary. This is the key difference
                # from the text-prompt -seg.pt checkpoint.

                logger.info(
                    f"✅ YOLOE Seg (prompt-free) model loaded in {time.time() - t0:.1f}s — "
                    f"classes: {model.names}"
                )

                # Warmup
                try:
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.predict(dummy, conf=YOLOE_SEG_CONF, verbose=False, device=device)
                    logger.info("YOLOE Seg (prompt-free) model warmup done.")
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