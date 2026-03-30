"""
YOLOE Trained VL Helper — Verification wrapper for the fine-tuned YOLOE model.

This helper provides a verification callback (`process_with_trained_vl`) that
routes detections through the following logic:

  Image → YOLOE Detection → Is class in VL_SKIP_CLASSES?
                              ├─ Yes → Keep detection directly
                              └─ No  → VL verification
                                        ├─ True  → Keep
                                        └─ False → Reject

The `VL_SKIP_CLASSES` list is **dynamic and configurable**. By default, both
signboard classes are kept directly (no VL call). Developers can modify this
list at runtime to control which classes bypass VL verification.

Model: models/yoloe_best.pt  (fine-tuned on 5 classes)
Confidence threshold: 0.70
"""

import logging
import threading
import time

import numpy as np
import torch
from ultralytics import YOLO

from app.helpers.vl_helper import process_with_vl

logger = logging.getLogger(__name__)

# ── Model Configuration ──────────────────────────────────────────────────────
YOLOE_TRAINED_MODEL_PATH = "models/yoloe_best.pt"
YOLOE_TRAINED_CONF = 0.70  # confidence threshold for the fine-tuned model

# ── Trained Model Classes (matching dataset.yaml) ────────────────────────────
# The model was trained with these 5 classes in this exact order:
TRAINED_CLASSES = {
    0: "defected_sign_board",
    1: "good_sign_board",
    2: "pothole",
    3: "road_crack",
    4: "damaged_road_marking",
}

# All valid class names from the trained model
ALL_TRAINED_CLASSES = set(TRAINED_CLASSES.values())

# Road damage classes (excludes good_sign_board)
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}

# ── Dynamic VL Skip Configuration ────────────────────────────────────────────
# Classes listed here will BYPASS VL verification and be kept directly.
# All other classes will be sent to VL for cross-verification.
#
# DEFAULT: Both signboard classes skip VL (shown directly).
#          pothole, road_crack, damaged_road_marking → sent to VL.
#
# USAGE EXAMPLES:
#   - To also skip VL for potholes:
#       VL_SKIP_CLASSES.append("pothole")
#   - To send EVERYTHING to VL (no skips):
#       VL_SKIP_CLASSES.clear()
#   - To skip VL for signboards + potholes (road_crack + damaged_road_marking → VL):
#       VL_SKIP_CLASSES = ["defected_sign_board", "good_sign_board", "pothole"]
#
VL_SKIP_CLASSES = [
    "defected_sign_board",
    "good_sign_board",
]


# ── Lazy Singleton for Trained YOLOE Model ───────────────────────────────────
_trained_model_instance = None
_trained_model_lock = threading.Lock()


def load_yoloe_trained_model() -> YOLO:
    """Return the singleton fine-tuned YOLOE model, creating it on first call."""
    global _trained_model_instance
    if _trained_model_instance is None:
        with _trained_model_lock:
            if _trained_model_instance is None:  # double-checked locking
                logger.info(f"⏳ Loading trained YOLOE model: {YOLOE_TRAINED_MODEL_PATH}")
                t0 = time.time()
                model = YOLO(YOLOE_TRAINED_MODEL_PATH)
                logger.info(
                    f"Trained YOLOE model loaded in {time.time() - t0:.1f}s — "
                    f"{len(TRAINED_CLASSES)} classes"
                )
                # Warmup
                try:
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.predict(dummy, conf=YOLOE_TRAINED_CONF, verbose=False)
                    logger.info("Trained YOLOE warmup done.")
                except Exception as e:
                    logger.warning(f"Trained YOLOE warmup failed (non-fatal): {e}")

                _trained_model_instance = model
    return _trained_model_instance


def process_with_trained_vl(frame, bbox, predicted_class):
    """
    Verification callback for the trained YOLOE model.

    Logic:
      - If predicted_class is in VL_SKIP_CLASSES → return success directly
        (no VL API call, detection is kept as-is)
      - Otherwise → delegate to the standard VL helper (process_with_vl)
        for cross-verification

    Args:
        frame: Original video frame (numpy array)
        bbox: Tuple (x1, y1, x2, y2) of bounding box
        predicted_class: Class name predicted by the trained YOLOE model

    Returns:
        dict with verification results or None if verification fails/skipped
    """
    if predicted_class in VL_SKIP_CLASSES:
        logger.info(
            f"VL SKIP (trained): class='{predicted_class}' is in VL_SKIP_CLASSES — "
            f"keeping detection directly"
        )
        return {
            "category": predicted_class,
            "confidence": "high",
            "belongs_to_category": True,
            "yolo_prediction": predicted_class,
            "_vl_elapsed_s": 0.0,  # no VL call made
        }

    # Delegate to the standard VL helper for cross-verification
    logger.info(
        f"VL VERIFY (trained): class='{predicted_class}' NOT in VL_SKIP_CLASSES — "
        f"sending to VL for verification"
    )
    return process_with_vl(frame, bbox, predicted_class)
