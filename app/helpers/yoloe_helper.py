"""
YOLOE Helper — Fine-tuned YOLOE model for road damage detection.

Fine-tuned on 5 classes:
  0: defected_sign_board
  1: good_sign_board
  2: pothole
  3: road_crack
  4: damaged_road_marking

Confidence threshold: 0.70 (flat across all classes).

Called by YoloeDetector — this helper owns the model and inference;
the detector owns the video loop, progress, GPS, DB etc.

Return contract for process_frame_with_yoloe():
    [
        {
            "class_name":  str,    # one of the 5 trained class names
            "confidence":  float,
            "bbox":        tuple,  # (x1, y1, x2, y2)
            "center":      tuple,  # (cx, cy)
        },
        ...
    ]
"""

import logging
import threading
import time

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ── YOLOE Configuration ──────────────────────────────────────────────────────
YOLOE_MODEL_WEIGHTS = "models\\yoloe_best.pt"
YOLOE_CONF_THRESHOLD = 0.70  # single flat confidence threshold for all classes

# ── Class names (must match training dataset.yaml order) ──────────────────────
YOLOE_CLASS_NAMES = [
    "defected_sign_board",   # 0
    "good_sign_board",       # 1
    "pothole",               # 2
    "road_crack",            # 3
    "damaged_road_marking",  # 4
]

NUM_CLASSES = len(YOLOE_CLASS_NAMES)

# Road damage classes (good_sign_board is excluded from damage count)
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}

EXCLUDED_CLASSES = {"good_sign_board"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES


# ── Lazy singleton for YOLOE model ───────────────────────────────────────────
_model_instance = None
_model_lock = threading.Lock()


def load_yoloe_model() -> YOLO:
    """Return the singleton fine-tuned YOLOE model, creating it on first call."""
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:  # double-checked
                logger.info(f"⏳ Loading fine-tuned YOLOE model: {YOLOE_MODEL_WEIGHTS}")
                t0 = time.time()
                model = YOLO(YOLOE_MODEL_WEIGHTS)
                logger.info(
                    f"✅ Fine-tuned YOLOE model loaded in {time.time() - t0:.1f}s — "
                    f"{NUM_CLASSES} classes: {YOLOE_CLASS_NAMES}"
                )
                # Warmup
                try:
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.predict(dummy, conf=YOLOE_CONF_THRESHOLD, verbose=False)
                    logger.info("YOLOE warmup done.")
                except Exception as e:
                    logger.warning(f"YOLOE warmup failed (non-fatal): {e}")

                _model_instance = model
    return _model_instance


def process_frame_with_yoloe(model, frame) -> list[dict]:
    """
    Run fine-tuned YOLOE inference on a single frame and return detections.

    Args:
        model: Loaded YOLOE model (from load_yoloe_model)
        frame: BGR numpy array

    Returns:
        List of detection dicts with keys:
            class_name, confidence, bbox, center
    """
    results = model.predict(frame, conf=YOLOE_CONF_THRESHOLD, verbose=False)

    detections = []
    r = results[0]
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names  # dict {class_id: class_name}

        for box, conf, cls_id in zip(boxes, confs, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = names.get(cls_id, f"class_{cls_id}")

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            detections.append({
                "class_name": class_name,
                "confidence": round(float(conf), 3),
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
            })

    return detections