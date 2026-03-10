"""
YOLOE Helper — Open-vocabulary detection with text prompts.

Encapsulates all YOLOE-specific logic:
  • 40 text prompts across 7 categories
  • Defective class filtering (29 defective indices)
  • YOLOE model loading (lazy singleton)
  • Per-frame inference + class name mapping to backend standard names

Called by YoloeDetector — this helper owns the model, prompts, and inference;
the detector owns the video loop, progress, GPS, DB etc.

Return contract for process_frame_with_yoloe():
    [
        {
            "class_name":  str,    # standard backend name (e.g. "defected_sign_board")
            "prompt_name": str,    # original YOLOE prompt
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
from ultralytics import YOLOE

logger = logging.getLogger(__name__)

# ── YOLOE Configuration ──────────────────────────────────────────────────────
YOLOE_MODEL_WEIGHTS = "yoloe-11m-seg.pt"
YOLOE_CONF_THRESHOLD = 0.65

# ── Open-vocabulary text prompts ──────────────────────────────────────────────
TARGET_PROMPTS = [
    # ── Intact / Good Signboards ──
    "clean traffic sign on pole",
    "intact road sign on metal post",
    "visible traffic signboard on road",
    "good condition municipal road sign",

    # ── Triangular Warning Signs ──
    "triangular warning traffic sign",
    "faded triangular road sign red border",
    "damaged triangle traffic sign on pole",
    "weathered triangular signboard",

    # ── Circular / Round Signs ──
    "circular traffic sign on road",
    "blank white circular traffic sign",
    "faded erased circular road sign",
    "shattered broken circular road sign",
    "damaged convex road mirror sign",
    "round prohibitory traffic sign",
    "circular no parking sign",

    # ── Rectangular / Informational Signs ──
    "rectangular road information sign",
    "faded bus stop signboard",
    "faded rectangular traffic sign pole",
    "blank white rectangular road sign",

    # ── Damaged / Defective Signboards (General) ──
    "damaged traffic signboard on road",
    "broken traffic signboard on pole",
    "faded traffic signboard on road",
    "blank white faded signboard",
    "rusted metal traffic sign",
    "bent traffic sign pole",
    "graffiti covered traffic sign",
    "cracked traffic signboard",
    "weathered signboard with peeling paint",

    # ── Commercial / Non-traffic Signboards ──
    "roadside advertisement billboard",
    "shop signboard near road",
    "commercial banner on roadside",

    # ── Road Defects ──
    "pothole on asphalt road",
    "deep road pothole",
    "disintegrating road surface patch",
    "loose gravel patch on road",
    "water puddle on asphalt road",
    "standing water on road surface",
    "longitudinal crack on asphalt",
    "road surface crack",
    "faded road lane marking",
    "worn lane line on road",
]

PROMPTS = TARGET_PROMPTS
NUM_TARGET_CLASSES = len(TARGET_PROMPTS)

# ── Display filter — only these indices are treated as defects ────────────────
DEFECTIVE_CLASS_INDICES = {
    TARGET_PROMPTS.index("faded triangular road sign red border"),
    TARGET_PROMPTS.index("damaged triangle traffic sign on pole"),
    TARGET_PROMPTS.index("weathered triangular signboard"),
    TARGET_PROMPTS.index("blank white circular traffic sign"),
    TARGET_PROMPTS.index("faded erased circular road sign"),
    TARGET_PROMPTS.index("shattered broken circular road sign"),
    TARGET_PROMPTS.index("damaged convex road mirror sign"),
    TARGET_PROMPTS.index("faded bus stop signboard"),
    TARGET_PROMPTS.index("faded rectangular traffic sign pole"),
    TARGET_PROMPTS.index("blank white rectangular road sign"),
    TARGET_PROMPTS.index("damaged traffic signboard on road"),
    TARGET_PROMPTS.index("broken traffic signboard on pole"),
    TARGET_PROMPTS.index("faded traffic signboard on road"),
    TARGET_PROMPTS.index("blank white faded signboard"),
    TARGET_PROMPTS.index("rusted metal traffic sign"),
    TARGET_PROMPTS.index("bent traffic sign pole"),
    TARGET_PROMPTS.index("graffiti covered traffic sign"),
    TARGET_PROMPTS.index("cracked traffic signboard"),
    TARGET_PROMPTS.index("weathered signboard with peeling paint"),
    TARGET_PROMPTS.index("pothole on asphalt road"),
    TARGET_PROMPTS.index("deep road pothole"),
    TARGET_PROMPTS.index("disintegrating road surface patch"),
    TARGET_PROMPTS.index("loose gravel patch on road"),
    TARGET_PROMPTS.index("water puddle on asphalt road"),
    TARGET_PROMPTS.index("standing water on road surface"),
    TARGET_PROMPTS.index("longitudinal crack on asphalt"),
    TARGET_PROMPTS.index("road surface crack"),
    TARGET_PROMPTS.index("faded road lane marking"),
    TARGET_PROMPTS.index("worn lane line on road"),
}

# ── Prompt → Backend standard class name mapping ─────────────────────────────
# Every prompt maps to one of the 5 backend classes used by the existing YOLO
# detector, so results are consistent across detection modes.
_PROMPT_TO_BACKEND_CLASS = {
    # Good signboards
    "clean traffic sign on pole": "good_sign_board",
    "intact road sign on metal post": "good_sign_board",
    "visible traffic signboard on road": "good_sign_board",
    "good condition municipal road sign": "good_sign_board",

    # Triangular (good = triangular warning, defective = damaged triangular)
    "triangular warning traffic sign": "good_sign_board",
    "faded triangular road sign red border": "defected_sign_board",
    "damaged triangle traffic sign on pole": "defected_sign_board",
    "weathered triangular signboard": "defected_sign_board",

    # Circular (generic = good, defective = defected)
    "circular traffic sign on road": "good_sign_board",
    "blank white circular traffic sign": "defected_sign_board",
    "faded erased circular road sign": "defected_sign_board",
    "shattered broken circular road sign": "defected_sign_board",
    "damaged convex road mirror sign": "defected_sign_board",
    "round prohibitory traffic sign": "good_sign_board",
    "circular no parking sign": "good_sign_board",

    # Rectangular (generic = good, defective = defected)
    "rectangular road information sign": "good_sign_board",
    "faded bus stop signboard": "defected_sign_board",
    "faded rectangular traffic sign pole": "defected_sign_board",
    "blank white rectangular road sign": "defected_sign_board",

    # Damaged / Defective signboards (all → defected_sign_board)
    "damaged traffic signboard on road": "defected_sign_board",
    "broken traffic signboard on pole": "defected_sign_board",
    "faded traffic signboard on road": "defected_sign_board",
    "blank white faded signboard": "defected_sign_board",
    "rusted metal traffic sign": "defected_sign_board",
    "bent traffic sign pole": "defected_sign_board",
    "graffiti covered traffic sign": "defected_sign_board",
    "cracked traffic signboard": "defected_sign_board",
    "weathered signboard with peeling paint": "defected_sign_board",

    # Commercial boards (silently filtered by DEFECTIVE_CLASS_INDICES)
    "roadside advertisement billboard": "good_sign_board",
    "shop signboard near road": "good_sign_board",
    "commercial banner on roadside": "good_sign_board",

    # Road defects
    "pothole on asphalt road": "pothole",
    "deep road pothole": "pothole",
    "disintegrating road surface patch": "pothole",
    "loose gravel patch on road": "pothole",
    "water puddle on asphalt road": "pothole",
    "standing water on road surface": "pothole",
    "longitudinal crack on asphalt": "road_crack",
    "road surface crack": "road_crack",
    "faded road lane marking": "damaged_road_marking",
    "worn lane line on road": "damaged_road_marking",
}

# Backend road-damage classes (same as YoloDetector)
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}
EXCLUDED_CLASSES = {"good_sign_board"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES


# ── Lazy singleton for YOLOE model ───────────────────────────────────────────
_model_instance: YOLOE | None = None
_model_lock = threading.Lock()


def load_yoloe_model() -> YOLOE:
    """Return the singleton YOLOE model, creating it on first call."""
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:  # double-checked
                logger.info(f"⏳ Loading YOLOE model: {YOLOE_MODEL_WEIGHTS}")
                t0 = time.time()
                model = YOLOE(YOLOE_MODEL_WEIGHTS)
                model.set_classes(PROMPTS)
                logger.info(
                    f"✅ YOLOE model loaded in {time.time() - t0:.1f}s — "
                    f"{NUM_TARGET_CLASSES} prompts set"
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


def process_frame_with_yoloe(model: YOLOE, frame) -> list[dict]:
    """
    Run YOLOE inference on a single frame and return filtered detections.

    Only detections whose class index is in DEFECTIVE_CLASS_INDICES are
    returned.  Good signs, generic signs, and commercial boards are silently
    filtered (but still sent to the model as prompts for better discrimination).

    Args:
        model: Loaded YOLOE model (from load_yoloe_model)
        frame: BGR numpy array

    Returns:
        List of detection dicts with keys:
            class_name, prompt_name, confidence, bbox, center
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
            # Filter 1: skip anything outside our target classes
            if cls_id >= NUM_TARGET_CLASSES:
                continue

            # Filter 2: skip non-defective classes
            if cls_id not in DEFECTIVE_CLASS_INDICES:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            prompt_name = names.get(cls_id, f"class_{cls_id}")
            backend_class = _PROMPT_TO_BACKEND_CLASS.get(
                prompt_name, "defected_sign_board"
            )

            detections.append({
                "class_name": backend_class,
                "prompt_name": prompt_name,
                "confidence": round(float(conf), 3),
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
            })

    return detections
