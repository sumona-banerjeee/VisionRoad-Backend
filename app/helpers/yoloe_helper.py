"""
YOLOE Helper — Open-vocabulary detection with text prompts.

Encapsulates all YOLOE-specific logic:
  • 56 text prompts across sign + pothole + drain categories
  • Triple confidence thresholds (signboard=0.65, pothole=0.10, drain=0.10)
  • 5 post-detection filters for pothole FP removal
  • Display label mapping (Defective Signboard / Pothole / Drain Issue)
  • YOLOE model loading (lazy singleton)
  • Per-frame inference with full filter pipeline

Called by YoloeDetector — this helper owns the model, prompts, and inference;
the detector owns the video loop, progress, GPS, DB etc.

Return contract for process_frame_with_yoloe():
    [
        {
            "class_name":  str,    # backend name ("defected_sign_board" / "pothole" / "drain_issue")
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

import cv2
import numpy as np
from ultralytics import YOLOE

logger = logging.getLogger(__name__)

# ── YOLOE Configuration ──────────────────────────────────────────────────────
YOLOE_MODEL_WEIGHTS = "models/yoloe-11m-seg.pt"
YOLOE_CONF_SIGNBOARD = 0.65   # confidence threshold for signboard detections
YOLOE_CONF_POTHOLE   = 0.10   # confidence threshold for pothole detections
YOLOE_CONF_DRAIN     = 0.10   # confidence threshold for drain/water issue detections
# Minimum conf used for model.predict() — post-filtering applies per-category
YOLOE_CONF_THRESHOLD = YOLOE_CONF_POTHOLE

# ── Open-vocabulary text prompts ──────────────────────────────────────────────
TARGET_PROMPTS = [
    # ── Intact / Good Signboards ──
    "clean traffic sign on pole",
    "intact road sign on metal post",
    "visible traffic signboard on road",
    "good condition municipal road sign",          # Indian municipal context

    # ── Triangular Warning Signs ──  ← NEW CATEGORY
    "triangular warning traffic sign",
    "faded triangular road sign red border",
    "damaged triangle traffic sign on pole",
    "weathered triangular signboard",

    # ── Circular / Round Signs ──
    "circular traffic sign on road",
    "blank white circular traffic sign",           # covers Image 3
    "faded erased circular road sign",             # covers Image 3 specifically
    "shattered broken circular road sign",         # covers Image 2
    "damaged convex road mirror sign",             # covers Image 2 specifically
    "round prohibitory traffic sign",
    "circular no parking sign",                    # covers Image 6 background

    # ── Rectangular / Informational Signs ──      ← NEW CATEGORY
    "rectangular road information sign",
    "faded bus stop signboard",                    # covers Image 6
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
    "weathered signboard with peeling paint",      # NEW — covers Images 1, 5

    # ── Commercial / Non-traffic Signboards ──
    "advertisement billboard",
    "roadside advertisement billboard",
    "shop signboard near road",
    "commercial banner on roadside",
    "large outdoor advertising billboard",
    "commercial hoarding on metal frame",
    "branded advertisement hoarding",

    # ── Core potholes ──
    "pothole on road",
    "deep pothole on asphalt",
    "shallow pothole on road",
    "small pothole on asphalt",
    "pothole with rough sandy interior",

    # ── Exposed sublayer (white/sandy texture) ──
    "exposed aggregate patch on road",
    "white sandy rough patch on road surface",
    "rough granular white patch on asphalt",
    "crumbling asphalt with exposed aggregate",

    # ── Surface raveling (grey granular texture) ──
    "grey rough granular patch on asphalt",
    "surface raveling on road",
    "loose aggregate patch on asphalt",
    "coarse granular patch on road surface",
    "rough irregular worn patch on road",

    # ── Developing / edges ──
    "pothole with broken crumbling edges",
    "developing pothole forming in asphalt",

    # ── Multiple / severe ──
    "multiple potholes on road surface",
    "pothole with rough sandy interior texture",

    # ── Upright pole prompts ──
    "straight vertical traffic sign pole",
    "upright traffic sign on straight pole",
    "traffic sign on vertical metal post",

    # ── Bent pole prompts ──
    "tilted traffic sign pole",
    "bent signpost",
    "leaning traffic sign pole",
    "crooked road sign pole",

    # ── Drain / water issue prompts ──
    "drain opening",
    "broken drain",
    "missing drain cover",
    "clogged drain",
    "overflowing drain",
    "water clogging",
    "garbage blockage",
]

PROMPTS = TARGET_PROMPTS
NUM_TARGET_CLASSES = len(TARGET_PROMPTS)

# ── Only these pothole prompts are accepted (all others silently dropped) ─────
ACCEPTED_POTHOLE_PROMPTS = {
    "pothole with rough sandy interior",
    "pothole with rough sandy interior texture",
}

# ── Good sign prompts (skipped entirely) ──────────────────────────────────────
_GOOD_SIGN_PROMPTS = {
    "clean traffic sign on pole",
    "intact road sign on metal post",
    "visible traffic signboard on road",
    "good condition municipal road sign",
    "roadside advertisement billboard",
    "shop signboard near road",
    "rectangular road information sign",
    "commercial banner on roadside",
    "large outdoor advertising billboard",
    "commercial hoarding on metal frame",
    "branded advertisement hoarding",
    "straight vertical traffic sign pole",
    "upright traffic sign on straight pole",
    "traffic sign on vertical metal post",
}

# ── All other sign prompts → "defected_sign_board" ───────────────────────────
_DEFECTIVE_SIGN_PROMPTS = {
    # Triangular
    "triangular warning traffic sign",
    "faded triangular road sign red border",
    "damaged triangle traffic sign on pole",
    "weathered triangular signboard",
    # Circular
    "circular traffic sign on road",
    "blank white circular traffic sign",
    "faded erased circular road sign",
    "shattered broken circular road sign",
    "damaged convex road mirror sign",
    "round prohibitory traffic sign",
    "circular no parking sign",
    # Rectangular
    "faded bus stop signboard",
    "faded rectangular traffic sign pole",
    "blank white rectangular road sign",
    # Damaged General
    "damaged traffic signboard on road",
    "broken traffic signboard on pole",
    "faded traffic signboard on road",
    "blank white faded signboard",
    "rusted metal traffic sign",
    "bent traffic sign pole",
    "graffiti covered traffic sign",
    "cracked traffic signboard",
    "weathered signboard with peeling paint",
    "tilted traffic sign pole",
    "bent signpost",
    "leaning traffic sign pole",
    "crooked road sign pole",
}

# ── Drain / water issue prompts → "drain_issue" ────────────────────────
_DRAIN_PROMPTS = {
    "drain opening",
    "broken drain",
    "missing drain cover",
    "clogged drain",
    "overflowing drain",
    "water clogging",
    "garbage blockage",
}


def get_display_label(prompt_name: str) -> str | None:
    """Map raw prompt to backend class name. Returns None to skip."""
    if prompt_name in _DEFECTIVE_SIGN_PROMPTS:
        return "defected_sign_board"
    if prompt_name in ACCEPTED_POTHOLE_PROMPTS:
        return "pothole"
    if prompt_name in _DRAIN_PROMPTS:
        return "drain_issue"
    return None  # skip good signs and non-accepted pothole prompts


def get_conf_threshold(prompt_name: str) -> float:
    """Return the correct confidence threshold for this prompt."""
    if prompt_name in _DEFECTIVE_SIGN_PROMPTS:
        return YOLOE_CONF_SIGNBOARD
    if prompt_name in _DRAIN_PROMPTS:
        return YOLOE_CONF_DRAIN
    return YOLOE_CONF_POTHOLE


# Backend road-damage classes (same as YoloDetector)
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "drain_issue",
}
ALL_CLASSES = ROAD_DAMAGE_CLASSES


# ══════════════════════════════════════════════
# POST-DETECTION FILTERS (pothole only)
# ══════════════════════════════════════════════

def _get_roi(frame, x1, y1, x2, y2):
    """Safely clip ROI to frame bounds."""
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    return frame[y1c:y2c, x1c:x2c]


def _filter_shadow(roi) -> tuple[bool, str]:
    """
    FILTER 1 — Shadow
    Shadow = dark (mean < 80) AND smooth (std < 18).
    Real pothole = rough texture → high std even if dark.
    """
    if roi.size == 0:
        return True, "empty ROI"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    if mean_val < 80 and std_val < 18:
        return True, f"shadow [mean={mean_val:.1f}, std={std_val:.1f}]"
    return False, ""


def _filter_size(x1, y1, x2, y2, frame_w, frame_h) -> tuple[bool, str]:
    """
    FILTER 2 — Size
    Too small  (< 0.3% of frame) → noise/dust spec.
    Too large  (> 60% of frame)  → whole-frame false match.
    """
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    ratio = box_area / frame_area
    if ratio < 0.003:
        return True, f"too small [area={ratio:.4f}]"
    if ratio > 0.60:
        return True, f"too large [area={ratio:.2f}]"
    return False, ""


def _filter_texture(roi) -> tuple[bool, str]:
    """
    FILTER 3 — Texture (Laplacian variance)
    Real potholes/raveling = rough surface → high variance.
    Shadows/paint/glare    = smooth        → low variance.
    Threshold: variance < 12 → too smooth.
    """
    if roi.size == 0:
        return True, "empty ROI"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    if variance < 12:
        return True, f"too smooth [laplacian={variance:.1f}]"
    return False, ""


def _filter_aspect(x1, y1, x2, y2) -> tuple[bool, str]:
    """
    FILTER 4 — Aspect Ratio
    Potholes are blob-shaped (ratio 0.2–5.0).
    Very thin/long boxes = road markings, cracks, poles.
    """
    w = x2 - x1
    h = y2 - y1
    if h == 0:
        return True, "zero height"
    aspect = w / h
    if aspect < 0.2 or aspect > 5.0:
        return True, f"bad aspect [{aspect:.2f}]"
    return False, ""


def _filter_brightness(roi) -> tuple[bool, str]:
    """
    FILTER 5 — Brightness
    Very bright + smooth = road paint / glare / white marking.
    Bright (mean > 210) AND smooth (std < 15) → not a pothole.
    """
    if roi.size == 0:
        return True, "empty ROI"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    if mean_val > 210 and std_val < 15:
        return True, f"bright uniform [mean={mean_val:.1f}, std={std_val:.1f}]"
    return False, ""


def _run_pothole_filters(frame, x1, y1, x2, y2) -> tuple[bool, str]:
    """Run all 5 pothole post-detection filters. Returns (should_skip, reason)."""
    h, w = frame.shape[:2]
    roi = _get_roi(frame, x1, y1, x2, y2)
    for fn in [
        lambda: _filter_shadow(roi),
        lambda: _filter_size(x1, y1, x2, y2, w, h),
        lambda: _filter_texture(roi),
        lambda: _filter_aspect(x1, y1, x2, y2),
        lambda: _filter_brightness(roi),
    ]:
        failed, reason = fn()
        if failed:
            return True, reason
    return False, ""


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

    Pipeline per detection:
      1. Skip if outside target class range
      2. Map prompt → display label (skip good signs & non-accepted pothole)
      3. Apply per-category confidence threshold (signboard=0.65, pothole=0.10)
      4. For potholes only: run 5 post-detection filters (shadow/size/texture/aspect/brightness)

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
            # Filter 1: skip anything outside target classes
            if cls_id >= NUM_TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            prompt_name = names.get(cls_id, f"class_{cls_id}")

            # Filter 2: map to backend class (skip unwanted prompts)
            backend_class = get_display_label(prompt_name)
            if backend_class is None:
                continue

            # Filter 3: per-category confidence threshold
            min_conf = get_conf_threshold(prompt_name)
            if conf < min_conf:
                continue

            # Filter 4: post-detection filters (pothole only)
            if backend_class == "pothole":
                skip, reason = _run_pothole_filters(frame, x1, y1, x2, y2)
                if skip:
                    logger.debug(
                        f"Pothole filtered: \"{prompt_name}\" conf={conf:.2f} → {reason}"
                    )
                    continue

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            detections.append({
                "class_name": backend_class,
                "prompt_name": prompt_name,
                "confidence": round(float(conf), 3),
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
            })

    return detections