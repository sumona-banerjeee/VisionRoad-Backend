"""
YOLOE - Pothole Detection with Full Post-Detection Filter Pipeline
Model  : yoloe-11m-seg.pt

Post-Detection Filters (applied after every detection):
  1. Shadow Filter     → removes dark smooth regions (no texture)
  2. Size Filter       → removes tiny/huge boxes (noise or whole-frame matches)
  3. Texture Filter    → ROI must have rough texture (Laplacian variance)
  4. Aspect Filter     → removes very thin/long boxes (not pothole-shaped)
  5. Brightness Filter → removes very bright uniform patches (glare/white paint)
"""

import cv2
import os
import argparse
from ultralytics import YOLOE

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

MODEL_WEIGHTS        = "yoloe-11m-seg.pt"
INPUT_SOURCE         = r"C:\Users\sumon\OneDrive\Pictures\Screenshots\Screenshot 2026-03-11 133053.png"
OUTPUT_PATH          = r"Test\output\img1-output.jpg"
CONF_SIGNBOARD = 0.65   # confidence threshold for signboard detections
CONF_POTHOLE   = 0.10   # confidence threshold for pothole detections (filters handle FPs)

# ── Texture-Based Prompts (shadow-resistant) ──
TARGET_PROMPTS = [
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
    "roadside advertisement billboard",
    "shop signboard near road",
    "commercial banner on roadside",
    
    # Core potholes
    "pothole on road",
    "deep pothole on asphalt",
    "shallow pothole on road",
    "small pothole on asphalt",
    "pothole with rough sandy interior",

    # Exposed sublayer (white/sandy texture)
    "exposed aggregate patch on road",
    "white sandy rough patch on road surface",
    "rough granular white patch on asphalt",
    "crumbling asphalt with exposed aggregate",

    # Surface raveling (grey granular texture)
    "grey rough granular patch on asphalt",
    "surface raveling on road",
    "loose aggregate patch on asphalt",
    "coarse granular patch on road surface",
    "rough irregular worn patch on road",

    # Developing / edges
    "pothole with broken crumbling edges",
    "developing pothole forming in asphalt",

    # Multiple / severe
    "multiple potholes on road surface",
    "pothole with rough sandy interior texture",
]

PROMPTS            = TARGET_PROMPTS
NUM_TARGET_CLASSES = len(TARGET_PROMPTS)

# ── Only these pothole prompts are shown (all others silently dropped) ──
ACCEPTED_POTHOLE_PROMPTS = {
    "pothole with rough sandy interior",
    "pothole with rough sandy interior texture",
}

# ── Display label mapping ─────────────────────────────────────────────────────
# Defective signboard prompts → "Defective Signboard"
# Pothole prompts             → "Pothole"
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
    "rectangular road information sign",
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
    # Commercial
    "roadside advertisement billboard",
    "shop signboard near road",
    "commercial banner on roadside",
}

def get_display_label(prompt_name: str) -> str | None:
    """Map raw prompt to simplified display label. Returns None to skip."""
    if prompt_name in _DEFECTIVE_SIGN_PROMPTS:
        return "Defective Signboard"
    if prompt_name in ACCEPTED_POTHOLE_PROMPTS:
        return "Pothole"
    return None   # skip good signs, commercial, and non-accepted pothole prompts

def get_conf_threshold(prompt_name: str) -> float:
    """Return the correct confidence threshold for this prompt."""
    if prompt_name in _DEFECTIVE_SIGN_PROMPTS:
        return CONF_SIGNBOARD
    return CONF_POTHOLE

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

CLASS_COLOURS = [
    (0, 255, 255), (0, 200, 255), (0, 150, 255), (0, 100, 255),
    (255, 0, 255), (255, 50, 200), (255, 100, 150), (200, 0, 200),
    (0, 255, 150), (0, 255, 100), (100, 255, 0),  (150, 255, 0),
    (255, 200, 0), (255, 150, 0), (200, 100, 0),  (0, 180, 255),
    (180, 0, 255), (255, 0, 100),
]

# ══════════════════════════════════════════════
# POST-DETECTION FILTERS
# ══════════════════════════════════════════════

def get_roi(frame, x1, y1, x2, y2):
    """Safely clip ROI to frame bounds."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def filter_shadow(roi) -> tuple[bool, str]:
    """
    FILTER 1 — Shadow
    Shadow = dark (mean < 80) AND smooth (std < 18).
    Real pothole = rough texture → high std even if dark.
    """
    if roi.size == 0:
        return True, "empty ROI"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val  = float(gray.std())
    if mean_val < 80 and std_val < 18:
        return True, f"shadow [mean={mean_val:.1f}, std={std_val:.1f}]"
    return False, ""


def filter_size(x1, y1, x2, y2, frame_w, frame_h) -> tuple[bool, str]:
    """
    FILTER 2 — Size
    Too small  (< 0.3% of frame) → noise/dust spec.
    Too large  (> 60% of frame)  → whole-frame false match.
    """
    box_area   = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    ratio      = box_area / frame_area
    if ratio < 0.003:
        return True, f"too small [area={ratio:.4f}]"
    if ratio > 0.60:
        return True, f"too large [area={ratio:.2f}]"
    return False, ""


def filter_texture(roi) -> tuple[bool, str]:
    """
    FILTER 3 — Texture (Laplacian variance)
    Real potholes/raveling = rough surface → high variance.
    Shadows/paint/glare    = smooth        → low variance.
    Threshold: variance < 12 → too smooth.
    """
    if roi.size == 0:
        return True, "empty ROI"
    gray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance  = float(laplacian.var())
    if variance < 12:
        return True, f"too smooth [laplacian={variance:.1f}]"
    return False, ""


def filter_aspect(x1, y1, x2, y2) -> tuple[bool, str]:
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


def filter_brightness(roi) -> tuple[bool, str]:
    """
    FILTER 5 — Brightness
    Very bright + smooth = road paint / glare / white marking.
    Bright (mean > 210) AND smooth (std < 15) → not a pothole.
    """
    if roi.size == 0:
        return True, "empty ROI"
    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val  = float(gray.std())
    if mean_val > 210 and std_val < 15:
        return True, f"bright uniform [mean={mean_val:.1f}, std={std_val:.1f}]"
    return False, ""


def run_all_filters(frame, x1, y1, x2, y2) -> tuple[bool, str]:
    """Run all 5 filters. Returns (should_skip, reason)."""
    h, w = frame.shape[:2]
    roi  = get_roi(frame, x1, y1, x2, y2)
    for fn in [
        lambda: filter_shadow(roi),
        lambda: filter_size(x1, y1, x2, y2, w, h),
        lambda: filter_texture(roi),
        lambda: filter_aspect(x1, y1, x2, y2),
        lambda: filter_brightness(roi),
    ]:
        failed, reason = fn()
        if failed:
            return True, reason
    return False, ""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_model(weights: str = MODEL_WEIGHTS) -> YOLOE:
    print(f"[INFO] Loading model : {weights}")
    model = YOLOE(weights)
    model.set_classes(PROMPTS)
    print(f"[INFO] Prompts loaded: {len(PROMPTS)}")
    return model


def get_colour(class_id: int) -> tuple:
    return CLASS_COLOURS[class_id % len(CLASS_COLOURS)]


def draw_and_log(frame, results, source_label: str = ""):
    """Draw only detections that pass display-label + confidence + post-filters."""
    count    = 0
    passed   = []
    filtered = []
    skipped  = []

    if results.boxes is not None and len(results.boxes) > 0:
        boxes     = results.boxes.xyxy.cpu().numpy()
        confs     = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        names     = results.names

        for box, conf, cls_id in zip(boxes, confs, class_ids):
            if cls_id >= NUM_TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            prompt_name     = names.get(cls_id, f"class_{cls_id}")

            # ── Step 1: Map to display label (skip unwanted prompts) ──
            display_label = get_display_label(prompt_name)
            if display_label is None:
                skipped.append((prompt_name, conf, "not an accepted prompt"))
                continue

            # ── Step 2: Per-category confidence threshold ──
            min_conf = get_conf_threshold(prompt_name)
            if conf < min_conf:
                skipped.append((prompt_name, conf, f"below conf {min_conf}"))
                continue

            # ── Step 3: Run all 5 post-detection filters (pothole only) ──
            if display_label == "Pothole":
                skip, reason = run_all_filters(frame, x1, y1, x2, y2)
                if skip:
                    filtered.append((prompt_name, conf, reason))
                    continue

            # ── Passed everything → draw on frame ──
            colour     = get_colour(cls_id)
            label_text = f"{display_label}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            passed.append((display_label, prompt_name, conf))
            count += 1

    # ── Terminal output ──
    print(f"\n{'='*65}")
    print(f"  SOURCE   : {source_label}")
    if passed:
        print(f"  DETECTED : {len(passed)} detection(s)  ✅")
        print(f"{'─'*65}")
        for i, (disp, prompt, conf) in enumerate(passed, 1):
            print(f"  [{i:02d}] ✅  {disp}  (conf: {conf:.2f})  ← \"{prompt}\"")
    else:
        print(f"  DETECTED : 0  ❌")

    if filtered:
        print(f"{'─'*65}")
        print(f"  FILTERED : {len(filtered)} rejected by post-filters")
        for prompt, conf, reason in filtered:
            print(f"       ⛔  \"{prompt}\"  conf={conf:.2f}  → {reason}")
    if skipped:
        print(f"  SKIPPED  : {len(skipped)} (unwanted prompt or low conf)")
    print(f"{'='*65}")

    return frame, count


# ──────────────────────────────────────────────
# Image detection
# ──────────────────────────────────────────────

def detect_image(model: YOLOE, image_path: str, save_path: str | None = None):
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    print(f"\n[INFO] Detecting on: {image_path}")
    img     = cv2.imread(image_path)
    results = model.predict(img, conf=CONF_POTHOLE, imgsz=1280)
    annotated, det_count = draw_and_log(
        img, results[0], source_label=os.path.basename(image_path)
    )
    print(f"[INFO] Final detections after filters: {det_count}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, annotated)
        print(f"[INFO] Saved -> {save_path}")

    cv2.imshow("YOLOE Pothole Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Video detection
# ──────────────────────────────────────────────

def detect_video(model: YOLOE, video_path: str,
                 save_path: str | None = None, show: bool = True):
    source = 0 if video_path == "0" else video_path
    if isinstance(source, str) and not os.path.isfile(source):
        print(f"[ERROR] Video not found: {source}")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}x{height} @ {fps:.1f} FPS | {total} frames")

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        print(f"[INFO] Saving -> {save_path}")

    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            results    = model.predict(frame, conf=CONF_POTHOLE,
                                       imgsz=1280, verbose=False)
            annotated, det_count = draw_and_log(
                frame, results[0],
                source_label=f"Frame {frame_num}/{total}"
            )

            cv2.putText(annotated, f"Frame: {frame_num} | Det: {det_count}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("YOLOE Pothole Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\n[DONE] {frame_num} frames processed.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOE Pothole Detection")
    parser.add_argument("source",    nargs="?",  default=INPUT_SOURCE)
    parser.add_argument("--weights", default=MODEL_WEIGHTS)
    parser.add_argument("--conf",    type=float, default=CONF_POTHOLE)
    parser.add_argument("--save",    default=OUTPUT_PATH)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    model = load_model(args.weights)
    ext   = os.path.splitext(args.source)[1].lower()

    if args.source == "0" or ext in VIDEO_EXTENSIONS:
        detect_video(model, args.source, save_path=args.save, show=not args.no_show)
    elif ext in IMAGE_EXTENSIONS:
        detect_image(model, args.source, save_path=args.save)
    else:
        print(f"[ERROR] Unsupported format: {ext}")


if __name__ == "__main__":
    main()








# """
# YOLOE - Pothole & Surface Defect Detection
# Model  : yoloe-11m-seg.pt
# Fix    : Shadow false-positives removed via texture check + conf=0.35
# """

# import cv2
# import os
# import argparse
# from ultralytics import YOLOE

# # ──────────────────────────────────────────────
# # Configuration
# # ──────────────────────────────────────────────

# MODEL_WEIGHTS        = "yoloe-11m-seg.pt"
# INPUT_SOURCE         = r"Test\img\img3.png"
# OUTPUT_PATH          = r"Test\output\img3-output.jpg"
# CONFIDENCE_THRESHOLD = 0.10   # raised from 0.20 → filters shadow false-positives

# # ── Shadow-Resistant Prompts (texture-based only) ──
# TARGET_PROMPTS = [
#     # Core potholes
#     "pothole on road",
#     "deep pothole on asphalt",
#     "shallow pothole on road",
#     "small pothole on asphalt",
#     "pothole with rough sandy interior",

#     # Exposed sublayer (white/sandy texture)
#     "exposed aggregate patch on road",
#     "white sandy rough patch on road surface",
#     "rough granular white patch on asphalt",
#     "crumbling asphalt with exposed aggregate",

#     # Surface raveling (grey granular texture)
#     "grey rough granular patch on asphalt",
#     "surface raveling on road",
#     "loose aggregate patch on asphalt",
#     "coarse granular patch on road surface",
#     "rough irregular worn patch on road",

#     # Developing / edges
#     "pothole with broken crumbling edges",
#     "developing pothole forming in asphalt",

#     # Multiple / severe
#     "multiple potholes on road surface",
#     "pothole with rough sandy interior texture",
# ]

# PROMPTS            = TARGET_PROMPTS
# NUM_TARGET_CLASSES = len(TARGET_PROMPTS)

# VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v")
# IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# CLASS_COLOURS = [
#     (0, 255, 255), (0, 200, 255), (0, 150, 255), (0, 100, 255),
#     (255, 0, 255), (255, 50, 200), (255, 100, 150), (200, 0, 200),
#     (0, 255, 150), (0, 255, 100), (100, 255, 0),  (150, 255, 0),
#     (255, 200, 0), (255, 150, 0), (200, 100, 0),  (0, 180, 255),
#     (180, 0, 255), (255, 0, 100),
# ]

# # ──────────────────────────────────────────────
# # Shadow Filter
# # ──────────────────────────────────────────────

# def is_shadow(frame, x1, y1, x2, y2) -> bool:
#     """
#     Returns True if the detected ROI is likely a shadow (not a real pothole).

#     Logic:
#       - Shadows are DARK (low mean brightness) AND SMOOTH (low texture variance)
#       - Real potholes have rough texture → high std deviation even if dark
      
#     Thresholds:
#       mean  < 80  →  dark region
#       std   < 18  →  smooth / no texture  →  shadow
#     """
#     roi = frame[y1:y2, x1:x2]
#     if roi.size == 0:
#         return False
#     gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     mean_val = float(gray.mean())
#     std_val  = float(gray.std())
#     return mean_val < 80 and std_val < 18


# # ──────────────────────────────────────────────
# # Helpers
# # ──────────────────────────────────────────────

# def load_model(weights: str = MODEL_WEIGHTS) -> YOLOE:
#     print(f"[INFO] Loading model : {weights}")
#     model = YOLOE(weights)
#     model.set_classes(PROMPTS)
#     print(f"[INFO] Prompts loaded: {len(PROMPTS)}")
#     return model


# def get_colour(class_id: int) -> tuple:
#     return CLASS_COLOURS[class_id % len(CLASS_COLOURS)]


# def draw_and_log(frame, results, source_label: str = ""):
#     """Draw detections, skip shadows, log matched prompts to terminal."""
#     count           = 0
#     matched_prompts = []
#     skipped_shadows = 0

#     if results.boxes is not None and len(results.boxes) > 0:
#         boxes     = results.boxes.xyxy.cpu().numpy()
#         confs     = results.boxes.conf.cpu().numpy()
#         class_ids = results.boxes.cls.cpu().numpy().astype(int)
#         names     = results.names

#         for box, conf, cls_id in zip(boxes, confs, class_ids):
#             if cls_id >= NUM_TARGET_CLASSES:
#                 continue

#             x1, y1, x2, y2 = map(int, box)

#             # ── Shadow filter ──
#             if is_shadow(frame, x1, y1, x2, y2):
#                 skipped_shadows += 1
#                 label = names.get(cls_id, f"class_{cls_id}")
#                 print(f"  [SHADOW FILTERED] \"{label}\"  conf={conf:.2f}  box=({x1},{y1},{x2},{y2})")
#                 continue

#             colour     = get_colour(cls_id)
#             label      = names.get(cls_id, f"class_{cls_id}")
#             label_text = f"{label}: {conf:.2f}"

#             cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
#             (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
#             cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)
#             cv2.putText(frame, label_text, (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

#             matched_prompts.append((label, conf))
#             count += 1

#     # ── Terminal output ──
#     if matched_prompts:
#         print(f"\n{'='*60}")
#         print(f"  SOURCE  : {source_label}")
#         print(f"  MATCHED : {len(matched_prompts)} detection(s)"
#               + (f"  |  {skipped_shadows} shadow(s) filtered" if skipped_shadows else ""))
#         print(f"{'─'*60}")
#         for i, (prompt, conf) in enumerate(matched_prompts, 1):
#             print(f"  [{i:02d}] ✅  \"{prompt}\"  (conf: {conf:.2f})")
#         print(f"{'='*60}")
#     else:
#         shadow_note = f"  ({skipped_shadows} shadow(s) filtered)" if skipped_shadows else ""
#         print(f"  [{source_label}] ❌  No pothole detected.{shadow_note}")

#     return frame, count


# # ──────────────────────────────────────────────
# # Image detection
# # ──────────────────────────────────────────────

# def detect_image(model: YOLOE, image_path: str, save_path: str | None = None):
#     if not os.path.isfile(image_path):
#         print(f"[ERROR] Image not found: {image_path}")
#         return

#     print(f"\n[INFO] Detecting on: {image_path}")
#     img     = cv2.imread(image_path)
#     results = model.predict(img, conf=CONFIDENCE_THRESHOLD, imgsz=1280)
#     annotated, det_count = draw_and_log(img, results[0],
#                                         source_label=os.path.basename(image_path))
#     print(f"[INFO] Total detections: {det_count}")

#     if save_path:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         cv2.imwrite(save_path, annotated)
#         print(f"[INFO] Saved -> {save_path}")

#     cv2.imshow("YOLOE Pothole Detection", annotated)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# # ──────────────────────────────────────────────
# # Video detection
# # ──────────────────────────────────────────────

# def detect_video(model: YOLOE, video_path: str,
#                  save_path: str | None = None, show: bool = True):
#     source = 0 if video_path == "0" else video_path
#     if isinstance(source, str) and not os.path.isfile(source):
#         print(f"[ERROR] Video not found: {source}")
#         return

#     cap = cv2.VideoCapture(source)
#     if not cap.isOpened():
#         print("[ERROR] Could not open video.")
#         return

#     fps    = cap.get(cv2.CAP_PROP_FPS) or 30
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"[INFO] {width}x{height} @ {fps:.1f} FPS | {total} frames")

#     writer = None
#     if save_path:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
#                                  fps, (width, height))
#         print(f"[INFO] Saving -> {save_path}")

#     frame_num = 0
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_num += 1
#             results    = model.predict(frame, conf=CONFIDENCE_THRESHOLD,
#                                        imgsz=1280, verbose=False)
#             annotated, det_count = draw_and_log(
#                 frame, results[0],
#                 source_label=f"Frame {frame_num}/{total}"
#             )

#             cv2.putText(annotated, f"Frame: {frame_num} | Det: {det_count}",
#                         (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             if writer:
#                 writer.write(annotated)
#             if show:
#                 cv2.imshow("YOLOE Pothole Detection", annotated)
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break

#     except KeyboardInterrupt:
#         print("\n[INFO] Stopped by user.")
#     finally:
#         cap.release()
#         if writer:
#             writer.release()
#         cv2.destroyAllWindows()

#     print(f"\n[DONE] {frame_num} frames processed.")


# # ──────────────────────────────────────────────
# # Main
# # ──────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(description="YOLOE Pothole Detection")
#     parser.add_argument("source",    nargs="?",  default=INPUT_SOURCE)
#     parser.add_argument("--weights", default=MODEL_WEIGHTS)
#     parser.add_argument("--conf",    type=float, default=CONFIDENCE_THRESHOLD)
#     parser.add_argument("--save",    default=OUTPUT_PATH)
#     parser.add_argument("--no-show", action="store_true")
#     args = parser.parse_args()

#     model = load_model(args.weights)
#     ext   = os.path.splitext(args.source)[1].lower()

#     if args.source == "0" or ext in VIDEO_EXTENSIONS:
#         detect_video(model, args.source, save_path=args.save, show=not args.no_show)
#     elif ext in IMAGE_EXTENSIONS:
#         detect_image(model, args.source, save_path=args.save)
#     else:
#         print(f"[ERROR] Unsupported format: {ext}")


# if __name__ == "__main__":
#     main()