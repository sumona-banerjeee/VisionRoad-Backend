"""
YOLOE - Open Vocabulary Real-Time Object Detection
Detects: signboards (good/damaged), potholes, puddles, road damage from images and videos.

Uses Ultralytics YOLOE with text prompts (open-vocabulary detection).
Docs: https://docs.ultralytics.com/models/yoloe/
"""

import cv2
import sys
import os
import argparse
from ultralytics import YOLOE


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

MODEL_WEIGHTS = "yoloe-11m-seg.pt"

# ── Input / Output paths ──
INPUT_SOURCE = r"C:\Users\Administrator\Desktop\AiML\Sumona\VisionRoad-Backend\Test\video\live-vid-2.mp4"
OUTPUT_PATH  = r"C:\Users\Administrator\Desktop\AiML\Sumona\VisionRoad-Backend\Test\uml.mp4"

# ── Open-vocabulary text prompts ──
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
    "faded worn circular traffic sign",
    "shattered broken circular road sign",
    "damaged convex road mirror sign",
    "round prohibitory traffic sign",
    "circular no parking sign",

    # ── Rectangular / Informational Signs ──
    "rectangular road information sign",
    "faded bus stop signboard",
    "faded rectangular traffic sign pole",
    "erased blank rectangular traffic sign on pole",

    # ── Damaged / Defective Signboards (General) ──
    "damage traffic signboard",
    "defective signboard",
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
    "parking sign on road",

    # ── Potholes & Localized Road Damage ──
    "small localized pothole",
    "pothole",
    "road pothole",
    "damaged road",
    "broken road surface",
    "peeling asphalt road",
    "exposed gravel on road",
    "eroded damaged pavement",
    "small patch of broken road",

    # ── Cracks ──
    "road surface crack",
    "cracked pavement",

    # ── Lane Markings ──
    "faded road lane marking",
    "worn lane line on road",

    # ── Negative / Contrastive Prompts (Not flagged as defects) ──
    "smooth clean asphalt road",
    "shadows on road surface",
    "clear dark pavement",
    "dry asphalt road",
    "sunlight on road surface",
]

PROMPTS = TARGET_PROMPTS
NUM_TARGET_CLASSES = len(TARGET_PROMPTS)

# ── Display filter ──────────────────────────────────────────────────────────
# Only classes listed here will be drawn on output frames.
# All prompts are still sent to the model unchanged — they help open-vocabulary
# discrimination. This set purely controls what is visible in the output.
#
# Excluded from display (detected internally but never drawn):
#   • "clean traffic sign on pole"
#   • "intact road sign on metal post"
#   • "visible traffic signboard on road"
#   • "good condition municipal road sign"
#   • "circular traffic sign on road"        (generic, non-defective)
#   • "round prohibitory traffic sign"       (generic, non-defective)
#   • "circular no parking sign"             (generic, non-defective)
#   • "rectangular road information sign"    (generic, non-defective)
#   • "roadside advertisement billboard"
#   • "shop signboard near road"
#   • "commercial banner on roadside"
# ───────────────────────────────────────────────────────────────────────────
DEFECTIVE_CLASS_INDICES = {
    # ── Defective Signs ──
    TARGET_PROMPTS.index("faded triangular road sign red border"),
    TARGET_PROMPTS.index("damaged triangle traffic sign on pole"),
    TARGET_PROMPTS.index("weathered triangular signboard"),
    TARGET_PROMPTS.index("faded worn circular traffic sign"),
    TARGET_PROMPTS.index("shattered broken circular road sign"),
    TARGET_PROMPTS.index("damaged convex road mirror sign"),
    TARGET_PROMPTS.index("faded bus stop signboard"),
    TARGET_PROMPTS.index("faded rectangular traffic sign pole"),
    TARGET_PROMPTS.index("erased blank rectangular traffic sign on pole"),
    TARGET_PROMPTS.index("damage traffic signboard"),
    TARGET_PROMPTS.index("defective signboard"),
    TARGET_PROMPTS.index("damaged traffic signboard on road"),
    TARGET_PROMPTS.index("broken traffic signboard on pole"),
    TARGET_PROMPTS.index("faded traffic signboard on road"),
    TARGET_PROMPTS.index("blank white faded signboard"),
    TARGET_PROMPTS.index("rusted metal traffic sign"),
    TARGET_PROMPTS.index("bent traffic sign pole"),
    TARGET_PROMPTS.index("graffiti covered traffic sign"),
    TARGET_PROMPTS.index("cracked traffic signboard"),
    TARGET_PROMPTS.index("weathered signboard with peeling paint"),
    # ── Road Defects ──
    TARGET_PROMPTS.index("small localized pothole"),
    TARGET_PROMPTS.index("pothole"),
    TARGET_PROMPTS.index("road pothole"),
    TARGET_PROMPTS.index("damaged road"),
    TARGET_PROMPTS.index("broken road surface"),
    TARGET_PROMPTS.index("peeling asphalt road"),
    TARGET_PROMPTS.index("exposed gravel on road"),
    TARGET_PROMPTS.index("eroded damaged pavement"),
    TARGET_PROMPTS.index("small patch of broken road"),
    TARGET_PROMPTS.index("road surface crack"),
    TARGET_PROMPTS.index("cracked pavement"),
    TARGET_PROMPTS.index("faded road lane marking"),
    TARGET_PROMPTS.index("worn lane line on road"),
}

# ── Confidence thresholds ────────────────────────────────────────────────────
# Global threshold passed to model.predict() — kept low so pothole candidates
# are not discarded before they reach our per-class filter.
CONFIDENCE_THRESHOLD = 0.20

# Per-class thresholds applied in draw_detections():
#   • Road defects score very low (0.15–0.40) because CLIP was not trained
#     on road-surface textures. We accept 0.15+ and rely on spatial filtering.
#   • Signboards are visually distinct; 0.55+ avoids false positives.
POTHOLE_CONF_THRESHOLD = 0.15
SIGN_CONF_THRESHOLD    = 0.55

# ── Deduplication thresholds ────────────────────────────────────────────────
# Minimum distance (pixels) between two detections to consider them unique
MIN_DISTANCE_THRESHOLD = 150
# Time window (seconds) to consider detections of the same category as duplicates
TIME_THRESHOLD = 5.0

# ── Spatial constraints to reduce False Positives ────────────────────────────
# Maximum area of a detection relative to the frame area (e.g. 0.25 = 25%)
# Potholes are usually localized; huge boxes are often false detection of the whole road.
MAX_DETECTION_AREA_RATIO = 0.25
# Maximum aspect ratio (width/height or height/width) to filter out extremely thin boxes
MAX_ASPECT_RATIO = 6.0

# Supported file extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# ── Road-defect class indices (for per-class confidence filtering) ──────────
# These indices into TARGET_PROMPTS identify road-surface defects.
ROAD_DEFECT_INDICES = {
    TARGET_PROMPTS.index("small localized pothole"),
    TARGET_PROMPTS.index("pothole"),
    TARGET_PROMPTS.index("road pothole"),
    TARGET_PROMPTS.index("damaged road"),
    TARGET_PROMPTS.index("broken road surface"),
    TARGET_PROMPTS.index("peeling asphalt road"),
    TARGET_PROMPTS.index("exposed gravel on road"),
    TARGET_PROMPTS.index("eroded damaged pavement"),
    TARGET_PROMPTS.index("small patch of broken road"),
    TARGET_PROMPTS.index("road surface crack"),
    TARGET_PROMPTS.index("cracked pavement"),
    TARGET_PROMPTS.index("faded road lane marking"),
    TARGET_PROMPTS.index("worn lane line on road"),
}

# ── Colours per class (BGR) ──
CLASS_COLOURS = [
    # ── Good Signboards (4) ──
    (0, 255, 0),       # Lime Green       — clean traffic sign on pole
    (0, 220, 0),       # Medium Green     — intact road sign on metal post
    (0, 200, 0),       # Green            — visible traffic signboard on road
    (0, 180, 0),       # Dark Green       — good condition municipal road sign

    # ── Triangular Signs (4) ──
    (0, 255, 128),     # Spring Green     — triangular warning traffic sign
    (0, 0, 255),       # Red              — faded triangular road sign
    (0, 80, 255),      # Orange-Red       — damaged triangle traffic sign
    (0, 50, 200),      # Dark Red         — weathered triangular signboard

    # ── Circular Signs (6) ──
    (100, 200, 255),   # Light Yellow     — circular traffic sign (good)
    (200, 0, 140),     # Deep Pink        — faded worn circular traffic sign
    (220, 0, 100),     # Crimson Pink     — shattered broken circular road sign
    (180, 0, 180),     # Magenta          — damaged convex road mirror sign
    (128, 200, 255),   # Pale Yellow      — round prohibitory traffic sign
    (100, 220, 200),   # Mint             — circular no parking sign

    # ── Rectangular Signs (4) ──
    (200, 200, 0),     # Teal             — rectangular road information sign
    (0, 140, 255),     # Orange           — faded bus stop signboard
    (0, 100, 220),     # Dark Orange      — faded rectangular traffic sign pole
    (0, 200, 255),     # Yellow           — erased blank rectangular traffic sign on pole

    # ── Damaged Signboards General (11) ──
    (0, 10, 240),      # Bright Red       — damage traffic signboard
    (0, 40, 220),      # Scarlet          — defective signboard
    (0, 0, 255),       # Red              — damaged traffic signboard on road
    (0, 80, 255),      # Orange-Red       — broken traffic signboard on pole
    (0, 200, 255),     # Yellow           — faded traffic signboard on road
    (0, 180, 200),     # Olive            — blank white faded signboard
    (0, 140, 255),     # Orange           — rusted metal traffic sign
    (0, 60, 200),      # Dark Orange-Red  — bent traffic sign pole
    (60, 0, 200),      # Dark Red-Purple  — graffiti covered traffic sign
    (0, 30, 180),      # Dark Red         — cracked traffic signboard
    (80, 0, 160),      # Purple           — weathered signboard with peeling paint

    # ── Commercial Boards (3) ──
    (0, 215, 255),     # Gold             — roadside advertisement billboard
    (0, 200, 220),     # Dark Yellow      — shop signboard near road
    (0, 160, 180),     # Dark Olive       — commercial banner on roadside
    (0, 150, 160),     # Olive-Brown      — parking sign on road

    # ── Potholes & Localized Road Damage (10) ──
    (255, 0, 255),     # Magenta          — small localized pothole
    (255, 0, 255),     # Magenta          — pothole
    (255, 0, 200),     # Hot Pink         — road pothole
    (255, 50, 150),    # Rose             — damaged road
    (150, 0, 200),     # Purple-Violet    — broken road surface
    (200, 0, 255),     # Violet           — peeling asphalt road
    (180, 50, 200),    # Orchid           — exposed gravel on road
    (160, 0, 220),     # Purple           — eroded damaged pavement
    (255, 80, 200),    # Pinkish          — small patch of broken road

    # ── Cracks (2) ──
    (255, 255, 0),     # Cyan             — road surface crack
    (200, 255, 0),     # Blue-Cyan        — cracked pavement

    # ── Lane markings (2) ──
    (255, 100, 0),     # Azure Blue       — faded road lane marking
    (200, 80, 0),      # Dark Azure       — worn lane line on road

    # ── Negative Contrastive (5) ──
    (128, 128, 128),   # Grey
    (100, 100, 100),   # Dark Grey
    (80, 80, 80),      # Darkest Grey
    (150, 150, 150),   # Light Grey
    (200, 200, 200),   # Very Light Grey
]


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def load_model(weights: str = MODEL_WEIGHTS) -> YOLOE:
    """Load YOLOE model and set open-vocabulary classes."""
    print(f"[INFO] Loading YOLOE model: {weights}")
    model = YOLOE(weights)
    model.set_classes(PROMPTS)
    print(f"[INFO] Classes set: {PROMPTS}")
    return model


def get_colour(class_id: int) -> tuple:
    """Return a BGR colour for a given class id."""
    return CLASS_COLOURS[class_id % len(CLASS_COLOURS)]


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def is_duplicate_location(cx, cy, class_id, current_time, spatial_locations):
    """
    Check if this location/class was already counted recently.
    Implements 'Sticky Tracking': if a duplicate is found, we update its 
    position and time to follow it as it moves across the screen.
    """
    found_duplicate = False
    
    # Signboards move faster and cover more distance in frame than potholes.
    # We increase the distance threshold for signs to catch them as they sweep past.
    is_sign = class_id not in ROAD_DEFECT_INDICES
    dist_threshold = MIN_DISTANCE_THRESHOLD * 2 if is_sign else MIN_DISTANCE_THRESHOLD

    for existing in spatial_locations:
        prev_cx, prev_cy = existing["center"]
        prev_class = existing["class_id"]
        prev_time = existing["time"]

        distance = calculate_distance((cx, cy), (prev_cx, prev_cy))
        time_gap = current_time - prev_time

        if (
            prev_class == class_id
            and distance < dist_threshold
            and time_gap < TIME_THRESHOLD
        ):
            # STICKY TRACKING: Update the historical record to the new position.
            # This allows us to "follow" the object and prevent it from ever
            # being far enough from "history" to be re-counted.
            existing["center"] = (cx, cy)
            existing["time"] = current_time
            found_duplicate = True
            break
            
    return found_duplicate


def draw_detections(frame, results):
    """
    Draw bounding boxes and labels on the frame.

    Only detections whose class index appears in DEFECTIVE_CLASS_INDICES
    are drawn. Good signs, generic signs, and commercial boards are detected
    internally (they improve model accuracy) but never shown in the output.

    Parameters
    ----------
    frame  : numpy array (BGR image)
    results: Ultralytics Results object (single element)

    Returns
    -------
    annotated_frame : numpy array with drawn detections
    detections_list : list of (cx, cy, cls_id, conf, box) for valid detections
    """
    valid_detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        h, w = frame.shape[:2]
        frame_area = h * w
        boxes     = results.boxes.xyxy.cpu().numpy()
        confs     = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        names     = results.names

        for box, conf, cls_id in zip(boxes, confs, class_ids):
            # ── Filter 1: skip anything outside our target classes ───────────
            if cls_id >= NUM_TARGET_CLASSES:
                continue

            # ── Filter 2: skip non-defective classes (e.g. negative prompts) ──
            if cls_id not in DEFECTIVE_CLASS_INDICES:
                continue

            x1, y1, x2, y2 = map(int, box)
            bw, bh = x2 - x1, y2 - y1
            
            # ── Filter 3: Spatial Filtering (Area & Aspect Ratio) ───────────
            detection_area = bw * bh
            area_ratio = detection_area / frame_area
            
            # Ignore huge detections (likely the whole road/shadows)
            if area_ratio > MAX_DETECTION_AREA_RATIO:
                continue
                
            # Ignore extremely thin/long boxes (often artifacts or road lines)
            if bw > 0 and bh > 0:
                aspect_ratio = max(bw / bh, bh / bw)
                if aspect_ratio > MAX_ASPECT_RATIO:
                    continue

            # ── Filter 4: per-class confidence threshold ────────────────────
            if cls_id in ROAD_DEFECT_INDICES:
                if conf < POTHOLE_CONF_THRESHOLD:
                    continue
            else:
                if conf < SIGN_CONF_THRESHOLD:
                    continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            colour = get_colour(cls_id)
            label  = names.get(cls_id, f"class_{cls_id}")

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Label background
            label_text = f"{label}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)

            # Label text
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            valid_detections.append((cx, cy, cls_id, conf, box))

    return frame, valid_detections


# ──────────────────────────────────────────────
# Image detection
# ──────────────────────────────────────────────

def detect_image(model: YOLOE, image_path: str, save_path: str | None = None):
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    print(f"\n[INFO] Running detection on image: {image_path}")
    img = cv2.imread(image_path)

    results = model.predict(img, conf=CONFIDENCE_THRESHOLD)
    annotated, valid_dets = draw_detections(img, results[0])
    print(f"[INFO] Defective detections found: {len(valid_dets)}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, annotated)
        print(f"[INFO] Saved annotated image → {save_path}")

    cv2.imshow("YOLOE - Image Detection", annotated)
    print("[INFO] Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Video detection
# ──────────────────────────────────────────────

def detect_video(
    model: YOLOE,
    video_path: str,
    save_path: str | None = None,
    show: bool = True,
):
    source = 0 if video_path == "0" else video_path
    if isinstance(source, str) and not os.path.isfile(source):
        print(f"[ERROR] Video not found: {source}")
        return

    print(f"\n[INFO] Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output video → {save_path}")

    frame_num        = 0
    unique_detections = 0
    spatial_locations = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            current_time = frame_num / fps

            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            annotated, valid_dets = draw_detections(frame, results[0])
            
            # Deduplicate detections to count unique objects
            current_frame_defects = 0
            for cx, cy, cls_id, conf, box in valid_dets:
                if not is_duplicate_location(cx, cy, cls_id, current_time, spatial_locations):
                    spatial_locations.append({
                        "center": (cx, cy),
                        "time": current_time,
                        "class_id": cls_id
                    })
                    unique_detections += 1
                current_frame_defects += 1

            info_text = f"Frame: {frame_num}/{total_frames}  |  Total Unique: {unique_detections}"
            cv2.putText(
                annotated,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("YOLOE - Video Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    print("[INFO] Stopped by user.")
                    break

            if frame_num % 100 == 0:
                print(f"  [PROGRESS] Frame {frame_num}/{total_frames}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Processed {frame_num} frames, total unique defective detections: {unique_detections}")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLOE Open-Vocabulary Detection — Signboards & Road Defects"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=INPUT_SOURCE,
        help=f"Path to image, video file, or '0' for webcam (default: {INPUT_SOURCE})",
    )
    parser.add_argument(
        "--weights",
        default=MODEL_WEIGHTS,
        help=f"YOLOE model weights (default: {MODEL_WEIGHTS})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--save",
        default=OUTPUT_PATH,
        help=f"Path to save annotated output (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the output window",
    )

    args = parser.parse_args()

    model  = load_model(args.weights)
    source = args.source
    ext    = os.path.splitext(source)[1].lower()

    if source == "0" or ext in VIDEO_EXTENSIONS:
        print(f"[INFO] Detected source type: VIDEO")
        detect_video(model, source, save_path=args.save, show=not args.no_show)
    elif ext in IMAGE_EXTENSIONS:
        print(f"[INFO] Detected source type: IMAGE")
        detect_image(model, source, save_path=args.save)
    else:
        print(f"[ERROR] Unsupported file format: '{ext}'")
        print(f"  Supported images : {', '.join(IMAGE_EXTENSIONS)}")
        print(f"  Supported videos : {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(1)


if __name__ == "__main__":
    main()