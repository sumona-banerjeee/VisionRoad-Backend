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

# YOLOE model weights (auto-downloaded on first run)
# Options: yoloe-11s-seg.pt, yoloe-11m-seg.pt, yoloe-11l-seg.pt
#          yoloe-v8s-seg.pt, yoloe-v8m-seg.pt, yoloe-v8l-seg.pt
MODEL_WEIGHTS = "yoloe-11m-seg.pt"

# ── Input / Output paths ──
INPUT_SOURCE = r"Test\video\live-vid-2.mp4"
OUTPUT_PATH  = r"C:\Users\Administrator\Desktop\AiML\Sumona\VisionRoad-Backend\Test\output\output-5(yoloe-v1).mp4"

# ── Open-vocabulary text prompts ──
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
    "roadside advertisement billboard",
    "shop signboard near road",
    "commercial banner on roadside",

    # ── Road Defects ──
    "pothole on asphalt road",
    "deep road pothole",
    "disintegrating road surface patch",           # NEW — covers Image 4
    "loose gravel patch on road",                  # NEW — covers Image 4
    "water puddle on asphalt road",
    "standing water on road surface",
    "longitudinal crack on asphalt",
    "road surface crack",
    "faded road lane marking",
    "worn lane line on road",
]

# No contrastive prompts — removed entirely
PROMPTS = TARGET_PROMPTS
NUM_TARGET_CLASSES = len(TARGET_PROMPTS)

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.65

# Supported file extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# ── Colours per class (BGR) — names reflect rendered screen colour ──
CLASS_COLOURS = [
    # ── Good Signboards ──
    (0, 255, 0),       # Lime Green       — clean intact traffic sign
    (0, 220, 0),       # Medium Green     — clear visible road sign
    (0, 180, 0),       # Dark Green       — good condition traffic signboard

    # ── Damaged Signboards ──
    (0, 0, 255),       # Red              — damaged traffic signboard
    (0, 80, 255),      # Orange-Red       — broken traffic signboard
    (0, 200, 255),     # Yellow           — faded colorless blank signboard
    (0, 140, 255),     # Orange           — rusted signboard
    (0, 60, 200),      # Dark Orange-Red  — bent traffic signboard
    (60, 0, 200),      # Dark Red-Purple  — graffiti covered signboard
    (0, 30, 180),      # Dark Red         — cracked traffic sign

    # ── Circular Signs ──
    (180, 0, 180),     # Magenta          — damaged circular round sign
    (200, 0, 140),     # Deep Pink        — faded circular round sign
    (220, 0, 100),     # Crimson Pink     — broken circular round sign
    (100, 200, 255),   # Light Yellow     — clean circular round sign

    # ── Commercial Boards ──
    (0, 215, 255),     # Gold             — advertisement billboard
    (0, 200, 220),     # Dark Yellow      — shop sign board
    (0, 180, 200),     # Olive Yellow     — building nameplate
    (0, 160, 180),     # Dark Olive       — commercial banner

    # ── Road Defects ──
    (255, 0, 255),     # Magenta-Pink     — pothole
    (255, 200, 0),     # Sky Blue         — puddle on road
    (255, 255, 0),     # Cyan             — road crack
    (255, 100, 0),     # Azure Blue       — damaged road marking
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


def draw_detections(frame, results):
    """
    Draw bounding boxes and labels on the frame.

    Parameters
    ----------
    frame  : numpy array (BGR image)
    results: Ultralytics Results object (single element)

    Returns
    -------
    annotated_frame : numpy array with drawn detections
    count           : number of detections
    """
    count = 0
    if results.boxes is not None and len(results.boxes) > 0:
        boxes     = results.boxes.xyxy.cpu().numpy()
        confs     = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        names     = results.names  # dict {class_id: class_name}

        for box, conf, cls_id in zip(boxes, confs, class_ids):
            # Skip anything outside our target classes
            if cls_id >= NUM_TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
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
            count += 1

    return frame, count


# ──────────────────────────────────────────────
# Image detection
# ──────────────────────────────────────────────

def detect_image(model: YOLOE, image_path: str, save_path: str | None = None):
    """
    Run YOLOE detection on a single image.

    Parameters
    ----------
    model      : loaded YOLOE model
    image_path : path to input image
    save_path  : optional path to save the annotated image
    """
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    print(f"\n[INFO] Running detection on image: {image_path}")
    img = cv2.imread(image_path)

    results = model.predict(img, conf=CONFIDENCE_THRESHOLD)
    annotated, det_count = draw_detections(img, results[0])
    print(f"[INFO] Detections found: {det_count}")

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
    """
    Run YOLOE detection on a video file frame-by-frame.

    Parameters
    ----------
    model      : loaded YOLOE model
    video_path : path to input video (or '0' for webcam)
    save_path  : optional path to save annotated video (.mp4)
    show       : whether to display frames in a window
    """
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
    total_detections = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            annotated, det_count = draw_detections(frame, results[0])
            total_detections += det_count

            # Frame info overlay
            info_text = f"Frame: {frame_num}/{total_frames}  |  Detections: {det_count}"
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

    print(f"\n[DONE] Processed {frame_num} frames, total detections: {total_detections}")


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

    # Use CLI confidence if provided
    conf_threshold = args.conf

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