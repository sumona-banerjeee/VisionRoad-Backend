"""
YOLOE - Open Vocabulary Real-Time Object Detection
Detects: defective signboards, potholes, road damage from images and videos.

Uses Ultralytics YOLOE with text prompts (open-vocabulary detection).
Docs: https://docs.ultralytics.com/models/yoloe/
"""

import cv2
import sys
import os
import argparse
from datetime import datetime
from ultralytics import YOLOE


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# YOLOE model weights (auto-downloaded on first run)
# Options: yoloe-11s-seg.pt, yoloe-11m-seg.pt, yoloe-11l-seg.pt
#          yoloe-v8s-seg.pt, yoloe-v8m-seg.pt, yoloe-v8l-seg.pt
# MODEL_WEIGHTS = "yoloe-11l-seg.pt"
MODEL_WEIGHTS = "yoloe-11m-seg.pt"

# ── Input / Output paths ──
# Set these to run directly: python Test/yoloe.py
# CLI args override these if provided.
INPUT_SOURCE = r"Test\video\live-vid-4.mp4"   # image or video path
OUTPUT_PATH  = r"C:\Users\Administrator\Desktop\AiML\Sumona\VisionRoad-Backend\Test\output\output-4(yoloe-defective-version).mp4"     # save annotated output here

# Open-vocabulary text prompts — what to detect
# Target classes (these are the defects we care about)
TARGET_PROMPTS = [
    "damaged traffic signboard",
    "broken traffic signboard",
    "faded colorless white blank signboard",
    "rusted signboard",
    "pothole",
    "puddle",
    "road crack",
    "damaged road marking",
    "damaged circular round traffic sign",
    "faded circular round traffic sign", 
    "broken circular round traffic sign",
    "advertisement poster", 
    "commercial billboard", 
    "banner", "shop sign", 
    "building nameplate",
    "clean intact traffic sign",
]

# Contrastive classes (helps model differentiate — we ignore these in output)
CONTRASTIVE_PROMPTS = [
    "normal traffic signboard",
    "good signboard",
    "non-defective signboard",
    "advertisement board",
]

# Combined prompts sent to model (target first, then contrastive)
PROMPTS = TARGET_PROMPTS + CONTRASTIVE_PROMPTS

# Only show detections for the first N classes (target classes)
NUM_TARGET_CLASSES = len(TARGET_PROMPTS)

# ── Display filter ──────────────────────────────────────────────────────────
# Indices (within TARGET_PROMPTS) that represent DEFECTIVE / DAMAGED items.
# ALL prompts above are kept intact — they help the model's open-vocabulary
# understanding and contrastive reasoning.
# This set only controls what gets drawn on the output frame.
# Classes NOT listed here (advertisements, good signs, etc.) are detected
# internally but silently suppressed in the visual output.
DEFECTIVE_CLASS_INDICES = {
    TARGET_PROMPTS.index("damaged traffic signboard"),
    TARGET_PROMPTS.index("broken traffic signboard"),
    TARGET_PROMPTS.index("faded colorless white blank signboard"),
    TARGET_PROMPTS.index("rusted signboard"),
    TARGET_PROMPTS.index("pothole"),
    TARGET_PROMPTS.index("puddle"),
    TARGET_PROMPTS.index("road crack"),
    TARGET_PROMPTS.index("damaged road marking"),
    TARGET_PROMPTS.index("damaged circular round traffic sign"),
    TARGET_PROMPTS.index("faded circular round traffic sign"),
    TARGET_PROMPTS.index("broken circular round traffic sign"),
    # "advertisement poster", "commercial billboard", "banner", "shop sign",
    # "building nameplate", "clean intact traffic sign"  ← intentionally excluded
}
# ───────────────────────────────────────────────────────────────────────────

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.43

# Supported file extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# Colours for each target class (BGR)
CLASS_COLOURS = [
    (0, 0, 255),     # Red     — damaged traffic sign
    (0, 128, 255),   # Orange  — broken traffic signboard
    (0, 255, 255),   # Yellow  — faded colorless blank signboard
    (0, 165, 255),   # Gold    — rusted signboard
    (255, 0, 255),   # Magenta — pothole
    (255, 255, 0),   # Cyan    — puddle
    (255, 0, 0),     # Blue    — road crack
    (128, 0, 255),   # Purple  — damaged road marking
    (0, 80, 255),    # Red-Org — damaged circular sign
    (0, 200, 255),   # Yellow  — faded circular sign
    (200, 100, 255), # Pink    — broken circular sign
]


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def load_model(weights: str = MODEL_WEIGHTS) -> YOLOE:
    """Load YOLOE model and set open-vocabulary classes."""
    print(f"[INFO] Loading YOLOE model: {weights}")
    model = YOLOE(weights)

    # Set the text prompts (open-vocabulary classes)
    # This only needs to be called once after loading the model.
    model.set_classes(PROMPTS)
    print(f"[INFO] Classes set: {PROMPTS}")
    return model


def get_colour(class_id: int) -> tuple:
    """Return a BGR colour for a given class id."""
    return CLASS_COLOURS[class_id % len(CLASS_COLOURS)]


def draw_detections(frame, results):
    """
    Draw bounding boxes and labels on the frame.

    Only detections whose class index appears in DEFECTIVE_CLASS_INDICES
    are drawn. All other detections (ads, good signs, contrastive classes)
    are skipped silently — the prompts themselves are NOT removed, as they
    improve the model's open-vocabulary discrimination.

    Parameters
    ----------
    frame  : numpy array (BGR image)
    results: Ultralytics Results object (single element)

    Returns
    -------
    annotated_frame : numpy array with drawn detections
    count           : number of defective detections drawn
    """
    count = 0
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()       # (N, 4) — x1, y1, x2, y2
        confs = results.boxes.conf.cpu().numpy()        # (N,)
        class_ids = results.boxes.cls.cpu().numpy().astype(int)  # (N,)

        # Map class ids to names
        names = results.names  # dict {class_id: class_name}

        for box, conf, cls_id in zip(boxes, confs, class_ids):
            # ── Filter 1: skip contrastive classes ──────────────────────────
            if cls_id >= NUM_TARGET_CLASSES:
                continue

            # ── Filter 2: skip non-defective target classes ─────────────────
            # (advertisements, good/intact signs, etc.)
            if cls_id not in DEFECTIVE_CLASS_INDICES:
                continue

            x1, y1, x2, y2 = map(int, box)
            colour = get_colour(cls_id)
            label = names.get(cls_id, f"class_{cls_id}")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Draw label background
            label_text = f"{label}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)

            # Draw label text
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

    # Run inference
    results = model.predict(img, conf=CONFIDENCE_THRESHOLD)

    # Draw detections (results is a list; take the first element)
    annotated, det_count = draw_detections(img, results[0])
    print(f"[INFO] Defective detections found: {det_count}")

    # Save annotated image
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, annotated)
        print(f"[INFO] Saved annotated image → {save_path}")

    # Display
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
    video_path : path to input video (or 0 for webcam)
    save_path  : optional path to save annotated video (.mp4)
    show       : whether to display frames in a window
    """
    # Open video source
    source = 0 if video_path == "0" else video_path
    if isinstance(source, str) and not os.path.isfile(source):
        print(f"[ERROR] Video not found: {source}")
        return

    print(f"\n[INFO] Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    # Video writer (optional)
    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output video → {save_path}")

    frame_num = 0
    total_detections = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Run inference
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # Draw defective detections only
            annotated, det_count = draw_detections(frame, results[0])
            total_detections += det_count

            # Add frame info overlay
            info_text = f"Frame: {frame_num}/{total_frames}  |  Defects: {det_count}"
            cv2.putText(
                annotated,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Write to output video
            if writer:
                writer.write(annotated)

            # Display
            if show:
                cv2.imshow("YOLOE - Video Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):  # ESC or Q to quit
                    print("[INFO] Stopped by user.")
                    break

            # Progress log every 100 frames
            if frame_num % 100 == 0:
                print(f"  [PROGRESS] Frame {frame_num}/{total_frames}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Processed {frame_num} frames, total defective detections: {total_detections}")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLOE Open-Vocabulary Detection for Road Defects"
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

    # Load model
    model = load_model(args.weights)

    # Auto-detect source type based on file extension
    source = args.source
    ext = os.path.splitext(source)[1].lower()

    if source == "0" or ext in VIDEO_EXTENSIONS:
        print(f"[INFO] Detected source type: VIDEO")
        detect_video(model, source, save_path=args.save, show=not args.no_show)
    elif ext in IMAGE_EXTENSIONS:
        print(f"[INFO] Detected source type: IMAGE")
        detect_image(model, source, save_path=args.save)
    else:
        print(f"[ERROR] Unsupported file format: '{ext}'")
        print(f"  Supported images: {', '.join(IMAGE_EXTENSIONS)}")
        print(f"  Supported videos: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(1)


if __name__ == "__main__":
    main()