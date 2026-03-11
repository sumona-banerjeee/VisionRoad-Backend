"""
YOLOE - Open Vocabulary Real-Time Object Detection
Detects: signboards (good/damaged), potholes (various sizes), road cracks, and other road defects.

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
MODEL_WEIGHTS = "yoloe-11m-seg.pt"

# # ── Input / Output paths video──
# INPUT_SOURCE = r"C:\Users\Administrator\Desktop\AiML\Sumona\VisionRoad-Backend\Test\video\live-vid-2.mp4"
# OUTPUT_PATH  = r"C:\Users\Administrator\Desktop\AiML\Sumona\VisionRoad-Backend\Test\output\sign_pothole_results.mp4"

# ── Input / Output paths image──
INPUT_SOURCE = r"Test\img\img1.png"
OUTPUT_PATH  = r"Test\output\img\output"

# ── Open-vocabulary text prompts ──
TARGET_PROMPTS = [
    # # ── Intact / Good Signboards ──
    # "clean traffic sign on pole",
    # "intact road sign on metal post",
    # "visible traffic signboard on road",
    # "good condition municipal road sign",

    # # ── Triangular Warning Signs ──
    # "triangular warning traffic sign",
    # "faded triangular road sign red border",
    # "damaged triangle traffic sign on pole",
    # "weathered triangular signboard",

    # # ── Circular / Round Signs ──
    # "circular traffic sign on road",
    # "blank white circular traffic sign",
    # "faded erased circular road sign",
    # "shattered broken circular road sign",
    # "damaged convex road mirror sign",
    # "round prohibitory traffic sign",
    # "circular no parking sign",

    # # ── Rectangular / Informational Signs ──
    # "rectangular road information sign",
    # "faded bus stop signboard",
    # "faded rectangular traffic sign pole",
    # "blank white rectangular road sign",

    # # ── Damaged / Defective Signboards (General) ──
    # "damaged traffic signboard on road",
    # "broken traffic signboard on pole",
    # "faded traffic signboard on road",
    # "blank white faded signboard",
    # "rusted metal traffic sign",
    # "bent traffic sign pole",
    # "graffiti covered traffic sign",
    # "cracked traffic signboard",
    # "weathered signboard with peeling paint",

    # # ── Commercial / Non-traffic Signboards ──
    # "roadside advertisement billboard",
    # "shop signboard near road",
    # "commercial banner on roadside",

    # ── Potholes (Detailed) ──
    "tiny circular pothole on road",
    "small shallow road pothole",
    "large deep crater-like pothole",
    "cluster of multiple small potholes",
    "rough potholed road surface",
    "developing pothole with loose asphalt",
    "deep road pothole",

    # ── Surface Raveling & Asphalt Patches (NEW - based on images) ──
    "shallow patch of missing asphalt",
    "exposed aggregate road surface",
    "surface raveling on asphalt",
    "broad irregular worn road patch",
    "disintegrated asphalt surface area",
    "exposed light colored road sublayer",

    # ── Other Road Defects & Issues ──
    "alligator cracking pattern on asphalt",
    "longitudinal road surface crack",
    "transverse crack across road",
    "damaged road divider or median",
    "missing circular manhole cover",
    "uneven road surface bump",
    "road shoulder erosion or drop-off",
    "construction debris on road lane",
    "loose gravel spill on road",
    "water puddle on asphalt road",
    "standing water on road surface",
    "faded road lane marking",
    "worn lane line on road",
]

PROMPTS = TARGET_PROMPTS
NUM_TARGET_CLASSES = len(TARGET_PROMPTS)

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.65

# Supported file extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# ── Colours per class (BGR) ──
# We use a diverse palette. The modulo operator in get_colour handles overflow.
CLASS_COLOURS = [
    # Good Signboards (Greens)
    (0, 255, 0), (0, 220, 0), (0, 180, 0), (0, 150, 0),
    # Damaged Signboards (Reds/Oranges)
    (0, 0, 255), (0, 80, 255), (0, 200, 255), (0, 140, 255), (0, 60, 200), (60, 0, 200), (0, 30, 180),
    # Circular Signs (Purples/Pinks)
    (180, 0, 180), (200, 0, 140), (220, 0, 100), (100, 200, 255),
    # Commercial (Yellows/Golds)
    (0, 215, 255), (0, 200, 220), (0, 180, 200), (0, 160, 180),
    # Potholes (Magenta/Bright Pink/Blue)
    (255, 0, 255), (255, 100, 255), (255, 0, 150), (200, 50, 200), (150, 0, 150),
    # Road Defects (Cyans/Blues)
    (255, 200, 0), (255, 255, 0), (255, 100, 0), (255, 150, 0), (200, 200, 0),
    # Marking/Misc
    (128, 128, 255), (128, 255, 128), (255, 128, 128)
]


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def load_model(weights: str = MODEL_WEIGHTS) -> YOLOE:
    """Load YOLOE model and set open-vocabulary classes."""
    print(f"[INFO] Loading YOLOE model: {weights}")
    model = YOLOE(weights)
    model.set_classes(PROMPTS)
    print(f"[INFO] Classes set: {len(PROMPTS)} prompts active.")
    return model


def get_colour(class_id: int) -> tuple:
    """Return a BGR colour for a given class id."""
    return CLASS_COLOURS[class_id % len(CLASS_COLOURS)]


def draw_detections(frame, results):
    """Draw bounding boxes and labels on the frame."""
    count = 0
    if results.boxes is not None and len(results.boxes) > 0:
        boxes     = results.boxes.xyxy.cpu().numpy()
        confs     = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        names     = results.names

        for box, conf, cls_id in zip(boxes, confs, class_ids):
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
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)

            # Label text
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            count += 1

    return frame, count


# ──────────────────────────────────────────────
# Image detection
# ──────────────────────────────────────────────

def detect_image(model: YOLOE, image_path: str, save_path: str | None = None):
    """Run YOLOE detection on a single image."""
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
        print(f"[INFO] Saved annotated image -> {save_path}")

    cv2.imshow("YOLOE Detection", annotated)
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
    """Run YOLOE detection on a video file frame-by-frame."""
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
        print(f"[INFO] Saving output video -> {save_path}")

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

            # Info overlay
            cv2.putText(
                annotated,
                f"Frame: {frame_num} | Detections: {det_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("YOLOE Road Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_num % 50 == 0:
                print(f"  [PROGRESS] Frame {frame_num}/{total_frames}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Processed {frame_num} frames, total detections: {total_detections}")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOE Expanded Road & Sign Detection")
    parser.add_argument("source", nargs="?", default=INPUT_SOURCE, help="Input source")
    parser.add_argument("--weights", default=MODEL_WEIGHTS, help="Model weights")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Threshold")
    parser.add_argument("--save", default=OUTPUT_PATH, help="Output path")
    parser.add_argument("--no-show", action="store_true", help="Hide window")

    args = parser.parse_args()
    model = load_model(args.weights)
    source = args.source
    ext = os.path.splitext(source)[1].lower()

    if source == "0" or ext in VIDEO_EXTENSIONS:
        detect_video(model, source, save_path=args.save, show=not args.no_show)
    elif ext in IMAGE_EXTENSIONS:
        detect_image(model, source, save_path=args.save)
    else:
        print(f"[ERROR] Unsupported format: {ext}")


if __name__ == "__main__":
    main()
