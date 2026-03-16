import cv2
import sys
import os
import argparse
from ultralytics import YOLOE


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

MODEL_WEIGHTS = r"models\yoloe-11m-seg.pt"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

INPUT_SOURCE = r"Test\drain_data\img\img4.png"

def _derive_output_path(src: str) -> str:
    """Auto-build output path from the input filename.
    - Images  → same stem + original extension  (e.g. img4_output.png)
    - Videos  → same stem + .mp4                (writer uses mp4v codec)
    - Webcam  → 'webcam_output.mp4'
    """
    if src == "0":
        return r"Test\drain_data\output\webcam_output.mp4"
    stem, ext = os.path.splitext(os.path.basename(src))
    out_ext = ext.lower() if ext.lower() in IMAGE_EXTENSIONS else ".mp4"
    return os.path.join(r"Test\drain_data\output", f"{stem}_output{out_ext}")

OUTPUT_PATH = _derive_output_path(INPUT_SOURCE)

# ── Prompts — underscores removed ──
TARGET_PROMPTS = [
    "drain opening",
    "broken drain",
    "missing drain cover",
    "clogged drain",
    "overflowing drain",
    "water clogging",
    "water puddle",
    "pothole",
    "cracked pavement",
    "garbage blockage",
    
    # ── Upright pole prompts ──
    "straight vertical traffic sign pole",
    "upright traffic sign on straight pole",
    "traffic sign on vertical metal post",
    
    # ── Bent pole prompts ──
    "tilted traffic sign pole",
    "bent signpost",
    "leaning traffic sign pole",
    "crooked road sign pole",
]

PROMPTS = TARGET_PROMPTS
NUM_TARGET_CLASSES = len(TARGET_PROMPTS)

CONFIDENCE_THRESHOLD = 0.02


# ── One distinct colour per class (BGR) ──
CLASS_COLOURS = [
    (255, 150,   0),   # 0 — drain opening        → Blue
    (0,     0, 255),   # 1 — broken drain          → Red
    (0,   130, 255),   # 2 — missing drain cover   → Orange
    (180,   0, 180),   # 3 — clogged drain         → Magenta
    (255, 255,   0),   # 4 — overflowing drain     → Cyan
    (255, 200, 100),   # 5 — water clogging        → Sky Blue
    (0,   220, 255),   # 6 — water puddle          → Yellow
    (100,   0, 255),   # 7 — pothole               → Pink
    (0,   200, 100),   # 8 — cracked pavement      → Green
    (30,  100, 180),   # 9 — garbage blockage      → Brown
    
    # Upright pole colours (Silvers / Grays)
    (200, 200, 200),   # 10 — straight vertical traffic sign pole    → Light Gray
    (150, 150, 150),   # 11 — upright traffic sign on straight pole  → Gray
    (100, 100, 100),   # 12 — traffic sign on vertical metal post    → Dark Gray
    
    # Bent pole colours (Reds / Oranges / Purples)
    (0,   128, 255),   # 13 — tilted traffic sign pole               → Deep Orange
    (0,     0, 150),   # 14 — bent signpost                          → Dark Red
    (128,   0, 128),   # 15 — leaning traffic sign pole              → Purple
    (0,   150, 150),   # 16 — crooked road sign pole                 → Dark Yellow
]


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def load_model(weights: str = MODEL_WEIGHTS) -> YOLOE:
    print(f"[INFO] Loading YOLOE model: {weights}")
    model = YOLOE(weights)
    model.set_classes(PROMPTS)
    print(f"[INFO] Classes set ({len(PROMPTS)}):")
    for i, p in enumerate(PROMPTS):
        print(f"  [{i:02d}] {p}")
    return model


def get_colour(class_id: int) -> tuple:
    return CLASS_COLOURS[class_id % len(CLASS_COLOURS)]


def draw_detections(frame, results):
    """
    Draw bounding boxes and labels on the frame.

    Returns
    -------
    annotated_frame  : numpy array with drawn detections
    count            : number of detections
    detection_list   : list of (label, confidence) sorted by confidence desc
    """
    count          = 0
    detection_list = []

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
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)

            # Label text
            cv2.putText(
                frame, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
            count += 1
            detection_list.append((label, float(conf)))

    # Sort by confidence descending — highest match at top
    detection_list.sort(key=lambda x: x[1], reverse=True)

    return frame, count, detection_list


def print_detections_terminal(detection_list, frame_num=None):
    """Print detections sorted by confidence descending in the terminal."""
    if frame_num is not None:
        print(f"\n── Frame {frame_num} {'─' * 40}")
    else:
        print(f"\n── Detections {'─' * 40}")

    if not detection_list:
        print("  No detections.")
        return

    print(f"  {'Rank':<6} {'Label':<25} {'Confidence':>10}   {'Bar'}")
    print(f"  {'────':<6} {'─────────────────────────':<25} {'──────────':>10}   {'───────────────────'}")
    for rank, (label, conf) in enumerate(detection_list, start=1):
        bar = "█" * int(conf * 20)
        print(f"  {rank:<6} {label:<25} {conf:>9.2%}   {bar}")


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
    annotated, det_count, detection_list = draw_detections(img, results[0])

    # ── Print sorted detections to terminal ──
    print_detections_terminal(detection_list)
    print(f"\n[INFO] Total detections: {det_count}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, annotated)
        print(f"[INFO] Saved annotated image → {save_path}")

    cv2.imshow("YOLOE - Drain Detection", annotated)
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
    print_interval: int = 30,   # print terminal output every N frames
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
    print(f"[INFO] Terminal output every {print_interval} frames  (change via print_interval)")

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
            annotated, det_count, detection_list = draw_detections(frame, results[0])
            total_detections += det_count

            # ── Print sorted detections every N frames ──
            if detection_list and frame_num % print_interval == 0:
                print_detections_terminal(detection_list, frame_num)

            # Frame info overlay
            info_text = f"Frame: {frame_num}/{total_frames}  |  Detections: {det_count}"
            cv2.putText(
                annotated, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("YOLOE - Drain Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    print("[INFO] Stopped by user.")
                    break

            if frame_num % 100 == 0:
                print(f"  [PROGRESS] Frame {frame_num}/{total_frames}  |  Total detections so far: {total_detections}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Processed {frame_num} frames  |  Total detections: {total_detections}")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLOE Open-Vocabulary Detection — Drain & Road Water Issues"
    )
    parser.add_argument(
        "source", nargs="?", default=INPUT_SOURCE,
        help=f"Path to image, video file, or '0' for webcam (default: {INPUT_SOURCE})",
    )
    parser.add_argument(
        "--weights", default=MODEL_WEIGHTS,
        help=f"YOLOE model weights (default: {MODEL_WEIGHTS})",
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--save", default=None,
        help="Path to save annotated output (auto-derived from source name if omitted)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display the output window",
    )
    parser.add_argument(
        "--print-interval", type=int, default=30,
        help="Print terminal detections every N frames for video (default: 30)",
    )

    args   = parser.parse_args()
    model  = load_model(args.weights)
    source = args.source
    ext    = os.path.splitext(source)[1].lower()

    # Derive save path from the ACTUAL runtime source (not the hardcoded default)
    save_path = args.save if args.save else _derive_output_path(source)
    print(f"[INFO] Output will be saved to: {save_path}")

    if source == "0" or ext in VIDEO_EXTENSIONS:
        print("[INFO] Detected source type: VIDEO")
        detect_video(
            model, source,
            save_path=save_path,
            show=not args.no_show,
            print_interval=args.print_interval,
        )
    elif ext in IMAGE_EXTENSIONS:
        print("[INFO] Detected source type: IMAGE")
        detect_image(model, source, save_path=save_path)
    else:
        print(f"[ERROR] Unsupported file format: '{ext}'")
        print(f"  Supported images : {', '.join(IMAGE_EXTENSIONS)}")
        print(f"  Supported videos : {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(1)


if __name__ == "__main__":
    main()