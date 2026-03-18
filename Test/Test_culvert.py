import os
import sys
import time
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# ██  CONFIGURATION — Edit these values before running  ██
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_SOURCE   = r"C:\Sentientgeeks_Project\VisionRoad-Backend\Test\video\culvert_video4.mp4"  # ← Put your video path here
MODEL_PATH     = r"C:\Sentientgeeks_Project\VisionRoad-Backend\models\culvert_best.pt"            # ← Path to your trained model
CONF_THRESHOLD = 0.60                            # ← Minimum confidence (0.0–1.0)  [>60%]
SAVE_OUTPUT    = True                            # ← Save annotated video?
SHOW_DISPLAY   = True                            # ← Show live preview window?
OUTPUT_DIR     = "output"                        # ← Folder to save results

# ══════════════════════════════════════════════════════════════════════════════

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: 'ultralytics' package not found.")
    print("Install it with:  pip install ultralytics opencv-python")
    sys.exit(1)


# ── Colour palette for classes ──────────────────────────────────────────────
# Extend this list if you have more classes
CLASS_COLORS = {
    0: (0, 255, 0),    # Class 0 → Green  (e.g. Good Culvert)
    1: (0, 0, 255),    # Class 1 → Red    (e.g. Defective Culvert)
}
DEFAULT_COLOR = (255, 165, 0)  # Orange fallback for unknown classes


def get_color(class_id: int) -> tuple:
    """Return a BGR colour for a given class id."""
    return CLASS_COLORS.get(class_id, DEFAULT_COLOR)


def draw_detections(frame, results):
    """
    Draw bounding boxes, labels, confidence scores and track IDs on the frame.
    Returns the annotated frame, the count of detections in this frame,
    and a set of track IDs seen in this frame.
    """
    detections = 0
    frame_track_ids = set()
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            # Bounding‑box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, f"class_{cls_id}")

            # Track ID (BotSORT assigns an integer ID to each tracked object)
            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is not None:
                frame_track_ids.add(track_id)

            color = get_color(cls_id)

            # Build label with track ID
            if track_id is not None:
                label = f"{cls_name} #{track_id} {conf:.0%}"
            else:
                label = f"{cls_name} {conf:.0%}"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), color, -1)

            # Label text (white on coloured background)
            cv2.putText(frame, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            detections += 1

    return frame, detections, frame_track_ids


def process_video(source: str, model_path: str, conf_threshold: float,
                  save_output: bool, show_display: bool, output_dir: str):
    """Run detection on every frame of the video."""

    # ── Load model ──────────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found at '{model_path}'")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"Model loaded. Classes: {model.names}\n")

    # ── Open video ──────────────────────────────────────────────────────────
    if not os.path.isfile(source):
        print(f"ERROR: Video file not found at '{source}'")
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video '{source}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video : {source}")
    print(f"Size  : {width}x{height} @ {fps:.1f} FPS  |  {total_frames} frames")
    print(f"Conf  : {conf_threshold}")
    print("-" * 60)

    # ── Prepare output writer ───────────────────────────────────────────────
    writer = None
    output_path = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(source).stem
        output_path = os.path.join(output_dir, f"{stem}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    # ── Frame‑by‑frame detection ────────────────────────────────────────────
    frame_num = 0
    all_track_ids = set()          # Collect every unique track ID across video
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Run inference with BotSORT tracking
            results = model.track(
                source=frame,
                conf=conf_threshold,
                tracker="botsort.yaml",
                persist=True,
                verbose=False,
            )

            # Draw bounding boxes
            annotated, det_count, frame_ids = draw_detections(frame, results)
            all_track_ids.update(frame_ids)

            # Draw unique-count overlay (top-left corner)
            unique_count = len(all_track_ids)
            counter_text = f"Total Unique Detections: {unique_count}"
            (cw, ch), cb = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(annotated, (10, 10), (20 + cw, 20 + ch + cb), (0, 0, 0), -1)
            cv2.putText(annotated, counter_text, (15, 15 + ch),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

            # Progress info
            elapsed = time.time() - start_time
            proc_fps = frame_num / elapsed if elapsed > 0 else 0
            progress = (frame_num / total_frames * 100) if total_frames > 0 else 0
            print(f"\rFrame {frame_num}/{total_frames}  "
                  f"({progress:5.1f}%)  |  "
                  f"Detections: {det_count}  |  "
                  f"Unique: {unique_count}  |  "
                  f"Speed: {proc_fps:.1f} FPS", end="")

            # Write to output video
            if writer is not None:
                writer.write(annotated)

            # Show live preview
            if show_display:
                cv2.imshow("Culvert Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n\n[INFO] Stopped by user (pressed 'q').")
                    break

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user.")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done!  Processed {frame_num} frames in {elapsed:.1f}s  "
          f"({frame_num / elapsed:.1f} FPS)")
    print(f"Total Unique Detections: {len(all_track_ids)}")
    if output_path:
        print(f"Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    process_video(
        source=VIDEO_SOURCE,
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        save_output=SAVE_OUTPUT,
        show_display=SHOW_DISPLAY,
        output_dir=OUTPUT_DIR,
    )
