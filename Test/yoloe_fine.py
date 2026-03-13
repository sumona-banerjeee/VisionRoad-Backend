"""
YOLOE Fine-tuned Model Inference Script
Detects: defected_sign_board, good_sign_board, pothole, road_crack, damaged_road_marking
Supports: Video file, webcam, and image input
"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#  ✏️  EDIT THESE THREE PATHS BEFORE RUNNING
# ══════════════════════════════════════════════════════
WEIGHTS_PATH = r"models\yoloe_best.pt"
SOURCE       = r"Test\video\live-vid-3.mp4"
OUTPUT_PATH  = r"Test\output\vid3_yoloe_best.mp4"  # set to None to skip saving
# ══════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CATEGORIES = [
    "defected_sign_board",
    "good_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
]

COLOR_MAP = {
    "defected_sign_board":   (0,   0,   255),   # Red
    "good_sign_board":       (0,   255,   0),   # Green
    "pothole":               (0,   165, 255),   # Orange
    "road_crack":            (255,   0,   0),   # Blue
    "damaged_road_marking":  (128,   0, 128),   # Purple
}

CONF_THRESHOLD = 0.70   # ← Only detections above 60% will show
IOU_THRESHOLD  = 0.45


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    print(f"[INFO] Loading model from: {weights_path}")
    model = YOLO(weights_path)
    # NOTE: Do NOT set model.conf or model.iou here.
    # They are passed directly into model() calls below — that is the reliable way.
    print(f"[INFO] Model loaded. Classes: {model.names}")
    print(f"[INFO] Conf threshold: {CONF_THRESHOLD}  |  IoU threshold: {IOU_THRESHOLD}")
    return model


# ─────────────────────────────────────────────
# DRAW DETECTIONS
# ─────────────────────────────────────────────
def draw_detections(frame: np.ndarray, results, model_names: dict) -> np.ndarray:
    if results[0].boxes is None:
        return frame
    for box in results[0].boxes:
        cls_id     = int(box.cls[0])
        conf       = float(box.conf[0])
        class_name = model_names.get(cls_id, f"class_{cls_id}")
        if class_name not in CATEGORIES:
            continue
        # Extra safety guard — double-check conf even after model filtering
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLOR_MAP.get(class_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_legend(frame: np.ndarray) -> np.ndarray:
    h, w    = frame.shape[:2]
    leg_x   = w - 280
    y_start = 20
    for i, cat in enumerate(CATEGORIES):
        color = COLOR_MAP[cat]
        y = y_start + i * 28
        cv2.rectangle(frame, (leg_x, y), (leg_x + 20, y + 18), color, -1)
        cv2.putText(frame, cat.replace("_", " "),
                    (leg_x + 26, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
# VIDEO / WEBCAM INFERENCE
# ─────────────────────────────────────────────
def run_video(model, source, output_path=None, show=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Source: {source}  |  {width}x{height}  FPS={fps:.1f}  Frames={total}")

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {output_path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ✅ conf and iou passed HERE — this is the correct way
            results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

            frame = draw_detections(frame, results, model.names)
            frame = draw_legend(frame)
            cv2.putText(frame, f"Frame: {frame_idx}/{total if total > 0 else '?'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2, cv2.LINE_AA)
            if writer:
                writer.write(frame)
            if show:
                cv2.imshow("YOLOE Road Defect Detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("[INFO] Stopped by user.")
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_idx} frames.")


# ─────────────────────────────────────────────
# IMAGE INFERENCE
# ─────────────────────────────────────────────
def run_image(model, image_path: str, output_path=None, show=True):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # ✅ conf and iou passed HERE — this is the correct way
    results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    frame = draw_detections(frame, results, model.names)
    frame = draw_legend(frame)

    print(f"\n[DETECTIONS] in {image_path}:")
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = model.names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  • {class_name:<28} conf={conf:.3f}  bbox=({x1},{y1},{x2},{y2})")
    else:
        print("  (no detections above conf threshold)")

    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"[INFO] Saved annotated image to: {output_path}")
    if show:
        cv2.imshow("YOLOE Detection", frame)
        print("[INFO] Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    model  = load_model(WEIGHTS_PATH)
    show   = True
    source = SOURCE

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
        print("[INFO] Mode: Webcam inference")
        run_video(model, int(source), OUTPUT_PATH, show=show)
        return

    suffix = Path(str(source)).suffix.lower()

    if suffix in image_exts:
        print("[INFO] Mode: Image inference")
        run_image(model, source, OUTPUT_PATH, show=show)

    elif suffix in video_exts:
        print("[INFO] Mode: Video inference")
        run_video(model, source, OUTPUT_PATH, show=show)

    else:
        print(f"[WARNING] Unknown source type '{source}'. Trying as video...")
        run_video(model, source, OUTPUT_PATH, show=show)


if __name__ == "__main__":
    main()