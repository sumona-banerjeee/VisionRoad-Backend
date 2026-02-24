import cv2
import json
import os
import time
from datetime import datetime
from pathlib import Path

# ===================== INSTALL CHECK =====================
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    print("ERROR: SAHI is not installed. Run: pip install sahi")
    raise

# ===================== CONFIG =====================
MODEL_PATH = r"models\final-v1.pt"
INPUT_PATH = r"C:\Users\SUMAN\Downloads\8177426-uhd_3840_2160_24fps.mp4"
OUTPUT_DIR = r"Test\output"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ---- SAHI Slicing Parameters ----
SLICE_HEIGHT = 640  # Tile height (match YOLO training resolution)
SLICE_WIDTH = 640  # Tile width
OVERLAP_RATIO = 0.20  # 20% overlap between tiles (catches edge objects)

# ---- Detection Thresholds ----
CONF_THRESHOLD = 0.35  # Lower than dashcam (0.50) — aerial objects score lower
NMS_THRESHOLD = 0.50  # IoU threshold for merging cross-tile duplicate boxes

# ---- Processing ----
FRAME_SKIP = 3  # Process every Nth frame (higher = faster, lower = more thorough)

# ---- Drone-tuned Spatial Dedup ----
# Objects appear smaller from altitude → smaller pixel distance threshold
MIN_DISTANCE_THRESHOLD = 60  # px (halved vs dashcam 120px)
TIME_WINDOW_PERCENT = 0.30  # 30% of video duration for dedup window

# ===================== CLASS DEFINITIONS =====================
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}
EXCLUDED_CLASSES = {"good_sign_board"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES

# ===================== INPUT TYPE =====================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

input_ext = os.path.splitext(INPUT_PATH)[1].lower()
IS_IMAGE = input_ext in IMAGE_EXTENSIONS
IS_VIDEO = input_ext in VIDEO_EXTENSIONS

if not IS_IMAGE and not IS_VIDEO:
    raise ValueError(f"Unsupported file type: {input_ext}")

input_basename = os.path.splitext(os.path.basename(INPUT_PATH))[0]
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, f"{input_basename}-drone-sahi.{'jpg' if IS_IMAGE else 'mp4'}"
)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, f"{input_basename}-drone-sahi.json")

# ===================== LOAD MODEL =====================
print("=" * 60)
print("DRONE ROAD DAMAGE DETECTION — SAHI")
print("=" * 60)
print(f"Model      : {MODEL_PATH}")
print(f"Input      : {INPUT_PATH} ({'IMAGE' if IS_IMAGE else 'VIDEO'})")
print(f"Slice size : {SLICE_WIDTH}x{SLICE_HEIGHT} | Overlap: {OVERLAP_RATIO*100:.0f}%")
print(f"Conf       : {CONF_THRESHOLD} | NMS: {NMS_THRESHOLD}")
print(f"Frame skip : {FRAME_SKIP}")
print(f"Spatial dedup threshold: {MIN_DISTANCE_THRESHOLD}px")
print("=" * 60)
print("Loading YOLO model via SAHI...")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRESHOLD,
    device="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
)
print("Model loaded.\n")

# ===================== OPEN INPUT =====================
if IS_IMAGE:
    frame = cv2.imread(INPUT_PATH)
    if frame is None:
        raise ValueError(f"Could not read image: {INPUT_PATH}")
    height, width = frame.shape[:2]
    fps = 1.0
    total_frames = 1
    video_duration = 1.0
else:
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {INPUT_PATH}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

TIME_THRESHOLD = video_duration * TIME_WINDOW_PERCENT

print(f"Resolution  : {width}x{height}")
if IS_VIDEO:
    print(
        f"Duration    : {video_duration:.1f}s ({total_frames} frames @ {fps:.1f} FPS)"
    )
    print(f"Frames to process: ~{total_frames // FRAME_SKIP}")
    print(f"Dedup window: {TIME_THRESHOLD:.1f}s")
print()

if IS_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_PATH, fourcc, fps / FRAME_SKIP, (width, height)
    )


# ===================== HELPERS =====================
def calc_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def is_spatial_duplicate(cx, cy, class_name, current_time, spatial_locations):
    """Check if this detection is spatially close to a recent one of the same class."""
    for loc in spatial_locations:
        if loc["class"] != class_name:
            continue
        dist = calc_distance((cx, cy), loc["center"])
        time_gap = current_time - loc["time"]
        if dist < MIN_DISTANCE_THRESHOLD and time_gap < TIME_THRESHOLD:
            return True, f"{dist:.1f}px away, {time_gap:.2f}s ago"
    return False, None


def run_sahi_on_frame(frame_bgr):
    """
    Run SAHI sliced inference on a BGR frame.
    Returns list of (class_name, conf, x1, y1, x2, y2).
    """
    # SAHI expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    result = get_sliced_prediction(
        image=frame_rgb,
        detection_model=detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        postprocess_type="NMM",  # Non-Maximum Merging across slices
        postprocess_match_threshold=NMS_THRESHOLD,
        verbose=0,
    )

    detections = []
    for obj in result.object_prediction_list:
        class_name = obj.category.name
        if class_name not in ALL_CLASSES:
            continue
        conf = float(obj.score.value)
        bbox = obj.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        detections.append((class_name, conf, x1, y1, x2, y2))
    return detections


def draw_detections(frame, detections, confirmed_ids):
    """Draw bounding boxes on frame."""
    CLASS_COLORS = {
        "pothole": (0, 0, 255),
        "road_crack": (0, 165, 255),
        "defected_sign_board": (0, 255, 255),
        "damaged_road_marking": (255, 0, 255),
        "good_sign_board": (0, 255, 0),
    }
    for class_name, conf, x1, y1, x2, y2 in detections:
        color = CLASS_COLORS.get(class_name, (200, 200, 200))
        is_confirmed = (class_name, (x1, y1, x2, y2)) in confirmed_ids
        thickness = 2 if is_confirmed else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    return frame


# ===================== TRACKING STRUCTURES =====================
confirmed = []  # List of confirmed detection dicts
spatial_locations = []  # For deduplication across frames
counted_ids = {cls: 0 for cls in ALL_CLASSES}

rejection_stats = {
    "spatial_duplicate": 0,
    "below_conf": 0,
    "skipped_frames": 0,
}

output_json = {
    "input_path": INPUT_PATH,
    "input_type": "image" if IS_IMAGE else "video",
    "processed_at": datetime.utcnow().isoformat(),
    "mode": "SAHI_DRONE",
    "sahi_config": {
        "slice_height": SLICE_HEIGHT,
        "slice_width": SLICE_WIDTH,
        "overlap_ratio": OVERLAP_RATIO,
        "conf_threshold": CONF_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
    },
    "input_info": {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration": round(video_duration, 2),
    },
    "detections": [],
    "summary": {},
    "rejection_stats": {},
}

# ===================== MAIN LOOP =====================
print("Processing frames...")
print("-" * 60)

frame_count = 0
processed_count = 0
start_time = time.time()

while True:
    # -- Read frame --
    if IS_IMAGE:
        if frame_count >= 1:
            break
        current_frame = frame
        ret = True
    else:
        ret, current_frame = cap.read()
        if not ret:
            break

    frame_count += 1

    # -- Frame skipping --
    if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
        rejection_stats["skipped_frames"] += 1
        continue

    current_time = frame_count / fps
    processed_count += 1

    # -- SAHI inference --
    sahi_detections = run_sahi_on_frame(current_frame)
    annotated_frame = current_frame.copy()

    frame_new_detections = []

    for class_name, conf, x1, y1, x2, y2 in sahi_detections:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        is_dup, reason = is_spatial_duplicate(
            cx, cy, class_name, current_time, spatial_locations
        )

        if is_dup:
            rejection_stats["spatial_duplicate"] += 1
            print(
                f"  ❌ Frame {frame_count:4d} ({current_time:5.2f}s): "
                f"DUPLICATE {class_name} at ({cx},{cy}) — {reason}"
            )
            continue

        # New unique detection
        det = {
            "detection_id": len(confirmed) + 1,
            "type": class_name,
            "is_road_damage": class_name in ROAD_DAMAGE_CLASSES,
            "first_detected_frame": frame_count,
            "first_detected_time": round(current_time, 2),
            "confidence": round(conf, 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center": {"x": cx, "y": cy},
        }
        confirmed.append(det)
        spatial_locations.append(
            {
                "center": (cx, cy),
                "time": current_time,
                "class": class_name,
            }
        )
        counted_ids[class_name] += 1
        frame_new_detections.append((class_name, conf, x1, y1, x2, y2))

        damage_tag = "🚨 DAMAGE" if class_name in ROAD_DAMAGE_CLASSES else "ℹ️  INFO"
        print(
            f"  ✅ Frame {frame_count:4d} ({current_time:5.2f}s): "
            f"NEW {class_name} conf={conf:.2f} at ({cx},{cy}) {damage_tag}"
        )

    # -- Annotate & write frame --
    confirmed_set = {
        (
            d["type"],
            (d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]),
        )
        for d in confirmed
    }
    annotated_frame = draw_detections(annotated_frame, sahi_detections, confirmed_set)

    road_damage_total = sum(counted_ids[c] for c in ROAD_DAMAGE_CLASSES)
    cv2.putText(
        annotated_frame,
        f"Road Damage: {road_damage_total}  |  Frame: {frame_count}/{total_frames}  |  SAHI DRONE",
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if IS_IMAGE:
        cv2.imwrite(OUTPUT_PATH, annotated_frame)
    else:
        video_writer.write(annotated_frame)

    cv2.imshow("Drone SAHI Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n⚠️  Interrupted by user.")
        break

# ===================== FINALIZE =====================
elapsed = time.time() - start_time

if IS_VIDEO:
    cap.release()
    video_writer.release()
cv2.destroyAllWindows()

# ===================== BUILD JSON OUTPUT =====================
road_damage_total = sum(counted_ids[c] for c in ROAD_DAMAGE_CLASSES)

output_json["detections"] = confirmed
output_json["summary"] = {
    "frames_processed": processed_count,
    "total_unique_detections": len(confirmed),
    "total_road_damage": road_damage_total,
    "unique_pothole": counted_ids["pothole"],
    "unique_road_crack": counted_ids["road_crack"],
    "unique_defected_sign_board": counted_ids["defected_sign_board"],
    "unique_damaged_road_marking": counted_ids["damaged_road_marking"],
    "unique_good_sign_board": counted_ids["good_sign_board"],
    "processing_time_s": round(elapsed, 2),
    "fps_processed": round(processed_count / elapsed, 1) if elapsed > 0 else 0,
}
output_json["rejection_stats"] = {
    "spatial_duplicates": rejection_stats["spatial_duplicate"],
    "frames_skipped": rejection_stats["skipped_frames"],
}

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(output_json, f, indent=2)

# ===================== FINAL REPORT =====================
print("\n" + "=" * 60)
print("DRONE SAHI DETECTION — COMPLETE")
print("=" * 60)
print(f"Frames processed   : {processed_count} / {total_frames}")
print(
    f"Processing time    : {elapsed:.1f}s  ({output_json['summary']['fps_processed']} FPS)"
)
print()
print("── ROAD DAMAGE (Counted) ──────────────────────────────")
print(f"  Potholes                : {counted_ids['pothole']}")
print(f"  Road Cracks             : {counted_ids['road_crack']}")
print(f"  Defected Sign Boards    : {counted_ids['defected_sign_board']}")
print(f"  Damaged Road Markings   : {counted_ids['damaged_road_marking']}")
print(f"  ─────────────────────────────────")
print(f"  TOTAL ROAD DAMAGE       : {road_damage_total}")
print()
print("── OTHER (Not Counted as Damage) ──────────────────────")
print(f"  Good Sign Boards        : {counted_ids['good_sign_board']}")
print()
print("── REJECTION STATS ────────────────────────────────────")
print(f"  Spatial duplicates      : {rejection_stats['spatial_duplicate']}")
print(f"  Frames skipped          : {rejection_stats['skipped_frames']}")
print()
print(f"Output video : {OUTPUT_PATH}")
print(f"JSON report  : {OUTPUT_JSON_PATH}")
print("=" * 60)

if len(confirmed) == 0:
    print()
    print("⚠️  No detections found.")
    print("   Tips to improve:")
    print("   1. Lower CONF_THRESHOLD (currently 0.35) — try 0.25")
    print("   2. Increase OVERLAP_RATIO (currently 0.20) — try 0.30")
    print("   3. Decrease SLICE_HEIGHT/WIDTH for higher-altitude footage")
    print("   4. The model was trained on dashcam footage —")
    print("      retraining on aerial data will significantly improve results.")
