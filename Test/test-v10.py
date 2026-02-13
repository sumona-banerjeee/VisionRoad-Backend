"""
===============
V10 - 5-CLASS ROAD DAMAGE DETECTION
Detects: defected_sign_board, good_sign_board, pothole, road_crack, damaged_road_marking
Counts only road damage (excludes good_sign_board from damage statistics)
===============
"""

import cv2
import json
import uuid
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
from pathlib import Path

# ===================== CONFIG =====================
MODEL_PATH = r"models\final-v1.pt"
INPUT_PATH = r"Test\video\road-sign-ml.mp4"  # Can be video or image
OUTPUT_DIR = r"Test\output"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.50  # set 0.50 for default
TRACKER = "botsort.yaml"  # Using BoT-SORT for better tracking

# ===================== PREPROCESSING CONFIG =====================
# Enable preprocessing to handle lighting variations and improve detection consistency
ENABLE_CLAHE = True  # Contrast Limited Adaptive Histogram Equalization (recommended)
ENABLE_DENOISE = True  # Reduce noise for cleaner detections
ENABLE_SHARPEN = False  # Enhance edges (use cautiously, may amplify noise)

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0  # Contrast limiting (1.0-4.0, higher = more contrast)
CLAHE_GRID_SIZE = (8, 8)  # Grid size for local equalization

# Denoising parameters
DENOISE_H = 10  # Filter strength (higher = more denoising, but may blur)
DENOISE_TEMPLATE_WINDOW = 7  # Template patch size
DENOISE_SEARCH_WINDOW = 21  # Search area size

# ===================== CLASS DEFINITIONS =====================
# Road damage classes that will be counted
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}

# Classes detected but not counted as damage
EXCLUDED_CLASSES = {"good_sign_board"}

ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES

# ===================== LOAD MODEL =====================
model = YOLO(MODEL_PATH)

# ===================== DETECT INPUT TYPE =====================
import os

input_ext = os.path.splitext(INPUT_PATH)[1].lower()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

IS_IMAGE = input_ext in IMAGE_EXTENSIONS
IS_VIDEO = input_ext in VIDEO_EXTENSIONS

if not IS_IMAGE and not IS_VIDEO:
    raise ValueError(f"Unsupported file type: {input_ext}. Must be image or video.")

# Generate output paths based on input
input_basename = os.path.splitext(os.path.basename(INPUT_PATH))[0]
if IS_IMAGE:
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{input_basename}-v10.jpg")
    OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, f"{input_basename}-v10.json")
else:
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{input_basename}-v10.mp4")
    OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, f"{input_basename}-v10.json")

print("=" * 60)
print(f"INPUT TYPE: {'IMAGE' if IS_IMAGE else 'VIDEO'}")
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")
print("=" * 60)
print()

# ===================== LOAD INPUT =====================
if IS_IMAGE:
    # For images, read directly
    frame = cv2.imread(INPUT_PATH)
    if frame is None:
        raise ValueError(f"Could not read image: {INPUT_PATH}")

    height, width = frame.shape[:2]
    fps = 1.0  # Dummy FPS for images
    total_frames = 1
    video_duration = 1.0  # Treat as 1 second

else:
    # For videos, use VideoCapture
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {INPUT_PATH}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps  # Total video duration in seconds

# ===================== ADAPTIVE PARAMETERS =====================
# These parameters now scale with video duration!

# Percentage-based time windows
DETECTION_TIME_WINDOW_PERCENT = (
    0.25  # 25% of video duration (increase for better catch rate)
)
TIME_THRESHOLD_PERCENT = (
    0.30  # 30% of video duration (Max time gap for spatial deduplication)
)

# Confidence-based confirmation thresholds
HIGH_CONFIDENCE_THRESHOLD = (
    0.75  # High confidence detections get immediate confirmation
)
LOW_CONFIDENCE_MIN_FRAMES = 2  # Low confidence needs multiple frames

# Calculate actual time values based on video duration
DETECTION_TIME_WINDOW = video_duration * DETECTION_TIME_WINDOW_PERCENT
TIME_THRESHOLD = video_duration * TIME_THRESHOLD_PERCENT

# For images or very short videos, use relaxed confirmation
if (
    IS_IMAGE or video_duration < 10.0
):  # Increased from 5.0 to 10.0 for better short video coverage
    DEFAULT_MIN_FRAMES = 1  # Single-frame confirmation for images/short videos
    if IS_IMAGE:
        print("📸 IMAGE MODE - Using single-frame detection")
    else:
        print("⚠️  SHORT VIDEO DETECTED - Using relaxed single-frame confirmation")
else:
    DEFAULT_MIN_FRAMES = 2  # Multi-frame confirmation for normal videos

# Note: Actual MIN_DETECTION_FRAMES will be dynamically set based on confidence

# Spatial threshold remains fixed (pixel-based, not time-based)
MIN_DISTANCE_THRESHOLD = 120  # pixels (tuned to catch edge cases like 102px duplicates)

# ===================== PARAMETER LOGGING =====================
print("=" * 60)
print("ADAPTIVE DETECTION PARAMETERS (v10 - 5-Class Road Damage)")
print("=" * 60)
print(f"Road Damage Classes: {', '.join(sorted(ROAD_DAMAGE_CLASSES))}")
print(f"Excluded Classes: {', '.join(sorted(EXCLUDED_CLASSES))}")
print()
if IS_IMAGE:
    print(f"Image Resolution: {width}x{height}")
else:
    print(
        f"Video Duration: {video_duration:.2f}s ({total_frames} frames @ {fps:.1f} FPS)"
    )
    print(
        f"Detection Time Window: {DETECTION_TIME_WINDOW:.2f}s ({DETECTION_TIME_WINDOW_PERCENT*100:.0f}% of video)"
    )
    print(
        f"Deduplication Window: {TIME_THRESHOLD:.2f}s ({TIME_THRESHOLD_PERCENT*100:.0f}% of video)"
    )
print(f"Min Frames (High Conf): 1, (Low Conf): {LOW_CONFIDENCE_MIN_FRAMES}")
print(f"Spatial Distance Threshold: {MIN_DISTANCE_THRESHOLD}px")
print(f"Tracker: {TRACKER}")
print()
print("Preprocessing Enabled:")
print(f"  - CLAHE (Lighting Normalization): {ENABLE_CLAHE}")
if ENABLE_CLAHE:
    print(f"    Clip Limit: {CLAHE_CLIP_LIMIT}, Grid Size: {CLAHE_GRID_SIZE}")
print(f"  - Denoising: {ENABLE_DENOISE}")
if ENABLE_DENOISE:
    print(f"    Strength: {DENOISE_H}")
print(f"  - Sharpening: {ENABLE_SHARPEN}")
print("=" * 60)
print()

if IS_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ===================== ROI =====================
# Expanded ROI to detect both signboards (upper) and road defects (lower)
ROI_LEFT = int(width * 0.0)
ROI_RIGHT = int(width * 1.0)
ROI_TOP = int(height * 0.05)  # Start from 5% to capture signboards
ROI_BOTTOM = int(height * 0.95)  # Extend to 95% to capture road defects

# ===================== JSON STRUCTURE =====================
detection_id = str(uuid.uuid4())

output_json = {
    "detection_id": detection_id,
    "input_path": INPUT_PATH,
    "input_type": "image" if IS_IMAGE else "video",
    "processed_at": datetime.utcnow().isoformat(),
    "input_info": {
        "total_frames": int(total_frames),
        "fps": float(fps),
        "duration": float(round(video_duration, 2)),
        "width": int(width),
        "height": int(height),
        "resolution": f"{width}x{height}",
    },
    "detection_config": {
        "detection_time_window": float(round(DETECTION_TIME_WINDOW, 2)),
        "time_threshold": float(round(TIME_THRESHOLD, 2)),
        "high_confidence_threshold": float(HIGH_CONFIDENCE_THRESHOLD),
        "low_confidence_min_frames": int(LOW_CONFIDENCE_MIN_FRAMES),
        "min_distance_threshold": int(MIN_DISTANCE_THRESHOLD),
        "tracker": TRACKER,
        "road_damage_classes": sorted(list(ROAD_DAMAGE_CLASSES)),
        "excluded_classes": sorted(list(EXCLUDED_CLASSES)),
    },
    "summary": {
        "total_frames": int(total_frames),
        "unique_defected_sign_board": 0,
        "unique_pothole": 0,
        "unique_road_crack": 0,
        "unique_damaged_road_marking": 0,
        "unique_good_sign_board": 0,  # Detected but not counted as damage
        "total_road_damage": 0,  # Sum of all damage classes
        "total_detections": 0,
        "frames_with_detections": 0,
        "detection_rate": 0.0,
    },
    "defected_sign_board_list": [],
    "pothole_list": [],
    "road_crack_list": [],
    "damaged_road_marking_list": [],
    "good_sign_board_list": [],  # Tracked separately
    "frames": [],
}

# Tracking structures
tracker_history = defaultdict(lambda: deque(maxlen=50))  # Increased for longer videos
confirmed = {}
counted_ids = set()

# Individual class counters
counted_defected_sign_board = set()
counted_pothole = set()
counted_road_crack = set()
counted_damaged_road_marking = set()
counted_good_sign_board = set()

spatial_locations = []
tracker_class_lock = {}  # Lock each tracker ID to its first detected class

# Rejection tracking for debugging
rejection_stats = {
    "multi_frame_pending": set(),  # Track IDs waiting for confirmation
    "spatial_duplicate": 0,
    "roi_outside": 0,
    "class_mismatch": 0,
}

frame_id = 0


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def find_closest_match(cx, cy, class_name):
    """Find the closest matching detection in spatial_locations"""
    closest = None
    min_distance = float("inf")

    for existing in spatial_locations:
        if existing["class"] == class_name:
            distance = calculate_distance((cx, cy), existing["center"])
            if distance < min_distance:
                min_distance = distance
                closest = {
                    "distance": distance,
                    "time": existing["time"],
                    "center": existing["center"],
                }

    return closest


def is_duplicate_location(cx, cy, class_name, current_time):
    """
    Check if this location/class was already counted recently.
    Returns (is_duplicate: bool, reason: str)
    """
    for existing in spatial_locations:
        prev_cx, prev_cy = existing["center"]
        prev_class = existing["class"]
        prev_time = existing["time"]

        # Calculate spatial distance
        distance = calculate_distance((cx, cy), (prev_cx, prev_cy))
        time_gap = current_time - prev_time

        # If same class, close in space, and within time window = duplicate
        if (
            prev_class == class_name
            and distance < MIN_DISTANCE_THRESHOLD
            and time_gap < TIME_THRESHOLD
        ):
            reason = f"{distance:.1f}px from existing, {time_gap:.2f}s ago"
            return True, reason

    return False, None


def preprocess_frame(frame):
    """
    Preprocess frame to handle lighting variations and improve detection consistency.
    Applies CLAHE, denoising, and optional sharpening based on config flags.
    """
    processed = frame.copy()

    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if ENABLE_CLAHE:
        # Convert to LAB color space (separates luminance from color)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel (luminance) only
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE
        )
        l = clahe.apply(l)

        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Denoising (reduces noise while preserving edges)
    if ENABLE_DENOISE:
        processed = cv2.fastNlMeansDenoisingColored(
            processed,
            None,
            h=DENOISE_H,
            hColor=DENOISE_H,
            templateWindowSize=DENOISE_TEMPLATE_WINDOW,
            searchWindowSize=DENOISE_SEARCH_WINDOW,
        )

    # 3. Sharpening (optional - enhances edges but may amplify noise)
    if ENABLE_SHARPEN:
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        processed = cv2.filter2D(processed, -1, kernel)

    return processed


# ===================== PROCESSING =====================
if IS_IMAGE:
    print("Processing image...")
else:
    print("Processing video...")
print("-" * 60)

# Create frame iterator based on input type
if IS_IMAGE:
    frames_to_process = [frame]  # Single frame for images
else:
    frames_to_process = None  # Will read from cap in loop

frame_id = 0
processing = True

while processing:
    # Get frame based on input type
    if IS_IMAGE:
        if frame_id >= 1:  # Only process once for images
            break
        current_frame = frames_to_process[0]
        ret = True
    else:
        ret, current_frame = cap.read()
        if not ret:
            break

    frame_id += 1
    current_time = frame_id / fps

    # Apply preprocessing to handle lighting variations
    # preprocessed_frame = preprocess_frame(current_frame)

    # Run detection on preprocessed frame
    results = model.track(
        current_frame,  # preprocessed_frame
        persist=True,
        conf=CONF_THRESHOLD,
        tracker=TRACKER,
        verbose=False,
    )

    annotated_frame = results[0].plot()

    frame_data = {"frame_id": int(frame_id), "detections": []}

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for tid, cid, box, conf in zip(track_ids, class_ids, boxes, confidences):
            tid = int(tid)
            cid = int(cid)

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            class_name = str(model.names[cid])

            # ROI check
            in_roi = ROI_LEFT < cx < ROI_RIGHT and ROI_TOP < cy < ROI_BOTTOM
            if not in_roi:
                print(
                    f"⊗ Frame {frame_id}: ID {tid} - {class_name} OUTSIDE ROI at ({cx}, {cy})"
                )
                rejection_stats["roi_outside"] += 1
                continue

            # ===== CLASS LOCKING: Ensure each tracker ID maintains consistent class =====
            if tid in tracker_class_lock:
                # ID already has a locked class - verify consistency
                locked_class = tracker_class_lock[tid]
                if locked_class != class_name:
                    print(
                        f"⚠️  Frame {frame_id}: ID {tid} CLASS MISMATCH - "
                        f"Expected '{locked_class}', got '{class_name}' (REJECTED)"
                    )
                    rejection_stats["class_mismatch"] += 1
                    continue  # Skip this detection entirely
            else:
                # First time seeing this ID - lock it to this class
                tracker_class_lock[tid] = class_name

                # Check if it's a road damage class or excluded class
                is_damage_class = class_name in ROAD_DAMAGE_CLASSES
                class_type = "DAMAGE" if is_damage_class else "EXCLUDED"
                print(
                    f"🔒 Frame {frame_id}: ID {tid} LOCKED to '{class_name}' [{class_type}]"
                )

            # Update temporal tracker
            tracker_history[tid].append(current_time)

            # Check how many recent detections this track has
            recent_detections = [
                t
                for t in tracker_history[tid]
                if current_time - t <= DETECTION_TIME_WINDOW
            ]

            # Confidence-based multi-frame requirement
            # High confidence (≥0.75) = immediate confirmation (1 frame)
            # Low confidence (<0.75) = needs multiple frames
            if conf >= HIGH_CONFIDENCE_THRESHOLD:
                MIN_FRAMES_NEEDED = 1  # High confidence = single frame OK
            else:
                MIN_FRAMES_NEEDED = (
                    LOW_CONFIDENCE_MIN_FRAMES  # Low confidence = need 2+ frames
                )

            # Debug: Show all detections
            print(
                f"🔍 Frame {frame_id:3d}: ID {tid:2d} - {class_name:25s} "
                f"conf={conf:.2f}, frames={len(recent_detections)}/{MIN_FRAMES_NEEDED}"
            )

            # Multi-frame confirmation
            if len(recent_detections) >= MIN_FRAMES_NEEDED and tid not in confirmed:

                # Spatial deduplication check
                is_dup, reason = is_duplicate_location(cx, cy, class_name, current_time)

                if not is_dup:
                    # This is a NEW confirmed detection
                    confirmed[tid] = {
                        "detection_id": tid,
                        "type": class_name,
                        "first_detected_frame": int(frame_id),
                        "first_detected_time": float(round(current_time, 2)),
                        "confidence": float(round(conf, 3)),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }

                    # Add to appropriate list based on class type
                    if class_name == "defected_sign_board":
                        output_json["defected_sign_board_list"].append(confirmed[tid])
                        counted_defected_sign_board.add(tid)
                    elif class_name == "pothole":
                        output_json["pothole_list"].append(confirmed[tid])
                        counted_pothole.add(tid)
                    elif class_name == "road_crack":
                        output_json["road_crack_list"].append(confirmed[tid])
                        counted_road_crack.add(tid)
                    elif class_name == "damaged_road_marking":
                        output_json["damaged_road_marking_list"].append(confirmed[tid])
                        counted_damaged_road_marking.add(tid)
                    elif class_name == "good_sign_board":
                        output_json["good_sign_board_list"].append(confirmed[tid])
                        counted_good_sign_board.add(tid)

                    counted_ids.add(tid)

                    # Add to spatial locations for future deduplication
                    spatial_locations.append(
                        {"center": (cx, cy), "time": current_time, "class": class_name}
                    )

                    # Remove from pending if it was there
                    rejection_stats["multi_frame_pending"].discard(tid)

                    # Show damage indicator
                    is_damage = class_name in ROAD_DAMAGE_CLASSES
                    damage_indicator = "🚨 DAMAGE" if is_damage else "ℹ️  INFO"

                    print(
                        f"✅ Frame {frame_id:3d} ({current_time:5.2f}s): "
                        f"CONFIRMED ID {tid:2d} - {class_name:25s} at ({cx:4d}, {cy:4d}) "
                        f"[conf={conf:.2f}, frames={len(recent_detections)}] {damage_indicator}"
                    )
                else:
                    # Spatial duplicate
                    rejection_stats["spatial_duplicate"] += 1
                    closest = find_closest_match(cx, cy, class_name)
                    print(
                        f"❌ Frame {frame_id:3d} ({current_time:5.2f}s): "
                        f"DUPLICATE ID {tid:2d} - {class_name:25s}"
                    )
                    print(f"   Reason: {reason}")

            elif tid not in confirmed:
                # Still waiting for multi-frame confirmation
                rejection_stats["multi_frame_pending"].add(tid)
                print(
                    f"   ⏳ Waiting for {MIN_FRAMES_NEEDED - len(recent_detections)} more frame(s)..."
                )

            # Count total detections (only confirmed ones)
            if tid in confirmed:
                output_json["summary"]["total_detections"] += 1

                # Add to frame data
                frame_data["detections"].append(
                    {
                        "frame_id": int(frame_id),
                        "detection_id": tid,
                        "type": class_name,
                        "confidence": float(round(conf, 3)),
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                        },
                        "center": {"x": int(cx), "y": int(cy)},
                        "area": int((x2 - x1) * (y2 - y1)),
                    }
                )

                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)

    if frame_data["detections"]:
        output_json["frames"].append(frame_data)
        output_json["summary"]["frames_with_detections"] += 1

    # ===================== DRAW ROI =====================
    overlay = annotated_frame.copy()
    cv2.rectangle(
        overlay, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), -1
    )
    cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
    cv2.rectangle(
        annotated_frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 3
    )

    # Calculate road damage count (excluding good_sign_board)
    road_damage_count = (
        len(counted_defected_sign_board)
        + len(counted_pothole)
        + len(counted_road_crack)
        + len(counted_damaged_road_marking)
    )

    cv2.putText(
        annotated_frame,
        f"Road Damage: {road_damage_count} | Total: {len(counted_ids)} | Frame: {frame_id}/{total_frames}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3,
    )

    # Save or write frame
    if IS_IMAGE:
        cv2.imwrite(OUTPUT_PATH, annotated_frame)
        print(f"✓ Image saved: {OUTPUT_PATH}")
    else:
        video_writer.write(annotated_frame)

    # Display frame
    window_title = (
        f"{'Image' if IS_IMAGE else 'Video'} Detection v10 (5-Class Road Damage)"
    )
    cv2.imshow(window_title, annotated_frame)

    if cv2.waitKey(1 if IS_VIDEO else 0) & 0xFF == ord("q"):
        break

# ===================== FINALIZE =====================
if IS_VIDEO:
    cap.release()
    video_writer.release()
cv2.destroyAllWindows()

# Update summary with individual class counts
output_json["summary"]["unique_defected_sign_board"] = int(
    len(counted_defected_sign_board)
)
output_json["summary"]["unique_pothole"] = int(len(counted_pothole))
output_json["summary"]["unique_road_crack"] = int(len(counted_road_crack))
output_json["summary"]["unique_damaged_road_marking"] = int(
    len(counted_damaged_road_marking)
)
output_json["summary"]["unique_good_sign_board"] = int(len(counted_good_sign_board))

# Calculate total road damage (excluding good_sign_board)
output_json["summary"]["total_road_damage"] = (
    len(counted_defected_sign_board)
    + len(counted_pothole)
    + len(counted_road_crack)
    + len(counted_damaged_road_marking)
)

output_json["summary"]["detection_rate"] = float(
    round((output_json["summary"]["frames_with_detections"] / total_frames) * 100, 2)
)

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(output_json, f, indent=2)

# ===================== FINAL REPORT =====================
print("\n" + "=" * 60)
print(
    f"✓ PROCESSING COMPLETE (v10 - 5-Class Road Damage {'Image' if IS_IMAGE else 'Video'})"
)
print("=" * 60)
if IS_IMAGE:
    print(f"Input Type: IMAGE")
    print(f"Resolution: {width}x{height}")
else:
    print(f"Input Type: VIDEO")
    print(f"Video Duration: {video_duration:.2f}s")

print()
print("=" * 60)
print("ROAD DAMAGE DETECTIONS (Counted)")
print("=" * 60)
print(f"Defected Sign Boards: {len(counted_defected_sign_board)}")
print(f"Potholes: {len(counted_pothole)}")
print(f"Road Cracks: {len(counted_road_crack)}")
print(f"Damaged Road Markings: {len(counted_damaged_road_marking)}")
print(f"─" * 60)
print(f"Total Road Damage: {output_json['summary']['total_road_damage']}")
print()
print("=" * 60)
print("OTHER DETECTIONS (Not Counted as Damage)")
print("=" * 60)
print(f"Good Sign Boards: {len(counted_good_sign_board)}")
print()
print("=" * 60)
print(f"Total Unique Objects: {len(counted_ids)}")
print(f"Total Detections (all frames): {output_json['summary']['total_detections']}")
print(f"Detection Rate: {output_json['summary']['detection_rate']}%")
print()
print("Rejection Statistics:")
print(
    f"  - Multi-frame pending (not confirmed): {len(rejection_stats['multi_frame_pending'])}"
)
print(f"  - Spatial duplicates rejected: {rejection_stats['spatial_duplicate']}")
print(f"  - Class mismatches (ID switching): {rejection_stats['class_mismatch']}")
print(f"  - Outside ROI: {rejection_stats['roi_outside']}")
print()
if not IS_IMAGE:
    print("Adaptive Parameters Used:")
    print(
        f"  - Detection window: {DETECTION_TIME_WINDOW:.2f}s ({DETECTION_TIME_WINDOW_PERCENT*100:.0f}%)"
    )
    print(
        f"  - Deduplication window: {TIME_THRESHOLD:.2f}s ({TIME_THRESHOLD_PERCENT*100:.0f}%)"
    )
    print(f"  - Min frames required: {DEFAULT_MIN_FRAMES}")
    print()
print(f"JSON saved: {OUTPUT_JSON_PATH}")
print(f"Output saved: {OUTPUT_PATH}")
print("=" * 60)

# Print detailed detection lists
if counted_defected_sign_board:
    print(f"\nDefected Sign Boards ({len(counted_defected_sign_board)}) [DAMAGE]:")
    for item in output_json["defected_sign_board_list"]:
        print(
            f"  ID {item['detection_id']}: {item['type']} "
            f"(Frame {item['first_detected_frame']}, "
            f"Time {item['first_detected_time']}s, "
            f"Conf {item['confidence']})"
        )

if counted_pothole:
    print(f"\nPotholes ({len(counted_pothole)}) [DAMAGE]:")
    for item in output_json["pothole_list"]:
        print(
            f"  ID {item['detection_id']}: {item['type']} "
            f"(Frame {item['first_detected_frame']}, "
            f"Time {item['first_detected_time']}s, "
            f"Conf {item['confidence']})"
        )

if counted_road_crack:
    print(f"\nRoad Cracks ({len(counted_road_crack)}) [DAMAGE]:")
    for item in output_json["road_crack_list"]:
        print(
            f"  ID {item['detection_id']}: {item['type']} "
            f"(Frame {item['first_detected_frame']}, "
            f"Time {item['first_detected_time']}s, "
            f"Conf {item['confidence']})"
        )

if counted_damaged_road_marking:
    print(f"\nDamaged Road Markings ({len(counted_damaged_road_marking)}) [DAMAGE]:")
    for item in output_json["damaged_road_marking_list"]:
        print(
            f"  ID {item['detection_id']}: {item['type']} "
            f"(Frame {item['first_detected_frame']}, "
            f"Time {item['first_detected_time']}s, "
            f"Conf {item['confidence']})"
        )

if counted_good_sign_board:
    print(f"\nGood Sign Boards ({len(counted_good_sign_board)}) [INFO ONLY]:")
    for item in output_json["good_sign_board_list"]:
        print(
            f"  ID {item['detection_id']}: {item['type']} "
            f"(Frame {item['first_detected_frame']}, "
            f"Time {item['first_detected_time']}s, "
            f"Conf {item['confidence']})"
        )

if not counted_ids:
    print("\n⚠️  No detections confirmed!")
    print("Check the rejection statistics above to see why.")
