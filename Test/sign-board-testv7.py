"""
===============
NEW MODEL
===============
"""

import cv2
import json
import uuid
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO

# ===================== CONFIG =====================
MODEL_PATH = r"models\pothole-signboard.pt"
VIDEO_PATH = r"Test\video\sign-short.mp4"
OUTPUT_VIDEO_PATH = r"Test\output\outputvideo-v7.mp4"
OUTPUT_JSON_PATH = r"Test\output\output-v7.json"

CONF_THRESHOLD = 0.6
TRACKER = "bytetrack.yaml"

# ===================== LOAD MODEL =====================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = float(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps  # Total video duration in seconds

# ===================== ADAPTIVE PARAMETERS =====================
# These parameters now scale with video duration!

# Percentage-based time windows
DETECTION_TIME_WINDOW_PERCENT = (
    0.25  # 25% of video duration (increased for better catch rate)
)
TIME_THRESHOLD_PERCENT = 0.30  # 30% of video duration

# Confidence-based confirmation thresholds
HIGH_CONFIDENCE_THRESHOLD = (
    0.75  # High confidence detections get immediate confirmation
)
LOW_CONFIDENCE_MIN_FRAMES = 2  # Low confidence needs multiple frames

# Calculate actual time values based on video duration
DETECTION_TIME_WINDOW = video_duration * DETECTION_TIME_WINDOW_PERCENT
TIME_THRESHOLD = video_duration * TIME_THRESHOLD_PERCENT

# For very short videos, use relaxed confirmation
if video_duration < 5.0:
    DEFAULT_MIN_FRAMES = 1  # Single-frame confirmation for short videos
    print("⚠️  SHORT VIDEO DETECTED - Using relaxed single-frame confirmation")
else:
    DEFAULT_MIN_FRAMES = 2  # Multi-frame confirmation for normal videos

# Note: Actual MIN_DETECTION_FRAMES will be dynamically set based on confidence

# Spatial threshold remains fixed (pixel-based, not time-based)
MIN_DISTANCE_THRESHOLD = 100  # pixels

# ===================== PARAMETER LOGGING =====================
print("=" * 60)
print("ADAPTIVE DETECTION PARAMETERS (v7)")
print("=" * 60)
print(f"Video Duration: {video_duration:.2f}s ({total_frames} frames @ {fps:.1f} FPS)")
print(
    f"Detection Time Window: {DETECTION_TIME_WINDOW:.2f}s ({DETECTION_TIME_WINDOW_PERCENT*100:.0f}% of video)"
)
print(
    f"Deduplication Window: {TIME_THRESHOLD:.2f}s ({TIME_THRESHOLD_PERCENT*100:.0f}% of video)"
)
print(f"Min Frames (High Conf): 1, (Low Conf): {LOW_CONFIDENCE_MIN_FRAMES}")
print(f"Spatial Distance Threshold: {MIN_DISTANCE_THRESHOLD}px")
print("=" * 60)
print()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# ===================== ROI =====================
ROI_LEFT = int(width * 0.0)
ROI_RIGHT = int(width * 1.0)
ROI_TOP = int(height * 0.10)
ROI_BOTTOM = int(height * 0.70)

# ===================== JSON STRUCTURE =====================
video_id = str(uuid.uuid4())

output_json = {
    "video_id": video_id,
    "video_path": VIDEO_PATH,
    "processed_at": datetime.utcnow().isoformat(),
    "video_info": {
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
    },
    "summary": {
        "total_frames": int(total_frames),
        "unique_signboards": 0,
        "total_detections": 0,
        "frames_with_detections": 0,
        "detection_rate": 0.0,
    },
    "signboard_list": [],
    "frames": [],
}

# Tracking structures
tracker_history = defaultdict(lambda: deque(maxlen=50))  # Increased for longer videos
confirmed = {}
counted_ids = set()
spatial_locations = []

# Rejection tracking for debugging
rejection_stats = {
    "multi_frame_pending": set(),  # Track IDs waiting for confirmation
    "spatial_duplicate": 0,
    "roi_outside": 0,
}

frame_id = 0


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def find_closest_match(cx, cy, class_name):
    """Find the closest matching signboard in spatial_locations"""
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


# ===================== PROCESS VIDEO =====================
print("Processing video...")
print("-" * 60)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    current_time = frame_id / fps

    results = model.track(
        frame, persist=True, conf=CONF_THRESHOLD, tracker=TRACKER, verbose=False
    )

    annotated_frame = results[0].plot()

    frame_data = {"frame_id": int(frame_id), "signboards": []}

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

            # ROI check
            in_roi = ROI_LEFT < cx < ROI_RIGHT and ROI_TOP < cy < ROI_BOTTOM
            if not in_roi:
                print(
                    f"⊗ Frame {frame_id}: ID {tid} - {class_name} OUTSIDE ROI at ({cx}, {cy})"
                )
                rejection_stats["roi_outside"] += 1
                continue

            class_name = str(model.names[cid])

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
                    # This is a NEW confirmed signboard
                    confirmed[tid] = {
                        "signboard_id": tid,
                        "type": class_name,
                        "first_detected_frame": int(frame_id),
                        "first_detected_time": float(round(current_time, 2)),
                        "confidence": float(round(conf, 3)),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }

                    output_json["signboard_list"].append(confirmed[tid])
                    counted_ids.add(tid)

                    # Add to spatial locations for future deduplication
                    spatial_locations.append(
                        {"center": (cx, cy), "time": current_time, "class": class_name}
                    )

                    # Remove from pending if it was there
                    rejection_stats["multi_frame_pending"].discard(tid)

                    print(
                        f"✅ Frame {frame_id:3d} ({current_time:5.2f}s): "
                        f"CONFIRMED ID {tid:2d} - {class_name:25s} at ({cx:4d}, {cy:4d}) "
                        f"[conf={conf:.2f}, frames={len(recent_detections)}]"
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
                frame_data["signboards"].append(
                    {
                        "frame_id": int(frame_id),
                        "signboard_id": tid,
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

    if frame_data["signboards"]:
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

    cv2.putText(
        annotated_frame,
        f"Confirmed: {len(counted_ids)} | Frame: {frame_id}/{total_frames}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3,
    )

    video_writer.write(annotated_frame)
    cv2.imshow("Signboard Detection v7 (Adaptive)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===================== FINALIZE =====================
cap.release()
video_writer.release()
cv2.destroyAllWindows()

output_json["summary"]["unique_signboards"] = int(len(counted_ids))
output_json["summary"]["detection_rate"] = float(
    round((output_json["summary"]["frames_with_detections"] / total_frames) * 100, 2)
)

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(output_json, f, indent=2)

# ===================== FINAL REPORT =====================
print("\n" + "=" * 60)
print("✓ PROCESSING COMPLETE (v7 - Adaptive)")
print("=" * 60)
print(f"Video Duration: {video_duration:.2f}s")
print(f"Unique Signboards Confirmed: {len(counted_ids)}")
print(f"Total Detections (all frames): {output_json['summary']['total_detections']}")
print(f"Detection Rate: {output_json['summary']['detection_rate']}%")
print()
print("Rejection Statistics:")
print(
    f"  - Multi-frame pending (not confirmed): {len(rejection_stats['multi_frame_pending'])}"
)
print(f"  - Spatial duplicates rejected: {rejection_stats['spatial_duplicate']}")
print(f"  - Outside ROI: {rejection_stats['roi_outside']}")
print()
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
print(f"Video saved: {OUTPUT_VIDEO_PATH}")
print("=" * 60)

# Print detailed signboard list
if counted_ids:
    print("\nConfirmed Signboards:")
    for sb in output_json["signboard_list"]:
        print(
            f"  ID {sb['signboard_id']}: {sb['type']} "
            f"(Frame {sb['first_detected_frame']}, "
            f"Time {sb['first_detected_time']}s, "
            f"Conf {sb['confidence']})"
        )
else:
    print("\n⚠️  No signboards confirmed!")
    print("Check the rejection statistics above to see why.")
