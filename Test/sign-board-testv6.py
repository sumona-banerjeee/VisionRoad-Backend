import cv2
import json
import uuid
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO

# ===================== CONFIG =====================
MODEL_PATH = r"models\pothole-signboard.pt"
VIDEO_PATH = r"Test\video\road-sign-ml.mp4"
OUTPUT_VIDEO_PATH = r"Test\output\outputvideo.mp4"
OUTPUT_JSON_PATH = r"Test\output\output.json"

CONF_THRESHOLD = 0.6
TRACKER = "bytetrack.yaml"

# Multi-frame confirmation settings
MIN_DETECTION_FRAMES = 2  # Must appear in at least 2 frames
DETECTION_TIME_WINDOW = 1.5  # Within 1.5 seconds

# Spatial deduplication settings
MIN_DISTANCE_THRESHOLD = 100  # pixels - minimum distance to consider as different signboard 
TIME_THRESHOLD = (
    4.0  # seconds - time window for spatial deduplication 
)

# ===================== LOAD MODEL =====================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = float(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
        "duration": float(round(total_frames / fps, 2)),
        "width": int(width),
        "height": int(height),
        "resolution": f"{width}x{height}",
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
tracker_history = defaultdict(
    lambda: deque(maxlen=20)
)  # Track ID -> deque of timestamps
confirmed = {}  # Track ID -> signboard data (after multi-frame confirmation)
counted_ids = set()
spatial_locations = (
    []
)  # List of {center: (x,y), time: T, class: name} for deduplication

frame_id = 0


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def is_duplicate_location(cx, cy, class_name, current_time):
    """
    Check if this location/class was already counted recently.
    Returns True if duplicate, False if unique.
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
            return True

    return False


# ===================== PROCESS VIDEO =====================
print(f"Processing video: {VIDEO_PATH}")
print(
    f"Multi-frame confirmation: {MIN_DETECTION_FRAMES} frames in {DETECTION_TIME_WINDOW}s window"
)
print(
    f"Spatial deduplication: {MIN_DISTANCE_THRESHOLD}px threshold, {TIME_THRESHOLD}s window"
)
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

            # Multi-frame confirmation: Only confirm if seen in MIN_DETECTION_FRAMES+
            if len(recent_detections) >= MIN_DETECTION_FRAMES and tid not in confirmed:

                # Spatial deduplication check
                if not is_duplicate_location(cx, cy, class_name, current_time):
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

                    print(
                        f"✓ Frame {frame_id}: Confirmed signboard ID {tid} ({class_name}) at ({cx}, {cy})"
                    )
                else:
                    print(
                        f"✗ Frame {frame_id}: Rejected duplicate - ID {tid} ({class_name}) too close to existing detection"
                    )

            # Count total detections (even unconfirmed)
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
    cv2.imshow("Signboard Detection v6", annotated_frame)

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

print("\n" + "=" * 60)
print("✓ Processing complete!")
print("=" * 60)
print(f"Unique signboards confirmed: {len(counted_ids)}")
print(f"Total detections (all frames): {output_json['summary']['total_detections']}")
print(f"Detection rate: {output_json['summary']['detection_rate']}%")
print(f"\nJSON saved: {OUTPUT_JSON_PATH}")
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
