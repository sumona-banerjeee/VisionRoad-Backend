import cv2
import json
import uuid
from datetime import datetime
from ultralytics import YOLO

# ===================== CONFIG =====================
MODEL_PATH = r"models\pothole-signboard.pt"
VIDEO_PATH = r"Test\video\Dash_Cam_Highway_Curve_Video.mp4"
OUTPUT_VIDEO_PATH = r"Test\output\outputvideo.mp4"
OUTPUT_JSON_PATH = r"Test\output\output.json"
CONF_THRESHOLD = 0.6

TRACKER = "bytetrack.yaml"

# ===================== LOAD MODEL =====================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = float(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        "resolution": f"{width}x{height}"
    },

    "summary": {
        "total_frames": int(total_frames),
        "unique_signboards": 0,
        "total_detections": 0,
        "frames_with_detections": 0,
        "detection_rate": 0.0
    },

    "signboard_list": [],
    "frames": []
}

first_seen = {}
counted_ids = set()

frame_id = 0

# ===================== PROCESS VIDEO =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    results = model.track(
        frame, 
        persist=True,
        conf=CONF_THRESHOLD, 
        tracker=TRACKER
    )
    annotated_frame = results[0].plot()

    frame_data = {
        "frame_id": int(frame_id),
        "signboards": []
    }

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

            in_roi = ROI_LEFT < cx < ROI_RIGHT and ROI_TOP < cy < ROI_BOTTOM
            if not in_roi:
                continue

            class_name = str(model.names[cid])

            output_json["summary"]["total_detections"] += 1

            # ---------- FIRST TIME SEEN ----------
            if tid not in first_seen:
                first_seen[tid] = {
                    "signboard_id": tid,
                    "type": class_name,
                    "first_detected_frame": int(frame_id),
                    "first_detected_time": float(round(frame_id / fps, 2)),
                    "confidence": float(round(conf, 3))
                }

                output_json["signboard_list"].append(first_seen[tid])
                counted_ids.add(tid)

            # ---------- FRAME DATA ----------
            frame_data["signboards"].append({
                "frame_id": int(frame_id),
                "signboard_id": tid,
                "type": class_name,
                "confidence": float(round(conf, 3)),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "center": {
                    "x": int(cx),
                    "y": int(cy)
                },
                "area": int((x2 - x1) * (y2 - y1))
            })

            cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)

    if frame_data["signboards"]:
        output_json["frames"].append(frame_data)
        output_json["summary"]["frames_with_detections"] += 1

    # ===================== DRAW ROI =====================
    overlay = annotated_frame.copy()
    cv2.rectangle(
        overlay,
        (ROI_LEFT, ROI_TOP),
        (ROI_RIGHT, ROI_BOTTOM),
        (0, 255, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
    cv2.rectangle(
        annotated_frame,
        (ROI_LEFT, ROI_TOP),
        (ROI_RIGHT, ROI_BOTTOM),
        (0, 255, 0),
        3
    )

    cv2.putText(
        annotated_frame,
        f"Total Counted: {len(counted_ids)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    video_writer.write(annotated_frame)
    cv2.imshow("Signboard Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
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

print(" Processing complete")
print(f" JSON saved at: {OUTPUT_JSON_PATH}")
print(f" Video saved at: {OUTPUT_VIDEO_PATH}")