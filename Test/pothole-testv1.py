# test_detection.py - Test pothole detection with unique pothole counting
# Usage: python test_detection.py [model_type] [device]
# model_type: pt, onnx, engine (default: engine)
# device: cpu, gpu (default: gpu)

import cv2
import time
import sys
import os
from ultralytics import YOLO

# Configuration
MODEL_PATHS = {
    "pt": "models/pothole-detector.pt",
    "onnx": "models/pothole-detector.onnx", 
    "engine": "models/pothole-detector.engine"
}

CONFIDENCE_THRESHOLD = 0.80  # 80% confidence (matching your original tests)

# Parse arguments
model_type = sys.argv[1] if len(sys.argv) > 1 else "engine"
device_type = sys.argv[2] if len(sys.argv) > 2 else "gpu"

# Validate model type
if model_type not in MODEL_PATHS:
    print(f"Invalid model type: {model_type}")
    print("Valid options: pt, onnx, engine")
    sys.exit(1)

model_path = MODEL_PATHS[model_type]
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    sys.exit(1)

# Set device
device = "cuda:0" if device_type == "gpu" else "cpu"
print(f"\n{'='*60}")
print(f"POTHOLE DETECTION TEST - UNIQUE COUNT MODE")
print(f"{'='*60}")
print(f"Model: {model_path}")
print(f"Device: {device}")
print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
print(f"{'='*60}\n")

# Load model
print("Loading model...")
start_load = time.time()

if model_type in ["onnx", "engine"]:
    model = YOLO(model_path, task='detect')
else:
    model = YOLO(model_path)
    if device == "cuda:0":
        try:
            model.to(device)
        except Exception as e:
            print(f"Warning: Could not move model to GPU: {e}")

load_time = time.time() - start_load
print(f"Model loaded in {load_time:.2f} seconds")

# Warmup
print("Warming up model...")
import numpy as np
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
for _ in range(3):
    try:
        model.predict(dummy, verbose=False, device=device)
    except:
        pass
print("Warmup complete")

# Find a test video
video_paths = [
    "uploads/b01308b9-c4b9-4aca-b92f-e1bc03fded5d.mp4",
    "uploads/test.mp4",
    "test.mp4",
]

video_path = None
for vp in video_paths:
    if os.path.exists(vp):
        video_path = vp
        break

if video_path is None:
    print("Error: No video found!")
    print("Place a test video at 'uploads/test.mp4'")
    sys.exit(1)

# Open video
print(f"\nOpening video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    sys.exit(1)

# Get video properties
fps_video = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps_video:.1f} FPS, {total_frames} frames")

# Tracking variables
frame_count = 0
unique_potholes = set()  # Track unique pothole IDs
fps_times = []

print("\nProcessing video (press 'Q' to quit early)...")
print("-" * 60)

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Video ended - don't loop
    
    frame_count += 1
    
    # Run detection with tracking
    # FIXED: Added persist=True and imgsz parameter
    start = time.time()
    results = model.track(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        tracker="bytetrack.yaml",
        persist=True,  # ✅ CRITICAL FIX: Enable tracking persistence
        imgsz=640,     # ✅ CRITICAL FIX: Match export image size
        verbose=False,
        device=device
    )
    inference_time = time.time() - start
    fps_times.append(inference_time)
    
    # Calculate FPS (rolling average of last 30 frames)
    if len(fps_times) > 30:
        fps_times.pop(0)
    current_fps = 1.0 / (sum(fps_times) / len(fps_times))
    
    # Draw detections and count unique potholes
    annotated_frame = frame.copy()
    detections_in_frame = 0
    
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None
            
            if ids is not None:
                for box, conf, track_id in zip(boxes, confs, ids):
                    track_id = int(track_id)
                    
                    # Add to unique potholes set
                    unique_potholes.add(track_id)
                    
                    x1, y1, x2, y2 = map(int, box)
                    detections_in_frame += 1
                    
                    # Draw box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"ID:{track_id} ({conf*100:.0f}%)"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Draw info overlay
    progress = (frame_count / total_frames) * 100
    info_lines = [
        f"Model: {model_type.upper()} | Device: {device} | Conf: {CONFIDENCE_THRESHOLD*100:.0f}%",
        f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames} ({progress:.1f}%)",
        f"UNIQUE POTHOLES DETECTED: {len(unique_potholes)}",
        f"Detections in this frame: {detections_in_frame}"
    ]
    
    y_offset = 30
    for i, line in enumerate(info_lines):
        # Highlight unique count line
        color = (0, 0, 255) if i == 2 else (255, 255, 255)
        thickness = 3 if i == 2 else 2
        cv2.putText(annotated_frame, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness + 2)  # Shadow
        cv2.putText(annotated_frame, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
        y_offset += 30
    
    # Show frame
    cv2.imshow(f"Pothole Detection - Unique Count", annotated_frame)
    
    # Handle key press (1ms wait for smooth playback)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nStopped by user")
        break

total_time = time.time() - start_time
cap.release()
cv2.destroyAllWindows()

# Print summary
avg_fps = frame_count / total_time if total_time > 0 else 0

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Model: {model_type.upper()}")
print(f"Device: {device}")
print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
print(f"-" * 60)
print(f"Frames processed: {frame_count}/{total_frames}")
print(f"Total processing time: {total_time:.1f} seconds")
print(f"Average FPS: {avg_fps:.1f}")
print(f"-" * 60)
print(f">>> UNIQUE POTHOLES DETECTED: {len(unique_potholes)} <<<")
print(f"{'='*60}")

# Print all unique pothole IDs
if unique_potholes:
    print(f"\nUnique Pothole IDs: {sorted(unique_potholes)}")