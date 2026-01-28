import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# 1. Load pre-trained model
model = YOLO(
    r"D:\VisionRoad-Backend\models\best-board-v2.pt"
)

# 2. Open the video file
video_path = r"D:\VisionRoad-Backend\Test\video\traffic sign test video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer to save output
output_path = r"D:\VisionRoad-Backend\Test\output\outputvideo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ========== ROI CONFIGURATION FOR ROADSIDE DETECTION ==========
# Configure for RIGHT side of road (where signs typically appear)
# Adjust these values based on which side your signs appear

SIDE = "BOTH"  # Change to "LEFT" if signs are on left side, or "BOTH" for both sides

if SIDE == "RIGHT":
    # Right side detection zone
    ROI_LEFT = int(width * 0.55)      # Start from 55% of frame width
    ROI_RIGHT = int(width * 0.95)     # Extend to 95% of frame width
elif SIDE == "LEFT":
    # Left side detection zone
    ROI_LEFT = int(width * 0.05)      # Start from 5% of frame width
    ROI_RIGHT = int(width * 0.45)     # Extend to 45% of frame width
else:  # BOTH
    # Full width but focused on sides - EXTENDED COVERAGE
    ROI_LEFT = int(width * 0.0)       # Changed from 0.05 to 0.0 (start from very edge)
    ROI_RIGHT = int(width * 1.0)      # Changed from 0.95 to 1.0 (extend to very edge)

# Vertical boundaries (middle section of frame where signs are clearly visible)
ROI_TOP = int(height * 0.10)          # Start from 25% of frame height
ROI_BOTTOM = int(height * 0.70)       # End at 70% of frame height

# Optional: Minimum confidence for counting
MIN_CONFIDENCE_FOR_COUNTING = 0.60    # Only count high-confidence detections

# Tracking dictionaries
counted_objects = set()               # Objects that have been counted
class_wise_counts = defaultdict(set)  # Track unique IDs per class
object_positions = {}                 # Track previous positions

print("="*60)
print("ROADSIDE SIGN DETECTION & COUNTING")
print("="*60)
print(f"Video: {width}x{height} @ {fps} FPS")
print(f"Detection Side: {SIDE}")
print(f"ROI Box: X[{ROI_LEFT}-{ROI_RIGHT}], Y[{ROI_TOP}-{ROI_BOTTOM}]")
print("="*60)
print("Starting video analysis... Press 'q' to exit.\n")

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("End of video or cannot read the frame.")
        break
    
    frame_count += 1

    # 3. Run YOLOv8 tracking with persistence
    results = model.track(frame, persist=True, conf=0.60)
    
    # 5. Visualize the results with YOLO annotations FIRST
    annotated_frame = results[0].plot()
    
    # 4. Process detections AFTER creating annotated_frame
    if results[0].boxes.id is not None:
        # Get tracking IDs, class IDs, bounding boxes, and confidence scores
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for track_id, cls_id, box, conf in zip(track_ids, cls_ids, boxes, confidences):
            x1, y1, x2, y2 = box
            
            # Calculate center point of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Get class name
            class_name = model.names[cls_id]
            
            # ========== ROADSIDE ROI COUNTING LOGIC ==========
            # Check if object center is inside the ROI box
            is_in_roi = (ROI_LEFT < center_x < ROI_RIGHT and 
                        ROI_TOP < center_y < ROI_BOTTOM)
            
            # Check if confidence is high enough
            is_confident = conf >= MIN_CONFIDENCE_FOR_COUNTING
            
            # Count only if inside ROI, confident, and not already counted
            if is_in_roi and is_confident and track_id not in counted_objects:
                counted_objects.add(track_id)
                class_wise_counts[class_name].add(track_id)
                print(f"✓ [Frame {frame_count}] Counted: {class_name} (ID: {track_id}, Conf: {conf:.2f})")
            
            # Update position tracking
            object_positions[track_id] = (center_x, center_y)
            
            # Optional: Draw center point for tracked objects in ROI
            if is_in_roi:
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 255), -1)

    # ========== DRAW ROI COUNTING ZONE ==========
    # Draw semi-transparent overlay for ROI
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, 
                 (ROI_LEFT, ROI_TOP), 
                 (ROI_RIGHT, ROI_BOTTOM), 
                 (0, 255, 0), -1)  # Filled rectangle
    cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
    
    # Draw ROI border (thick green rectangle)
    cv2.rectangle(annotated_frame, 
                 (ROI_LEFT, ROI_TOP), 
                 (ROI_RIGHT, ROI_BOTTOM), 
                 (0, 255, 0), 4)
    
    # ========== DISPLAY ONLY TOTAL COUNT ==========
    # Total count - Large and prominent with semi-transparent background
    total_count = len(counted_objects)
    
    # Create small background for count only
    count_text = f"Total Counted: {total_count}"
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    
    # Draw semi-transparent black background
    padding = 10
    info_overlay = annotated_frame.copy()
    cv2.rectangle(info_overlay, 
                 (20 - padding, 50 - text_size[1] - padding), 
                 (20 + text_size[0] + padding, 50 + padding), 
                 (0, 0, 0), -1)
    cv2.addWeighted(info_overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
    
    # Draw total count text
    cv2.putText(annotated_frame, count_text, 
               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # ========== SAVE FRAME TO OUTPUT VIDEO ==========
    video_writer.write(annotated_frame)
    
    # ========== DISPLAY VIDEO ==========
    cv2.imshow("Roadside Sign Detection & Counting", annotated_frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[User interrupted - Exiting...]")
        break

# ========== CLEANUP ==========
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# ========== FINAL SUMMARY ==========
print("\n" + "="*60)
print("FINAL DETECTION SUMMARY")
print("="*60)
print(f"Total Frames Processed: {frame_count}")
print(f"Total Unique Objects Counted: {total_count}")
print(f"Detection Zone: {SIDE} side")
print(f"\nROI Configuration:")
print(f"  X-axis: {ROI_LEFT} to {ROI_RIGHT} pixels")
print(f"  Y-axis: {ROI_TOP} to {ROI_BOTTOM} pixels")
print(f"  Min Confidence: {MIN_CONFIDENCE_FOR_COUNTING}")
print("\n" + "-"*60)
print("Class-wise Breakdown:")
print("-"*60)

if class_wise_counts:
    for class_name in sorted(class_wise_counts.keys()):
        count = len(class_wise_counts[class_name])
        ids = sorted(list(class_wise_counts[class_name]))
        print(f"  {class_name:25s}: {count:3d} | IDs: {ids}")
else:
    print("  No objects were counted.")

print("="*60)
print(f"\nOutput saved to: {output_path}")
print("="*60)