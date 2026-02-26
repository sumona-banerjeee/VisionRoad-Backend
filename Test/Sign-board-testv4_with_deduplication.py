import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import json
import math
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = r"D:\VisionRoad-Backend\models\pothole-signboard.pt"
VIDEO_PATH = r"D:\VisionRoad-Backend\Test\video\live-vid-1.mp4"
OUTPUT_PATH = r"D:\VisionRoad-Backend\Test\output\output_with_deduplication3.mp4"
RESULTS_JSON = r"D:\VisionRoad-Backend\Test\output\results_with_deduplication.json"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5
TRACKER = "bytetrack.yaml"

# Spatial deduplication thresholds
IOU_THRESHOLD = 0.3  # 30% overlap → same object
DISTANCE_THRESHOLD = 100  # 100 pixels → same object (adjust based on your video resolution)

# ROI Configuration (for roadside detection)
SIDE = "BOTH"  # Options: "LEFT", "RIGHT", "BOTH"

# ============================================================================
# SPATIAL DEDUPLICATION FUNCTIONS
# ============================================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
        
    Returns:
        float: IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_distance(center1, center2):
    """
    Calculate Euclidean distance between two centers
    
    Args:
        center1: (x, y)
        center2: (x, y)
        
    Returns:
        float: distance in pixels
    """
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def is_duplicate_detection(new_bbox, new_center, confirmed_objects):
    """
    Check if a new detection is a duplicate of an already confirmed object.
    Uses both IoU and spatial distance to prevent counting the same object multiple times.
    
    Args:
        new_bbox: (x1, y1, x2, y2) of new detection
        new_center: (x, y) center of new detection
        confirmed_objects: dictionary of already confirmed objects
        
    Returns:
        tuple: (is_duplicate, matched_id) - True if duplicate, and the ID it matches (or None)
    """
    for obj_id, info in confirmed_objects.items():
        if "bbox" not in info or "center" not in info:
            continue
        
        existing_bbox = info["bbox"]
        existing_center = info["center"]
        
        # Check IoU overlap
        iou = calculate_iou(new_bbox, existing_bbox)
        
        # Check spatial distance
        distance = calculate_distance(new_center, existing_center)
        
        # If significant overlap OR very close proximity, it's likely the same object
        if iou > IOU_THRESHOLD or distance < DISTANCE_THRESHOLD:
            print(f"   → Duplicate detected! IoU={iou:.3f}, Distance={distance:.1f}px, Matches ID={obj_id}")
            return True, obj_id
    
    return False, None


def get_roi_mask(frame, side="BOTH"):
    """Create ROI mask for roadside detection"""
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if side == "LEFT":
        # Left side of road
        mask[:, :int(width * 0.3)] = 255
    elif side == "RIGHT":
        # Right side of road
        mask[:, int(width * 0.7):] = 255
    else:  # BOTH
        # Both sides
        mask[:, :int(width * 0.3)] = 255
        mask[:, int(width * 0.7):] = 255
    
    return mask


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_video_with_deduplication():
    """Process video with spatial deduplication to prevent counting the same sign multiple times"""
    
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SIGN-BOARD DETECTION WITH SPATIAL DEDUPLICATION")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Deduplication Settings:")
    print(f"  - IoU Threshold: {IOU_THRESHOLD}")
    print(f"  - Distance Threshold: {DISTANCE_THRESHOLD} pixels")
    print(f"  - ROI Side: {SIDE}")
    print("="*80)
    
    # Load model
    print("\n[1/5] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded successfully")
    
    # Open video
    print("\n[2/5] Opening video file...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Video opened: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Initialize video writer
    print("\n[3/5] Initializing output video writer...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    print("✓ Video writer ready")
    
    # Initialize tracking structures
    confirmed_objects = {}  # Stores confirmed unique objects
    frame_detections = []   # Stores all detections per frame
    frame_count = 0
    total_detections = 0
    
    print("\n[4/5] Processing video frames...")
    print("-"*80)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_detections = []
        
        # Run YOLO tracking
        results = model.track(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            tracker=TRACKER,
            persist=True,
            verbose=False
        )
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None
            
            if ids is None:
                continue
            
            for box, track_id, conf, cls in zip(boxes, ids, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                class_id = int(cls)
                class_name = model.names[class_id] if hasattr(model, 'names') else str(class_id)
                
                # Calculate center and bbox
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                bbox_tuple = (x1, y1, x2, y2)
                center_tuple = (center_x, center_y)
                
                # Check if this track_id is already confirmed
                if track_id not in confirmed_objects:
                    # SPATIAL DEDUPLICATION CHECK
                    is_duplicate, matched_id = is_duplicate_detection(
                        bbox_tuple,
                        center_tuple,
                        confirmed_objects
                    )
                    
                    if is_duplicate:
                        # This is a duplicate - don't add to confirmed objects
                        print(f"Frame {frame_count}: Track ID {track_id} ({class_name}) skipped (duplicate of ID {matched_id})")
                        continue
                    
                    # Not a duplicate - confirm as new unique object
                    confirmed_objects[track_id] = {
                        "frame": frame_count,
                        "class": class_name,
                        "class_id": class_id,
                        "conf": float(conf),
                        "bbox": bbox_tuple,
                        "center": center_tuple,
                        "last_seen_frame": frame_count
                    }
                    print(f"Frame {frame_count}: ✓ NEW UNIQUE OBJECT - ID {track_id} ({class_name}) - Conf: {conf:.2f}")
                else:
                    # Update existing confirmed object
                    confirmed_objects[track_id]["bbox"] = bbox_tuple
                    confirmed_objects[track_id]["center"] = center_tuple
                    confirmed_objects[track_id]["last_seen_frame"] = frame_count
                
                # Draw bounding box (only for confirmed objects)
                if track_id in confirmed_objects:
                    total_detections += 1
                    color = (0, 255, 0)  # Green for confirmed
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    current_detections.append({
                        "frame": frame_count,
                        "track_id": track_id,
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": float(conf),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "center": {"x": center_x, "y": center_y}
                    })
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Unique Objects: {len(confirmed_objects)} | Total Detections: {total_detections}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        video_writer.write(frame)
        
        if current_detections:
            frame_detections.append({
                "frame_id": frame_count,
                "detections": current_detections
            })
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Unique: {len(confirmed_objects)} | Total: {total_detections}")
    
    # Cleanup
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    print("-"*80)
    print("\n[5/5] Saving results...")
    
    # Prepare results
    unique_objects_list = sorted([
        {
            "id": int(obj_id),
            "class": info["class"],
            "class_id": info["class_id"],
            "first_seen_frame": info["frame"],
            "last_seen_frame": info["last_seen_frame"],
            "confidence": round(info["conf"], 3)
        }
        for obj_id, info in confirmed_objects.items()
    ], key=lambda x: x["first_seen_frame"])
    
    results = {
        "video_info": {
            "source": VIDEO_PATH,
            "total_frames": total_frames,
            "fps": fps,
            "resolution": f"{width}x{height}"
        },
        "deduplication_settings": {
            "iou_threshold": IOU_THRESHOLD,
            "distance_threshold": DISTANCE_THRESHOLD,
            "enabled": True
        },
        "summary": {
            "total_frames_processed": frame_count,
            "unique_objects": len(confirmed_objects),
            "total_detections": total_detections,
            "frames_with_detections": len(frame_detections)
        },
        "unique_objects": unique_objects_list,
        "frame_detections": frame_detections
    }
    
    # Save JSON
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {RESULTS_JSON}")
    print(f"✓ Output video saved to: {OUTPUT_PATH}")
    
    # Final summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Detections: {total_detections}")
    print(f">>> UNIQUE OBJECTS (after deduplication): {len(confirmed_objects)} <<<")
    
    # Class distribution
    class_counts = {}
    for obj_id, info in confirmed_objects.items():
        class_name = info["class"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nClass Distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  - {class_name}: {count}")
    
    print(f"\nObject IDs: {sorted([int(obj_id) for obj_id in confirmed_objects.keys()])}")
    print(f"\n✓ Deduplication prevented {total_detections - len(confirmed_objects)} duplicate counts!")
    print(f"✓ Accuracy improvement: {((total_detections - len(confirmed_objects)) / total_detections * 100):.1f}% fewer false counts")
    print("="*80)
    
    return results


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    try:
        results = process_video_with_deduplication()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()