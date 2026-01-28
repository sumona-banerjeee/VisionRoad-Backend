# export_tensorrt_fixed.py - Fix TensorRT export to match PT model accuracy
# This addresses the 32 → 12 detection drop issue

import os
import torch
from ultralytics import YOLO
import cv2

print("=" * 80)
print("FIXING TENSORRT EXPORT - MATCHING PT MODEL ACCURACY")
print("=" * 80)

# ============================================================================
# STEP 1: Analyze PT model configuration
# ============================================================================
print("\n[1/6] Analyzing PT model configuration...")
pt_model = YOLO("models/pothole-detector.pt")

print(f"✓ PT model loaded")
print(f"  - Task: {pt_model.task}")
print(f"  - Names: {pt_model.names}")

# Get model input size
if hasattr(pt_model.model, 'yaml'):
    print(f"  - Model YAML config: {pt_model.model.yaml}")

# Check overrides
if hasattr(pt_model, 'overrides'):
    print(f"  - Overrides: {pt_model.overrides}")
    imgsz = pt_model.overrides.get('imgsz', 640)
else:
    imgsz = 640

print(f"  - Detected image size: {imgsz}")

# ============================================================================
# STEP 2: Test PT model on a single frame to establish baseline
# ============================================================================
print("\n[2/6] Testing PT model on sample frame...")

# Open video and extract first frame
video_path = "uploads/b01308b9-c4b9-4aca-b92f-e1bc03fded5d.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read video frame")
    exit(1)

# Save test frame
test_frame_path = "test_frame_export.jpg"
cv2.imwrite(test_frame_path, frame)
print(f"✓ Test frame saved: {test_frame_path}")

# Test PT model with EXACT same parameters as your test
pt_model_gpu = YOLO("models/pothole-detector.pt")
pt_model_gpu.to("cuda:0")

results_pt = pt_model_gpu.predict(
    source=test_frame_path,
    conf=0.8,  # Your 80% threshold
    imgsz=imgsz,
    device=0,
    verbose=False
)

pt_detections = len(results_pt[0].boxes)
print(f"✓ PT model baseline: {pt_detections} detections on test frame")

# ============================================================================
# STEP 3: Export to TensorRT with EXACT matching parameters
# ============================================================================
print("\n[3/6] Exporting to TensorRT with matching parameters...")
print("This will take 3-5 minutes...")
print("-" * 80)

try:
    result = pt_model.export(
        format="engine",
        device=0,
        half=False,          # FP32 - no quantization
        imgsz=imgsz,         # MATCH PT model
        batch=1,             # Single image
        workspace=4,         # 4GB workspace
        simplify=True,       # Simplify ONNX
        dynamic=False,       # Fixed input size
        verbose=True
    )
    print("-" * 80)
    print("✓ Export completed!")
    
except Exception as e:
    print(f"❌ Export failed: {str(e)}")
    exit(1)

engine_path = str(result) if result else "models/pothole-detector.engine"

# ============================================================================
# STEP 4: Verify engine file
# ============================================================================
print("\n[4/6] Verifying engine file...")

if os.path.exists(engine_path):
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"✓ Engine created: {engine_path}")
    print(f"  - Size: {size_mb:.2f} MB")
else:
    print(f"❌ Engine not found at: {engine_path}")
    exit(1)

# ============================================================================
# STEP 5: Test engine on same frame
# ============================================================================
print("\n[5/6] Testing TensorRT engine on same frame...")

engine_model = YOLO(engine_path)

results_engine = engine_model.predict(
    source=test_frame_path,
    conf=0.8,  # Same 80% threshold
    imgsz=imgsz,  # MUST match
    device=0,
    verbose=False
)

engine_detections = len(results_engine[0].boxes)
print(f"✓ Engine detections: {engine_detections} on test frame")

# ============================================================================
# STEP 6: Compare results
# ============================================================================
print("\n[6/6] Comparing PT vs Engine...")
print("=" * 80)

accuracy_match = abs(pt_detections - engine_detections) <= 1  # Allow 1 detection difference

if accuracy_match:
    print("✅ SUCCESS! Detection counts match (±1)")
    print(f"   PT model:  {pt_detections} detections")
    print(f"   Engine:    {engine_detections} detections")
    print(f"   Difference: {abs(pt_detections - engine_detections)}")
else:
    print("⚠️  WARNING! Detection counts differ significantly")
    print(f"   PT model:  {pt_detections} detections")
    print(f"   Engine:    {engine_detections} detections")
    print(f"   Difference: {abs(pt_detections - engine_detections)}")
    print("\nPossible issues:")
    print("   1. Model tracking state not preserved in engine")
    print("   2. Different NMS/IOU settings during export")
    print("   3. Precision loss in conversion")

print("=" * 80)

# ============================================================================
# DETAILED COMPARISON
# ============================================================================
print("\nDetailed comparison:")
print("-" * 80)

print("\nPT Model boxes:")
for i, box in enumerate(results_pt[0].boxes):
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    print(f"  Box {i+1}: class={cls}, conf={conf:.3f}")

print("\nEngine boxes:")
for i, box in enumerate(results_engine[0].boxes):
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    print(f"  Box {i+1}: class={cls}, conf={conf:.3f}")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if accuracy_match:
    print("✓ Engine export successful!")
    print("  Next: Run full video test with:")
    print("  python test_detection.py engine gpu")
    print("\n  Expected: Should detect ~32 unique potholes (matching PT model)")
else:
    print("⚠️  Engine accuracy issue detected!")
    print("\nTroubleshooting steps:")
    print("1. Check if your test_detection.py uses .track() instead of .predict()")
    print("   - TensorRT engines may not preserve tracking state")
    print("   - Try using .predict() instead of .track()")
    print("\n2. If still failing, try FP16 export:")
    print("   model.export(format='engine', half=True, ...)")
    print("\n3. Verify CUDA/TensorRT versions:")
    print("   pip list | grep tensorrt")
    print("   nvidia-smi")

print("=" * 80)

# Clean up
os.remove(test_frame_path)
print(f"\n✓ Cleaned up test frame")