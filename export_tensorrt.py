# export_tensorrt.py - Export YOLO model to TensorRT format
import torch
import sys

# Monkey-patch torch.load to disable weights_only by default
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # Force weights_only=False unless explicitly set
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Now load and export
from ultralytics import YOLO

print("Loading model...")
model = YOLO("models/pothole-detector.pt")

print("Exporting to TensorRT engine format...")
model.export(format="engine")

print("Export complete!")
