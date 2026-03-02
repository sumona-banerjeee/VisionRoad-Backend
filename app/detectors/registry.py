"""
Detector Registry

Maps DetectionMode enum values to factory functions that produce detector instances.

To add a new detection mode:
  1. Create a new sub-package under app/detectors/
  2. Implement a detector class there
  3. Add a single entry here in DETECTOR_REGISTRY

No changes to the API layer, upload_service, or routes are needed.
"""

from app.detectors.yolo_vl.detector import YoloVLDetector
from app.detectors.sam3.detector import Sam3Detector

# YOLO-only reuses the same detector as YOLO+VL but with VL forced off.
# This avoids duplicate video-processing code while keeping modes clearly separate.
DETECTOR_REGISTRY = {
    "yolo": lambda: YoloVLDetector(enable_vl=False),
    "yolo_vl": lambda: YoloVLDetector(enable_vl=True),
    "sam3": lambda: Sam3Detector(),
}
