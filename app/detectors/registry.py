"""
Detector Registry

Maps DetectionMode enum values to factory functions that produce detector instances.

The registry now uses a single YoloDetector for all modes. The detection mode
determines which helper function (if any) is injected as a verification callback:

  - yolo    : Pure YOLO inference (no verification)
  - yolo_vl : YOLO inference + VL verification via helpers/vl_helper
  - sam3    : YOLO inference + SAM3 processing via helpers/sam3_helper

To add a new detection mode:
  1. Create a helper function in app/helpers/
  2. Add a single entry here in DETECTOR_REGISTRY with the helper as verify_fn
"""

from app.detectors.yolo.detector import YoloDetector
from app.detectors.yoloe.detector import YoloeDetector
from app.helpers.vl_helper import process_with_vl
from app.helpers.sam3_helper import process_with_sam3

DETECTOR_REGISTRY = {
    "yolo": lambda: YoloDetector(detection_mode="yolo"),
    "yolo_vl": lambda: YoloDetector(verify_fn=process_with_vl, detection_mode="yolo_vl"),
    "sam3": lambda: YoloDetector(verify_fn=process_with_sam3, detection_mode="sam3"),
    "yoloe": lambda: YoloeDetector(),
}
