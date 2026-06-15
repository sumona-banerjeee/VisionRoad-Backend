"""
Detector Registry

Maps DetectionMode enum values to factory functions that produce detector instances.

  - yolo              : Pure YOLO inference (no verification)
  - yolo_vl           : YOLO + VL verification
  - sam3              : YOLO + SAM3 verification
  - yoloe             : YOLOE open-vocabulary
  - yoloe_trained_vl  : YOLOE trained + VL
  - culvert_detection : Culvert-specific model (culvert_best.pt) + BotSORT,
                        no VL/SAM3 verification, dedup via track IDs only.

To add a new detection mode:
  1. Create a new detector in app/detectors/<name>/detector.py
  2. Register a factory lambda here.
"""

from app.detectors.yolo.detector import YoloDetector
from app.detectors.yoloe.detector import YoloeDetector
from app.helpers.vl_helper import process_with_vl
from app.helpers.sam3_helper import process_with_sam3
from app.detectors.yoloe_trained_vl.detector import YoloeTrainedVlDetector
from app.detectors.culvert.detector import CulvertDetector
from app.detectors.combined.detector import CombinedDetector
from app.detectors.hf_road_damage.detector import HfRoadDamageDetector

def _make_gemini_detector():
    """Lazy import so the server boots even when google-genai is not installed."""
    from app.detectors.gemini.detector import GeminiVideoDetector
    return GeminiVideoDetector()


DETECTOR_REGISTRY = {
    "yolo":              lambda: YoloDetector(detection_mode="yolo"),
    "yolo_vl":           lambda: YoloDetector(verify_fn=process_with_vl, detection_mode="yolo_vl"),
    "sam3":              lambda: YoloDetector(verify_fn=process_with_sam3, detection_mode="sam3"),
    "yoloe":             lambda: YoloeDetector(),
    "yoloe_trained_vl":  lambda: YoloeTrainedVlDetector(),
    "culvert_detection": lambda: CulvertDetector(detection_mode="culvert_detection"),
    "combined":          lambda: CombinedDetector(),
    "hf_road_damage":    lambda: HfRoadDamageDetector(),
    "gemini_video":      _make_gemini_detector,
}
