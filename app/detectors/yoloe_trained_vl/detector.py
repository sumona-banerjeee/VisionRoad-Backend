"""
YoloeTrainedVlDetector — Fine-tuned YOLOE detection with selective VL verification.

Uses the fine-tuned YOLOE model (models/yoloe_best.pt) trained on 5 classes:
  defected_sign_board, good_sign_board, pothole, road_crack, damaged_road_marking

Detection flow:
  Image → YOLOE Detection → Is class in VL_SKIP_CLASSES?
                              ├─ Yes → Keep detection directly
                              └─ No  → VL verification
                                        ├─ True  → Keep
                                        └─ False → Reject

This detector reuses the YoloDetector from app/detectors/yolo/detector.py
by subclassing it and overriding only the model path, confidence threshold,
and verify function. All tracking, deduplication, GPS, DB, and progress
logic is inherited from YoloDetector.
"""

from app.detectors.yolo.detector import YoloDetector
from app.helpers.yoloe_vl_trained_helper import (
    YOLOE_TRAINED_MODEL_PATH,
    YOLOE_TRAINED_CONF,
    process_with_trained_vl,
)

import logging

logger = logging.getLogger(__name__)


class YoloeTrainedVlDetector(YoloDetector):
    """
    Fine-tuned YOLOE detector with selective VL verification.

    Inherits the full YoloDetector pipeline (BoTSORT tracking, multi-frame
    confirmation, spatial deduplication, async VL verification, GPS, DB save,
    NDJSON streaming, perf reporting).

    Overrides:
      - Model path → models/yoloe_best.pt
      - Confidence → 0.70
      - Verify callback → process_with_trained_vl (skips VL for signboards)
    """

    def __init__(self):
        # Initialize YoloDetector with our trained model and VL callback
        super().__init__(
            verify_fn=process_with_trained_vl,
            detection_mode="yoloe_trained_vl",
            conf_threshold=YOLOE_TRAINED_CONF,
            apply_roi=False,
        )
        logger.info(
            f"YoloeTrainedVlDetector ready — model={YOLOE_TRAINED_MODEL_PATH}, "
            f"conf={YOLOE_TRAINED_CONF}"
        )

    def _load_model(self):
        """Override to load the fine-tuned YOLOE model instead of the default YOLO model."""
        try:
            from app.helpers.yoloe_vl_trained_helper import load_yoloe_trained_model
            self.model = load_yoloe_trained_model()
            logger.info(f"YoloeTrainedVlDetector model loaded: {YOLOE_TRAINED_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load trained YOLOE model: {e}")
            raise
