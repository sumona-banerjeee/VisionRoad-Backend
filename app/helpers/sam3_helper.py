"""
SAM3 Helper — Standalone SAM3 processing function.

Provides `process_with_sam3()` which applies SAM3-based processing to
detection results. Currently a stub — not yet implemented.
"""

import logging

logger = logging.getLogger(__name__)


def process_with_sam3(frame, bbox, predicted_class):
    """
    Process a detection using SAM3 model.

    Args:
        frame: Original video frame (numpy array)
        bbox: Tuple (x1, y1, x2, y2) of bounding box
        predicted_class: YOLO's predicted class name

    Returns:
        dict with processing results or None

    Raises:
        NotImplementedError: SAM3 processing is not yet implemented.
    """
    logger.warning("SAM3 helper called — this function is not yet implemented.")
    raise NotImplementedError("SAM3 processing is not yet implemented.")
