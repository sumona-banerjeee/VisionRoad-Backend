"""
Configuration constants and environment-driven settings for the YOLO detector.
"""

import os
from concurrent.futures import ThreadPoolExecutor

# ── Model & Inference ─────────────────────────────────────────────────────────
MODEL_PATH = r"models\best-11m.pt"
TRACKER = "botsort.yaml"
CONF_THRESHOLD = 0.50

# ── Performance tuning ────────────────────────────────────────────────────────
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))       # Process every Nth frame
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))     # YOLO inference resolution

# ── Class definitions ─────────────────────────────────────────────────────────
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}
EXCLUDED_CLASSES = {"good_sign_board"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES

# ── Async verification ────────────────────────────────────────────────────────
MAX_VERIFY_CONCURRENT = int(os.getenv("MAX_VL_CONCURRENT", "4"))
VL_TIMEOUT = int(os.getenv("VL_TIMEOUT_SECONDS", "30"))

_async_verify_executor = ThreadPoolExecutor(
    max_workers=MAX_VERIFY_CONCURRENT, thread_name_prefix="verify_async"
)


def get_verify_executor() -> ThreadPoolExecutor:
    """Return the async-verify executor (for lifespan shutdown)."""
    return _async_verify_executor


# ── Adaptive parameters (video-dependent) ─────────────────────────────────────

def get_adaptive_params(video_duration: float, width: int, height: int) -> dict:
    """
    Compute adaptive thresholds derived from video properties.

    Returns a dict with keys:
        detection_time_window, time_threshold,
        high_confidence_threshold, low_confidence_min_frames,
        min_distance_threshold, roi (dict with left/right/top/bottom)
    """
    return {
        "detection_time_window": video_duration * 0.25,
        "time_threshold": video_duration * 0.30,
        "high_confidence_threshold": 0.75,
        "low_confidence_min_frames": 2,
        "min_distance_threshold": 120,
        "roi": {
            "left": 0,
            "right": width,
            "top": int(height * 0.05),
            "bottom": int(height * 0.95),
        },
    }
