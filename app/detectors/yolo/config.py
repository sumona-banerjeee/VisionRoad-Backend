"""
YOLO detector configuration — all constants and env-based settings.
"""

import os
import torch

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/final-v1.pt"
TRACKER = "botsort.yaml"
CONF_THRESHOLD = 0.50

# ── Performance tuning ───────────────────────────────────────────────────────
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))  # Process every Nth frame (1 = no skip)
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))  # YOLO inference resolution
USE_HALF = torch.cuda.is_available()  # FP16 on GPU, FP32 on CPU

# ── Road damage classes ──────────────────────────────────────────────────────
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}
EXCLUDED_CLASSES = {"good_sign_board"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES

# ── Async verification ───────────────────────────────────────────────────────
MAX_VERIFY_CONCURRENT = int(os.getenv("MAX_VL_CONCURRENT", "4"))
