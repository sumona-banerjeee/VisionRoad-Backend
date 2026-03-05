"""
SAM3 Helper — Local SAM3 verification callback for YoloDetector.

Provides `process_with_sam3(frame, bbox, predicted_class)` which verifies a
YOLO detection by running facebook/sam3 text-prompted segmentation on the
cropped region. Used as the `verify_fn` for YoloDetector in 'sam3' mode.

The Sam3Verifier is a **lazy singleton** — it is only loaded when the sam3 mode
is first used, so yolo / yolo_vl modes pay zero startup cost.

Return contract (same shape as vl_helper so YoloDetector handles it identically):
    {
        "category":            str,   # predicted_class if agrees, else "null"
        "confidence":          str,   # "high" / "medium" / "low"
        "belongs_to_category": bool,
        "_vl_elapsed_s":       float, # used by YoloDetector for perf tracking
        "sam3_score":          float, # raw SAM3 mask score
        "sam3_prompt":         str,   # which prompt matched
    }
"""

import logging
import os
import time
import threading

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image
from transformers import Sam3Model, Sam3Processor

load_dotenv()

logger = logging.getLogger(__name__)

# ── SAM3 Configuration ────────────────────────────────────────────────────────
SAM3_MODEL_ID = "facebook/sam3"
SAM3_VERIFY_THRESHOLD = float(os.getenv("SAM3_VERIFY_THRESHOLD", "0.55"))
SAM3_MIN_BBOX_SIZE = int(os.getenv("SAM3_MIN_BBOX_SIZE", "30"))
SAM3_CROP_MAX_SIZE = 512  # resize crop's largest edge to this before SAM3
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Per-class thresholds — geometrically subtle classes get a lower bar
SAM3_CLASS_THRESHOLDS = {
    "pothole": float(os.getenv("SAM3_THR_POTHOLE", "0.55")),
    "road_crack": float(os.getenv("SAM3_THR_ROAD_CRACK", "0.45")),
    "defected_sign_board": float(os.getenv("SAM3_THR_DEFECTED_SIGN", "0.40")),
    "good_sign_board": float(os.getenv("SAM3_THR_GOOD_SIGN", "0.55")),
    "damaged_road_marking": float(os.getenv("SAM3_THR_DAMAGED_MARKING", "0.55")),
}

# YOLO class → SAM3 text prompts (try all, keep best score)
YOLO_TO_SAM3_PROMPTS = {
    "pothole": ["pothole", "road puddle"],
    "road_crack": ["road crack", "pavement crack"],
    "defected_sign_board": ["broken traffic signboard", "discolored traffic signboard"],
    "good_sign_board": ["traffic signboard"],
    "damaged_road_marking": ["damaged road marking", "faded road marking"],
}


# Score → VL-compatible confidence bucket
def _score_to_confidence(score: float, threshold: float) -> str:
    ratio = score / max(threshold, 1e-6)
    if ratio >= 1.5:
        return "high"
    if ratio >= 1.1:
        return "medium"
    return "low"


# ── Lazy singleton ────────────────────────────────────────────────────────────
_verifier_instance: "Sam3Verifier | None" = None
_verifier_lock = threading.Lock()


def _get_verifier() -> "Sam3Verifier":
    """Return the singleton Sam3Verifier, creating it on first call."""
    global _verifier_instance
    if _verifier_instance is None:
        with _verifier_lock:
            if _verifier_instance is None:  # double-checked
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _verifier_instance = Sam3Verifier(device=device)
    return _verifier_instance


# ── Sam3Verifier ─────────────────────────────────────────────────────────────
class Sam3Verifier:
    """
    Wraps facebook/sam3 for crop-level verification.

    Thread-safe after __init__ (stateless per verify call).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.info(f"⏳ Loading SAM3 ({SAM3_MODEL_ID}) onto {device} …")
        t0 = time.time()

        if HF_TOKEN and "PASTE" not in HF_TOKEN:
            try:
                login(token=HF_TOKEN, add_to_git_credential=False)
            except Exception as e:
                logger.warning(f"HF login warning: {e}")

        self.model = Sam3Model.from_pretrained(
            SAM3_MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)
        self.model.eval()

        self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
        logger.info(f"✅ SAM3 loaded in {time.time() - t0:.1f}s")
        self._warmup()

    def _warmup(self):
        """Prime CUDA kernels — eliminates the first-call 3-4s JIT spike."""
        logger.info("SAM3 warmup — priming CUDA kernels …")
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.verify(dummy, "pothole")
            logger.info("SAM3 warmup done.")
        except Exception as e:
            logger.warning(f"SAM3 warmup failed (non-fatal): {e}")

    @torch.inference_mode()
    def verify(self, crop_bgr: np.ndarray, yolo_class: str) -> dict:
        """
        Run SAM3 text-prompted segmentation on the crop.

        Returns:
            {
                "sam3_agrees":  bool,
                "sam3_score":   float,
                "sam3_prompt":  str,
                "sam3_elapsed": float,
            }
        """
        t0 = time.time()
        prompts = YOLO_TO_SAM3_PROMPTS.get(yolo_class, [yolo_class])
        threshold = SAM3_CLASS_THRESHOLDS.get(yolo_class, SAM3_VERIFY_THRESHOLD)

        # Convert BGR crop → PIL RGB, resize so largest edge ≤ SAM3_CROP_MAX_SIZE
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        pil_img.thumbnail((SAM3_CROP_MAX_SIZE, SAM3_CROP_MAX_SIZE))

        best_score = 0.0
        best_prompt = "none"

        for prompt_text in prompts:
            try:
                inputs = self.processor(
                    images=[pil_img],
                    text=[prompt_text],
                    return_tensors="pt",
                )
                inputs = {
                    k: (
                        v.to(self.device, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in inputs.items()
                }
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(**inputs)

                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=threshold * 0.75,  # more lenient at post-process step
                    target_sizes=[(pil_img.height, pil_img.width)],
                )

                if results and len(results[0]["masks"]) > 0:
                    scores = results[0]["scores"].float().cpu().numpy()
                    top_score = float(scores.max())
                    if top_score > best_score:
                        best_score = top_score
                        best_prompt = prompt_text

                del outputs, inputs

            except Exception as e:
                logger.warning(f"SAM3 verify error prompt='{prompt_text}': {e}")

        agrees = best_score >= threshold
        elapsed = time.time() - t0

        icon = "✓" if agrees else "✗"
        logger.info(
            f"SAM3 {icon} [{elapsed:.2f}s] class={yolo_class!r} "
            f"prompt={best_prompt!r} score={best_score:.3f} threshold={threshold}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                f"CUDA cache cleared ONCE at end of verify() "
                f"(class={yolo_class!r}, prompts_tested={len(prompts)})"
            )

        return {
            "sam3_agrees": agrees,
            "sam3_score": round(best_score, 4),
            "sam3_prompt": best_prompt,
            "sam3_elapsed": round(elapsed, 3),
        }


# ── Public verify_fn (injected into YoloDetector) ────────────────────────────
def process_with_sam3(frame: np.ndarray, bbox: tuple, predicted_class: str):
    """
    Verify a YOLO detection using SAM3 local segmentation.

    Signature matches process_with_vl so YoloDetector handles it identically.

    Args:
        frame:           Full video frame (BGR numpy array)
        bbox:            (x1, y1, x2, y2) detection bounding box
        predicted_class: YOLO class name string

    Returns:
        VL-compatible dict, or None if bbox too small / error
    """
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    if bbox_w < SAM3_MIN_BBOX_SIZE or bbox_h < SAM3_MIN_BBOX_SIZE:
        logger.debug(f"SAM3 skipped — bbox too small: {bbox_w}x{bbox_h}px")
        return None

    try:
        verifier = _get_verifier()
        crop = frame[y1:y2, x1:x2].copy()
        result = verifier.verify(crop, predicted_class)

        agrees = result["sam3_agrees"]
        score = result["sam3_score"]
        threshold = SAM3_CLASS_THRESHOLDS.get(predicted_class, SAM3_VERIFY_THRESHOLD)

        return {
            # VL-compatible fields (read by YoloDetector._process_completed_verify_futures)
            "category": predicted_class if agrees else "null",
            "confidence": _score_to_confidence(score, threshold) if agrees else "low",
            "belongs_to_category": agrees,
            "_vl_elapsed_s": result["sam3_elapsed"],  # perf tracking key used by caller
            # SAM3-specific fields (stored in confirmed dict → appear in JSON report)
            "sam3_score": score,
            "sam3_prompt": result["sam3_prompt"],
        }

    except Exception as e:
        logger.error(f"process_with_sam3 error for class={predicted_class!r}: {e}")
        return None
