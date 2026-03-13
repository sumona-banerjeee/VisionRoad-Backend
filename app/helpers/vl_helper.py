"""
VL (Vision-Language) Helper — Standalone verification function.

Provides `process_with_vl()` which verifies a YOLO detection using the
Ollama VL API. API keys are loaded once at module import time and rotated
across calls in a thread-safe manner.
"""

import cv2
import json
import base64
import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import ollama

logger = logging.getLogger(__name__)

# VL Configuration
VL_MODEL = "qwen3-vl:235b-instruct-cloud"
VL_IMAGE_MAX_SIZE = 512  # Max dimension for VL input images
VL_MIN_BBOX_SIZE = 30  # Skip VL for bboxes smaller than this
VL_TIMEOUT = int(os.getenv("VL_TIMEOUT_SECONDS", "30"))  # Hard timeout
# Bboxes with BOTH width AND height >= this are considered "large enough" and
# are sent as tight crops (they already have sufficient detail for VL).
# Smaller bboxes get 2× padding so VL can see environmental context.
VL_CONTEXT_BBOX_THRESHOLD = int(os.getenv("VL_CONTEXT_BBOX_THRESHOLD", "150"))

# Thread pool for timeout enforcement inside VL calls
_vl_timeout_executor = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="vl_timeout"
)

# ── API key management (module-level, loaded once) ──────────────────────────
_api_keys = []
_current_key_index = 0
_key_lock = threading.Lock()


def _load_api_keys():
    """Load all available API keys from environment variables."""
    api_keys = []
    i = 1

    # Try to load numbered API keys (OLLAMA_API_KEY_1, OLLAMA_API_KEY_2, etc.)
    while True:
        key = os.getenv(f"OLLAMA_API_KEY_{i}")
        if not key:
            break
        api_keys.append(key)
        i += 1

    # Fallback to single key if numbered keys not found
    if not api_keys:
        single_key = os.getenv("OLLAMA_API_KEY")
        if single_key:
            api_keys.append(single_key)

    if api_keys:
        logger.info(f"VL Helper: Loaded {len(api_keys)} API key(s) for VL verification")
    else:
        logger.warning(
            "VL Helper: No API keys found. VL verification will be disabled."
        )

    return api_keys


def _get_next_api_key():
    """Get next API key in rotation (thread-safe). Returns (key, key_index) tuple."""
    global _current_key_index
    if not _api_keys:
        return None, None

    with _key_lock:
        index = _current_key_index
        key = _api_keys[index]
        _current_key_index = (_current_key_index + 1) % len(_api_keys)
        return key, index + 1  # 1-based index for human-readable logs


def _create_ollama_client(api_key):
    """Create Ollama client with given API key."""
    try:
        client = ollama.Client(
            host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"}
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create Ollama client: {e}")
        return None


# Load API keys on module import
_api_keys = _load_api_keys()


def process_with_vl(frame, bbox, predicted_class):
    """
    Verify a detection using VL model.

    Args:
        frame: Original video frame (numpy array)
        bbox: Tuple (x1, y1, x2, y2) of bounding box
        predicted_class: YOLO's predicted class name

    Returns:
        dict with verification results or None if verification fails/skipped
    """
    if not _api_keys:
        return None

    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Skip if bbox is too small
    if bbox_width < VL_MIN_BBOX_SIZE or bbox_height < VL_MIN_BBOX_SIZE:
        logger.debug(f"Skipping VL for small bbox: {bbox_width}x{bbox_height}")
        return None

    try:
        # Get next API key in rotation
        api_key, key_index = _get_next_api_key()
        if not api_key:
            return None

        # Log which key slot is being used (mask all but last 4 chars)
        logger.info(
            f"VL call using API key #{key_index}/{len(_api_keys)} "
            f"(key: ...{api_key[-4:]})"
        )

        # ── Adaptive crop: tight vs. context-padded ────────────────────────
        # Large bbox (≥ threshold on both sides): the object fills the crop
        # well enough — send the tight bbox as-is (existing behaviour).
        # Small bbox: add 2× padding on each side so VL can see the
        # surrounding environment (sky, road, trees) and judge plausibility.
        frame_h, frame_w = frame.shape[:2]
        large_bbox = (
            bbox_width >= VL_CONTEXT_BBOX_THRESHOLD
            and bbox_height >= VL_CONTEXT_BBOX_THRESHOLD
        )
        if large_bbox:
            cropped = frame[y1:y2, x1:x2]
            with_context = False
        else:
            # Expand by 2× bbox dimensions on each side, clamped to frame edges
            pad_x = bbox_width      # 2× means one full bbox-width on each side
            pad_y = bbox_height
            ctx_x1 = max(0, x1 - pad_x)
            ctx_y1 = max(0, y1 - pad_y)
            ctx_x2 = min(frame_w, x2 + pad_x)
            ctx_y2 = min(frame_h, y2 + pad_y)
            cropped = frame[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
            with_context = True
            logger.debug(
                f"VL context-crop: bbox={bbox_width}x{bbox_height} → "
                f"context patch={ctx_x2-ctx_x1}x{ctx_y2-ctx_y1}"
            )
        # ────────────────────────────────────────────────────────────────────

        # Resize to optimize tokens (maintain aspect ratio)
        crop_h, crop_w = cropped.shape[:2]
        max_dim = max(crop_w, crop_h)
        if max_dim > VL_IMAGE_MAX_SIZE:
            scale = VL_IMAGE_MAX_SIZE / max_dim
            cropped = cv2.resize(
                cropped, (int(crop_w * scale), int(crop_h * scale))
            )

        # Encode to base64
        _, buffer = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_image = base64.b64encode(buffer).decode("utf-8")

        # Optimized prompt for token efficiency.
        # When context padding is included, explicitly tell VL to use the
        # surroundings (sky, road surface, vegetation) in its decision.
        context_hint = (
            "The image shows the detected object WITH surrounding environment context — "
            "use the surroundings (road surface, sky, background) to judge plausibility. "
        ) if with_context else ""
        prompt = (
            "You are a road infrastructure inspector. "
            f"{context_hint}"
            f"YOLO predicted this {'region' if with_context else 'crop'} as: {predicted_class}. Verify if correct.\n"
            "Classify as exactly ONE of:\n"
            "- defected_sign_board (traffic/road regulatory sign: damaged, faded, or vandalized)\n"
            "- good_sign_board (traffic/road regulatory sign: intact and legible)\n"
            "- pothole (road surface hole or depression)\n"
            "- road_crack (crack or fracture in road/pavement surface)\n"
            "- damaged_road_marking (faded or worn lane lines or road paint)\n"
            "Non-traffic signs (shop signs, billboards, banners, posters) → null.\n"
            "If the object appears to be in the sky, on a wall, or otherwise NOT on a road surface, set belongs_to_category=false.\n"
            'JSON only: {"category": "name_or_null", "confidence": "high/medium/low", '
            '"belongs_to_category": true/false}'
        )

        # Create client with rotated API key
        vl_client = _create_ollama_client(api_key)
        if not vl_client:
            return None

        # Hard timeout enforcement via thread
        def _call_vl():
            return vl_client.chat(
                model=VL_MODEL,
                format="json",
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\n\nRespond ONLY with valid JSON.",
                        "images": [base64_image],
                    }
                ],
                options={"temperature": 0.1},
            )

        vl_start = time.time()
        future = _vl_timeout_executor.submit(_call_vl)
        try:
            response = future.result(timeout=VL_TIMEOUT)
        except FuturesTimeoutError:
            vl_elapsed = time.time() - vl_start
            logger.warning(
                f"VL call TIMED OUT after {vl_elapsed:.1f}s for YOLO={predicted_class}. "
                f"Skipping VL — YOLO result will be used as-is."
            )
            return None
        vl_elapsed = time.time() - vl_start

        # Parse response
        vl_result = json.loads(response["message"]["content"])
        vl_result["yolo_prediction"] = predicted_class
        vl_result["_vl_elapsed_s"] = vl_elapsed  # used by caller for perf tracking

        # Log VL result with timing
        vl_cat = vl_result.get("category")
        vl_conf = vl_result.get("confidence")
        match = "✓" if vl_cat == predicted_class else "✗"
        logger.info(
            f"VL call {match} [{vl_elapsed:.2f}s]: YOLO={predicted_class}, VL={vl_cat} "
            f"(confidence={vl_conf}, belongs={vl_result.get('belongs_to_category')})"
        )

        return vl_result

    except json.JSONDecodeError as e:
        logger.error(f"VL JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"VL verification error: {e}")
        return None
