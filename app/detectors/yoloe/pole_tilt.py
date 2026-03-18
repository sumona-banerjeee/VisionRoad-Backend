"""
Pole Tilt Analysis — Estimate whether a signboard pole is upright or bent.

Given a signboard bounding box (and optionally a segmentation mask),
this module extracts the pole region (lower portion of the bbox) and
estimates the tilt angle of the pole relative to vertical.

Two estimation methods:
  • Primary:  Segmentation mask → cv2.fitLine on pole-region mask pixels
  • Fallback: Canny edges → cv2.HoughLinesP → most-vertical line

Classification:
  • tilt < 15°  → "GOOD SIGNBOARD"
  • tilt ≥ 15°  → "BENT POLE"

Entry point: analyze_pole_tilt(frame, box, mask=None)
"""

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
POLE_REGION_TOP_RATIO = 0.40     # pole region starts at 40% down the bbox
POLE_REGION_BOTTOM_RATIO = 1.0   # pole region ends at 100% (bottom of bbox)
TILT_THRESHOLD_DEGREES = 15.0    # below this = upright, at or above = bent
MIN_MASK_PIXELS = 20             # min mask pixels required for fitLine
MIN_LINE_LENGTH = 15             # HoughLinesP minimum line length (px)
MAX_LINE_GAP = 10                # HoughLinesP maximum gap between segments
CANNY_LOW = 50                   # Canny lower threshold
CANNY_HIGH = 150                 # Canny upper threshold


# ══════════════════════════════════════════════════════════════════════════════
# MODULAR FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def extract_pole_region(frame: np.ndarray, box: tuple) -> np.ndarray | None:
    """
    Extract the pole region from the frame using the lower portion of the bbox.

    The pole is typically below the sign plate, so we crop the lower 40–60%
    of the bounding box.

    Args:
        frame: BGR video frame (H×W×3).
        box:   Bounding box as (x1, y1, x2, y2).

    Returns:
        Cropped pole region as BGR array, or None if the crop is too small.
    """
    x1, y1, x2, y2 = map(int, box)
    box_h = y2 - y1
    if box_h < 10:
        return None

    # Pole region: lower portion of the bounding box
    pole_y1 = y1 + int(box_h * POLE_REGION_TOP_RATIO)
    pole_y2 = y1 + int(box_h * POLE_REGION_BOTTOM_RATIO)

    # Clip to frame bounds
    h, w = frame.shape[:2]
    pole_y1 = max(0, min(pole_y1, h))
    pole_y2 = max(0, min(pole_y2, h))
    cx1 = max(0, min(x1, w))
    cx2 = max(0, min(x2, w))

    if pole_y2 - pole_y1 < 5 or cx2 - cx1 < 3:
        return None

    return frame[pole_y1:pole_y2, cx1:cx2].copy()


def compute_tilt_angle(vx: float, vy: float) -> float:
    """
    Convert a line direction vector (vx, vy) to angle from vertical (degrees).

    A perfectly vertical line has vx=0, vy=1 → 0°.
    A 45° tilt returns 45°.

    Args:
        vx: X component of the direction vector.
        vy: Y component of the direction vector.

    Returns:
        Angle from vertical in degrees [0, 90].
    """
    # Angle of line relative to vertical (Y-axis)
    # vertical = (0, 1), angle = arctan(|vx| / |vy|)
    if abs(vy) < 1e-9:
        return 90.0  # horizontal line
    angle_rad = math.atan2(abs(vx), abs(vy))
    return math.degrees(angle_rad)


def tilt_from_mask(mask: np.ndarray, box: tuple) -> float | None:
    """
    Estimate pole tilt using the segmentation mask.

    Extracts mask pixels in the pole region of the bbox, fits a line
    through them with cv2.fitLine, and computes the angle from vertical.

    Args:
        mask: Binary segmentation mask (H×W), same size as the original frame.
              Non-zero pixels belong to the detected object.
        box:  Bounding box as (x1, y1, x2, y2).

    Returns:
        Tilt angle in degrees, or None if the mask is too sparse.
    """
    x1, y1, x2, y2 = map(int, box)
    box_h = y2 - y1
    if box_h < 10:
        return None

    # Define the pole sub-region within the mask
    pole_y1 = y1 + int(box_h * POLE_REGION_TOP_RATIO)
    pole_y2 = y1 + int(box_h * POLE_REGION_BOTTOM_RATIO)

    # Clip to mask bounds
    h, w = mask.shape[:2]
    pole_y1 = max(0, min(pole_y1, h))
    pole_y2 = max(0, min(pole_y2, h))
    cx1 = max(0, min(x1, w))
    cx2 = max(0, min(x2, w))

    # Extract pole-region mask
    pole_mask = mask[pole_y1:pole_y2, cx1:cx2]
    if pole_mask.size == 0:
        return None

    # Get coordinates of non-zero (object) pixels
    ys, xs = np.nonzero(pole_mask)
    if len(xs) < MIN_MASK_PIXELS:
        logger.debug(
            f"tilt_from_mask: only {len(xs)} mask pixels "
            f"(need {MIN_MASK_PIXELS}), skipping"
        )
        return None

    # Fit a line through the mask points
    points = np.column_stack((xs, ys)).astype(np.float32)
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(line[0]), float(line[1])

    angle = compute_tilt_angle(vx, vy)
    logger.debug(
        f"tilt_from_mask: vx={vx:.4f} vy={vy:.4f} → "
        f"angle={angle:.1f}° ({len(xs)} mask px)"
    )
    return angle


def tilt_from_edges(pole_crop: np.ndarray) -> float | None:
    """
    Estimate pole tilt using edge detection (fallback method).

    Converts the pole crop to grayscale, applies Canny edge detection,
    finds straight lines via HoughLinesP, selects the most vertical line,
    and computes its angle from vertical.

    Args:
        pole_crop: BGR image of the pole region.

    Returns:
        Tilt angle in degrees, or None if no suitable line is found.
    """
    if pole_crop is None or pole_crop.size == 0:
        return None

    gray = cv2.cvtColor(pole_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    if lines is None or len(lines) == 0:
        logger.debug("tilt_from_edges: no lines detected by HoughLinesP")
        return None

    # Find the most vertical line (smallest angle from vertical)
    best_angle = None
    for line_seg in lines:
        lx1, ly1, lx2, ly2 = line_seg[0]
        dx = float(lx2 - lx1)
        dy = float(ly2 - ly1)

        # Skip near-zero-length segments
        length = math.sqrt(dx * dx + dy * dy)
        if length < 5:
            continue

        angle = compute_tilt_angle(dx, dy)

        # Keep the most vertical line (smallest angle from vertical)
        if best_angle is None or angle < best_angle:
            best_angle = angle

    if best_angle is not None:
        logger.debug(f"tilt_from_edges: best line angle = {best_angle:.1f}°")

    return best_angle


def classify_pole(tilt_angle: float) -> str:
    """
    Classify the pole condition based on tilt angle.

    Args:
        tilt_angle: Angle from vertical in degrees.

    Returns:
        "GOOD SIGNBOARD" if tilt < 15°, "BENT POLE" otherwise.
    """
    if tilt_angle < TILT_THRESHOLD_DEGREES:
        return "GOOD SIGNBOARD"
    return "BENT POLE"


def draw_detection(
    frame: np.ndarray,
    box: tuple,
    label: str,
    tilt: float,
) -> np.ndarray:
    """
    Draw bounding box and tilt annotation on the frame.

    Args:
        frame: BGR frame to annotate (modified in-place and returned).
        box:   Bounding box as (x1, y1, x2, y2).
        label: Classification label ("GOOD SIGNBOARD" / "BENT POLE").
        tilt:  Tilt angle in degrees.

    Returns:
        The annotated frame.
    """
    x1, y1, x2, y2 = map(int, box)

    # Green for good, red for bent
    if label == "GOOD SIGNBOARD":
        color = (0, 200, 0)       # green
    else:
        color = (0, 0, 220)       # red

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Build display text
    text = f"{label} | Tilt: {tilt:.1f}deg"

    # Text background for readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position label above the box
    text_y = max(y1 - 8, th + 4)
    cv2.rectangle(
        frame,
        (x1, text_y - th - 4),
        (x1 + tw + 4, text_y + baseline),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 2, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_pole_tilt(
    frame: np.ndarray,
    box: tuple,
    mask: np.ndarray | None = None,
) -> tuple[float, str]:
    """
    Analyze pole tilt for a detected signboard.

    Tries the segmentation-mask method first (more accurate).
    Falls back to edge-based detection if the mask is unavailable or
    has too few pixels.

    Args:
        frame: BGR video frame.
        box:   Bounding box (x1, y1, x2, y2) of the detected signboard.
        mask:  Optional binary segmentation mask (same size as frame).

    Returns:
        (tilt_angle, classification) tuple:
            tilt_angle:     float, degrees from vertical (0 = perfectly upright)
            classification: "GOOD SIGNBOARD" or "BENT POLE"
    """
    tilt_angle = None

    # ── Primary: mask-based tilt estimation ────────────────────────────────
    if mask is not None:
        try:
            tilt_angle = tilt_from_mask(mask, box)
            if tilt_angle is not None:
                logger.debug(
                    f"Pole tilt (mask method): {tilt_angle:.1f}°"
                )
        except Exception as e:
            logger.warning(f"tilt_from_mask failed: {e}")
            tilt_angle = None

    # ── Fallback: edge-based tilt estimation ──────────────────────────────
    if tilt_angle is None:
        try:
            pole_crop = extract_pole_region(frame, box)
            tilt_angle = tilt_from_edges(pole_crop)
            if tilt_angle is not None:
                logger.debug(
                    f"Pole tilt (edge method): {tilt_angle:.1f}°"
                )
        except Exception as e:
            logger.warning(f"tilt_from_edges failed: {e}")
            tilt_angle = None

    # ── Default if both methods fail ──────────────────────────────────────
    if tilt_angle is None:
        logger.debug("Both tilt methods failed — defaulting to 0° (upright)")
        tilt_angle = 0.0

    classification = classify_pole(tilt_angle)
    return tilt_angle, classification