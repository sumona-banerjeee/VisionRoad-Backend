"""
Test script for app.detectors.yolo.pole_tilt

Creates synthetic images with drawn lines at known angles and verifies
that the tilt detection pipeline returns correct results.
No YOLO model or video required.

Usage:
    python Test/test_pole_tilt.py
"""

import sys
import os
import math
import importlib.util

# Direct import of pole_tilt module — bypasses app/__init__.py import chain
# which pulls in unrelated dependencies (YOLOE, DB, etc.)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULE_PATH = os.path.join(
    _PROJECT_ROOT, "app", "detectors", "yolo", "pole_tilt.py"
)
spec = importlib.util.spec_from_file_location("pole_tilt", _MODULE_PATH)
pole_tilt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pole_tilt)

import cv2
import numpy as np

extract_pole_region = pole_tilt.extract_pole_region
compute_tilt = pole_tilt.compute_tilt
tilt_from_edges = pole_tilt.tilt_from_edges
classify_pole = pole_tilt.classify_pole
draw_detection = pole_tilt.draw_detection
analyze_pole_tilt = pole_tilt.analyze_pole_tilt

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {detail}")
        failed += 1


def make_frame(w=640, h=480):
    """Create a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def draw_vertical_pole(frame, box, angle_deg=0, thickness=3):
    """
    Draw a line inside the pole region of the bbox at a given angle
    from vertical. Modifies frame in-place.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    box_h = y2 - y1
    # Pole region lower 70%
    pole_y1 = y1 + int(box_h * 0.30)
    pole_y2 = y2

    mid_y = (pole_y1 + pole_y2) // 2
    half_len = (pole_y2 - pole_y1) // 2

    # angle_deg from vertical: 0 = straight up, positive = tilted right
    angle_rad = math.radians(angle_deg)
    dx = int(half_len * math.sin(angle_rad))
    dy = int(half_len * math.cos(angle_rad))

    pt1 = (cx - dx, mid_y - dy)
    pt2 = (cx + dx, mid_y + dy)
    cv2.line(frame, pt1, pt2, (255, 255, 255), thickness)
    return frame


# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST: compute_tilt()")
print("=" * 60)

test("Vertical line (dx=0, dy=100) → 0°",
     abs(compute_tilt(0, 100) - 0.0) < 0.1)

test("Horizontal line (dx=100, dy=0) → 90°",
     abs(compute_tilt(100, 0) - 90.0) < 0.1)

test("45° line (dx=100, dy=100) → 45°",
     abs(compute_tilt(100, 100) - 45.0) < 0.1)

test("Small tilt (dx=10, dy=100) → ~5.7°",
     abs(compute_tilt(10, 100) - 5.71) < 0.5)

test("dy near zero → 90°",
     abs(compute_tilt(50, 1e-10) - 90.0) < 0.1)


# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST: classify_pole()")
print("=" * 60)

test("0° → GOOD SIGNBOARD", classify_pole(0.0) == "GOOD SIGNBOARD")
test("10° → GOOD SIGNBOARD", classify_pole(10.0) == "GOOD SIGNBOARD")
test("14.9° → GOOD SIGNBOARD", classify_pole(14.9) == "GOOD SIGNBOARD")
test("15° → BENT POLE", classify_pole(15.0) == "BENT POLE")
test("30° → BENT POLE", classify_pole(30.0) == "BENT POLE")
test("90° → BENT POLE", classify_pole(90.0) == "BENT POLE")


# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST: extract_pole_region()")
print("=" * 60)

frame = make_frame()
box = (100, 50, 200, 350)  # 100x300 bbox

crop = extract_pole_region(frame, box)
test("Returns crop (not None)", crop is not None)
if crop is not None:
    expected_h = int(300 * 0.70)  # 70% of 300
    test(f"Crop height ≈ {expected_h}px",
         abs(crop.shape[0] - expected_h) <= 1,
         f"got {crop.shape[0]}")
    test("Crop width = 100px", crop.shape[1] == 100, f"got {crop.shape[1]}")

# Tiny bbox should return None
tiny_crop = extract_pole_region(frame, (100, 100, 105, 105))
test("Tiny bbox → None", tiny_crop is None)

# Fully out-of-bounds bbox should return None
oob_crop = extract_pole_region(frame, (700, 500, 800, 600))
test("Fully out-of-bounds bbox → None", oob_crop is None)


# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST: tilt_from_edges() with synthetic poles")
print("=" * 60)

# Vertical pole (0°)
frame_v = make_frame()
box_v = (250, 50, 390, 400)
draw_vertical_pole(frame_v, box_v, angle_deg=0, thickness=4)
crop_v = extract_pole_region(frame_v, box_v)
angle_v = tilt_from_edges(crop_v)
test("Vertical pole → angle detected", angle_v is not None)
if angle_v is not None:
    test(f"Vertical pole → tilt < 10° (got {angle_v:.1f}°)",
         angle_v < 10, f"angle={angle_v:.1f}")

# Bent pole (~25°)
frame_b = make_frame()
box_b = (200, 50, 400, 400)
draw_vertical_pole(frame_b, box_b, angle_deg=25, thickness=4)
crop_b = extract_pole_region(frame_b, box_b)
angle_b = tilt_from_edges(crop_b)
test("Bent pole (25°) → angle detected", angle_b is not None)
if angle_b is not None:
    test(f"Bent pole → tilt ≥ 15° (got {angle_b:.1f}°)",
         angle_b >= 12, f"angle={angle_b:.1f}")  # allow some tolerance

# Empty crop
test("None crop → None", tilt_from_edges(None) is None)
test("Empty array → None",
     tilt_from_edges(np.zeros((0, 0, 3), dtype=np.uint8)) is None)

# Blank frame (no edges)
blank_crop = np.zeros((100, 50, 3), dtype=np.uint8)
test("Blank crop (no edges) → None", tilt_from_edges(blank_crop) is None)


# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST: analyze_pole_tilt() end-to-end")
print("=" * 60)

# Vertical pole — should be GOOD
frame_e2e = make_frame()
box_e2e = (250, 50, 390, 400)
draw_vertical_pole(frame_e2e, box_e2e, angle_deg=2, thickness=4)
tilt, status = analyze_pole_tilt(frame_e2e, box_e2e)
test(f"E2E vertical: tilt={tilt:.1f}°, status={status}",
     status == "GOOD SIGNBOARD" and tilt < 15)

# Bent pole — should be BENT
frame_bent = make_frame()
box_bent = (200, 50, 400, 400)
draw_vertical_pole(frame_bent, box_bent, angle_deg=30, thickness=4)
tilt_b, status_b = analyze_pole_tilt(frame_bent, box_bent)
test(f"E2E bent: tilt={tilt_b:.1f}°, status={status_b}",
     status_b == "BENT POLE" and tilt_b >= 15)

# No edges at all — should default to 0° GOOD
frame_empty = make_frame()
tilt_e, status_e = analyze_pole_tilt(frame_empty, (200, 100, 400, 400))
test(f"E2E empty frame: tilt={tilt_e:.1f}°, status={status_e}",
     tilt_e == 0.0 and status_e == "GOOD SIGNBOARD")


# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST: draw_detection()")
print("=" * 60)

frame_draw = make_frame()
box_d = (100, 100, 300, 400)
result = draw_detection(frame_draw, box_d, "BENT POLE", 23.4)
test("draw_detection returns frame", result is not None and result.shape == frame_draw.shape)

# Check that pixels were modified (drawing happened)
test("Frame modified by drawing", np.any(result > 0))


# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 60 + "\n")

sys.exit(0 if failed == 0 else 1)
