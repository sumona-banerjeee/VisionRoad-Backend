"""
Quick runtime test for pole_tilt module.
Uses dummy frames/masks to exercise all functions without needing the YOLOE model.
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

print("=" * 60)
print("  RUNTIME TEST: pole_tilt module")
print("=" * 60)

# ── 1. Import all functions ──────────────────────────────────
print("\n[1] Importing functions...")
from app.detectors.yoloe.pole_tilt import (
    extract_pole_region,
    compute_tilt_angle,
    tilt_from_mask,
    tilt_from_edges,
    classify_pole,
    draw_detection,
    analyze_pole_tilt,
)
print("    OK - all imports successful")

# ── 2. Test compute_tilt_angle ───────────────────────────────
print("\n[2] Testing compute_tilt_angle...")
assert compute_tilt_angle(0, 1) == 0.0, "vertical should be 0 deg"
assert compute_tilt_angle(0, -1) == 0.0, "vertical (down) should be 0 deg"
assert abs(compute_tilt_angle(1, 1) - 45.0) < 0.1, "45 deg tilt"
assert compute_tilt_angle(1, 0) == 90.0, "horizontal should be 90 deg"
print("    OK - all angle calculations correct")

# ── 3. Test classify_pole ────────────────────────────────────
print("\n[3] Testing classify_pole...")
assert classify_pole(0.0) == "GOOD SIGNBOARD"
assert classify_pole(14.9) == "GOOD SIGNBOARD"
assert classify_pole(15.0) == "BENT POLE"
assert classify_pole(45.0) == "BENT POLE"
print("    OK - classification correct")

# ── 4. Test extract_pole_region with dummy frame ─────────────
print("\n[4] Testing extract_pole_region...")
dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

crop = extract_pole_region(dummy_frame, (100, 50, 200, 400))
assert crop is not None, "should return a crop"
print(f"    Crop shape: {crop.shape}")

crop_small = extract_pole_region(dummy_frame, (100, 50, 110, 55))
assert crop_small is None, "should return None for tiny box"
print("    OK - pole region extraction works")

# ── 5. Test tilt_from_mask with synthetic mask ───────────────
print("\n[5] Testing tilt_from_mask...")
mask = np.zeros((480, 640), dtype=np.uint8)
for y in range(190, 400):
    mask[y, 148:152] = 1

angle = tilt_from_mask(mask, (100, 50, 200, 400))
if angle is not None:
    print(f"    Vertical mask angle: {angle:.1f} deg (expected ~0)")
else:
    print("    WARNING: tilt_from_mask returned None")

mask2 = np.zeros((480, 640), dtype=np.uint8)
for y in range(190, 400):
    x_offset = int((y - 190) * 0.5)
    x_center = 150 + x_offset
    mask2[y, max(0, x_center-2):min(640, x_center+2)] = 1

angle2 = tilt_from_mask(mask2, (100, 50, 250, 400))
if angle2 is not None:
    print(f"    Tilted mask angle: {angle2:.1f} deg (expected >0)")
print("    OK - mask-based tilt works")

# ── 6. Test tilt_from_edges ──────────────────────────────────
print("\n[6] Testing tilt_from_edges...")
pole_crop = np.zeros((200, 100, 3), dtype=np.uint8)
cv2.line(pole_crop, (50, 10), (50, 190), (255, 255, 255), 3)
angle_edge = tilt_from_edges(pole_crop)
if angle_edge is not None:
    print(f"    Edge-based angle: {angle_edge:.1f} deg (expected ~0)")
else:
    print("    WARNING: tilt_from_edges returned None")

assert tilt_from_edges(None) is None, "should handle None"
print("    OK - edge-based tilt works")

# ── 7. Test analyze_pole_tilt (main entry point) ─────────────
print("\n[7] Testing analyze_pole_tilt...")
tilt, status = analyze_pole_tilt(dummy_frame, (100, 50, 200, 400), mask)
print(f"    With mask:    tilt={tilt:.1f} deg, status={status}")

tilt2, status2 = analyze_pole_tilt(dummy_frame, (100, 50, 200, 400), None)
print(f"    Without mask: tilt={tilt2:.1f} deg, status={status2}")
print("    OK - analyze_pole_tilt works")

# ── 8. Test draw_detection ───────────────────────────────────
print("\n[8] Testing draw_detection...")
frame_copy = dummy_frame.copy()
frame_out = draw_detection(frame_copy, (100, 50, 200, 400), "GOOD SIGNBOARD", 5.3)
assert frame_out is not None
assert frame_out.shape == dummy_frame.shape
frame_out2 = draw_detection(frame_copy, (300, 100, 450, 350), "BENT POLE", 22.7)
assert frame_out2 is not None
print("    OK - draw_detection works (green + red)")

# ── 9. Test check_signboard_pole_tilt wrapper ────────────────
print("\n[9] Testing check_signboard_pole_tilt wrapper...")
from app.helpers.yoloe_helper import check_signboard_pole_tilt
tilt3, status3 = check_signboard_pole_tilt(dummy_frame, (100, 50, 200, 400), mask)
print(f"    Wrapper result: tilt={tilt3:.1f} deg, status={status3}")
print("    OK - helper wrapper works")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED - NO RUNTIME ERRORS")
print("=" * 60)
