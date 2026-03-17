"""
Stateless filter functions for the YOLO detection pipeline.

Each filter is a standalone function — add, remove, or modify filters here
without touching the main detector loop or any other module.
"""

from collections import deque


def calculate_distance(p1: tuple, p2: tuple) -> float:
    """Calculate Euclidean distance between two (x, y) points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def calculate_ios(box1: tuple, box2: tuple) -> float:
    """
    Calculate Intersection over Smaller Box (IoS).

    Args:
        box1, box2: (x1, y1, x2, y2) bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    smaller_area = min(area1, area2)
    if smaller_area == 0:
        return 0.0

    return intersection_area / smaller_area


def is_inside_roi(cx: int, cy: int, roi: dict) -> bool:
    """
    Check if a detection center (cx, cy) falls inside the ROI.

    Args:
        roi: dict with keys 'left', 'right', 'top', 'bottom'.
    """
    return roi["left"] < cx < roi["right"] and roi["top"] < cy < roi["bottom"]


def is_duplicate_location(
    cx: int,
    cy: int,
    bbox: tuple,
    class_name: str,
    current_time: float,
    spatial_locations: dict,
    time_threshold: float,
    min_distance_threshold: float,
) -> tuple:
    """
    Check if this location/class was already counted recently.

    spatial_locations is a defaultdict(deque) keyed by class_name.
    Entries are appended in time order so expired ones are pruned
    from the front in O(1) before scanning.

    Returns:
        (is_duplicate: bool, reason: str | None)
    """
    bucket = spatial_locations[class_name]

    # Prune entries outside the time window (deque is time-ordered)
    while bucket and (current_time - bucket[0]["time"]) > time_threshold:
        bucket.popleft()

    # Only same-class entries remain — no class check needed in the loop
    for existing in bucket:
        distance = calculate_distance((cx, cy), existing["center"])
        if distance < min_distance_threshold:
            time_gap = current_time - existing["time"]
            return True, f"{distance:.1f}px from existing, {time_gap:.2f}s ago"

        if bbox is not None and "bbox" in existing:
            ios = calculate_ios(bbox, existing["bbox"])
            if ios > 0.5:
                time_gap = current_time - existing["time"]
                return True, f"Overlap IoS {ios:.2f} with existing, {time_gap:.2f}s ago"

    return False, None
