"""
Per-frame detection processing — ROI filter, class lock, confirm, dedup.

Contains the inner loop logic that was previously inlined inside
_process_video_blocking.  Operates on a TrackerState instance and
returns frame data for NDJSON streaming.
"""

from app.detectors.yolo.config import ROAD_DAMAGE_CLASSES
from app.detectors.yolo import verify_handler


def process_frame(
    results,
    model,
    frame,
    frame_count: int,
    current_time: float,
    tracker_state,
    *,
    roi_left: int,
    roi_right: int,
    roi_top: int,
    roi_bottom: int,
    detection_time_window: float,
    high_confidence_threshold: float,
    low_confidence_min_frames: int,
    min_distance_threshold: int,
    has_verify: bool,
    verify_fn=None,
    perf_timings: dict,
) -> tuple[dict, int]:
    """
    Process a single frame's YOLO results and update tracker_state.

    Returns:
        (frame_data dict, detection_count_delta)
    """
    frame_data = {"frame_id": frame_count, "detections": []}
    detection_count = 0

    # Process any completed verify futures from previous frames
    if has_verify:
        verify_handler.process_completed_futures(tracker_state, perf_timings)

    if results[0].boxes.id is None:
        return frame_data, detection_count

    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    for tid, cid, box, conf in zip(track_ids, class_ids, boxes, confidences):
        tid, cid = int(tid), int(cid)
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        class_name = str(model.names[cid])

        # ROI check
        if not (roi_left < cx < roi_right and roi_top < cy < roi_bottom):
            tracker_state.rejection_stats["roi_outside"] += 1
            continue

        # Class-lock enforcement
        if tid in tracker_state.tracker_class_lock:
            if tracker_state.tracker_class_lock[tid] != class_name:
                tracker_state.rejection_stats["class_mismatch"] += 1
                continue
        else:
            tracker_state.tracker_class_lock[tid] = class_name

        tracker_state.tracker_history[tid].append(current_time)
        tracker_state._tid_last_seen[tid] = current_time

        recent = [
            t
            for t in tracker_state.tracker_history[tid]
            if current_time - t <= detection_time_window
        ]
        min_needed = (
            1 if conf >= high_confidence_threshold else low_confidence_min_frames
        )

        if (
            len(recent) >= min_needed
            and tid not in tracker_state.confirmed
            and tid not in tracker_state.rejected_tids
        ):
            is_dup, _ = tracker_state.is_duplicate_location(
                cx, cy, class_name, current_time, min_distance_threshold
            )
            if not is_dup:
                # Optimistic accept — confirm now, verify async
                tracker_state.confirm_detection(
                    tid,
                    class_name,
                    frame_count,
                    current_time,
                    conf,
                    x1,
                    y1,
                    x2,
                    y2,
                    cx,
                    cy,
                )

                # Submit async verification if callback is provided
                if has_verify and verify_fn is not None:
                    verify_handler.submit_verification(
                        tid,
                        frame,
                        (x1, y1, x2, y2),
                        class_name,
                        verify_fn,
                        tracker_state,
                    )
            else:
                tracker_state.rejection_stats["spatial_duplicate"] += 1
        elif tid not in tracker_state.confirmed:
            tracker_state.rejection_stats["multi_frame_pending"].add(tid)

        if tid in tracker_state.confirmed:
            detection_count += 1
            frame_data["detections"].append(
                {
                    "frame_id": frame_count,
                    "detection_id": tid,
                    "type": class_name,
                    "confidence": round(float(conf), 3),
                    "count": {
                        "defected_sign_board": len(
                            tracker_state.counted_ids["defected_sign_board"]
                        ),
                        "pothole": len(tracker_state.counted_ids["pothole"]),
                        "road_crack": len(tracker_state.counted_ids["road_crack"]),
                        "damaged_road_marking": len(
                            tracker_state.counted_ids["damaged_road_marking"]
                        ),
                        "good_sign_board": len(
                            tracker_state.counted_ids["good_sign_board"]
                        ),
                    },
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center": {"x": cx, "y": cy},
                    "area": (x2 - x1) * (y2 - y1),
                }
            )

    return frame_data, detection_count
