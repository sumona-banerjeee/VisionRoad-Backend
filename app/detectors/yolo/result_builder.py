"""
Result building and performance reporting for the YOLO detection pipeline.
"""

import time
import logging
from datetime import datetime

from app.detectors.base.base_detector import BaseDetector
from app.detectors.yolo.config import ROAD_DAMAGE_CLASSES, FRAME_SKIP
from app.core.logging_config import perf_logger

logger = logging.getLogger(__name__)


def build_class_list(confirmed: dict, class_name: str, gps_points: list, perf_timings: dict = None) -> list:
    """
    Build a sorted list of detections for a given class from confirmed detections.

    Looks up GPS coordinates for each detection and accumulates timing.

    Returns:
        Sorted list of detection dicts.
    """
    lst = []
    # Pre-build timestamp list once — find_nearest_gps reuses it (O(log N) per call)
    gps_timestamps = (
        [p.get("timestamp", 0) for p in gps_points] if gps_points else []
    )
    for tid, info in confirmed.items():
        if info["type"] == class_name:
            det_data = {
                "detection_id": tid,
                "type": info["type"],
                "first_detected_frame": info["first_detected_frame"],
                "first_detected_time": info["first_detected_time"],
                "confidence": info["confidence"],
                "bbox": info.get("bbox", {}),
                "vl_verified": info.get("vl_verified"),
                "vl_confidence": info.get("vl_confidence"),
                "vl_category": info.get("vl_category"),
            }
            if gps_points:
                _t0 = time.perf_counter()
                gps_coords = BaseDetector.find_nearest_gps(
                    info["first_detected_time"], gps_points, gps_timestamps
                )
                _gps_elapsed = time.perf_counter() - _t0
                if perf_timings is not None:
                    perf_timings["gps_coord"]["total"] += _gps_elapsed
                    perf_timings["gps_coord"]["count"] += 1
                det_data.update(gps_coords)
            lst.append(det_data)
    return sorted(lst, key=lambda x: x["first_detected_frame"])


def build_result_dict(
    video_id: str,
    video_path: str,
    detection_mode: str,
    video_info: dict,
    state,
    class_lists: dict,
    frames: list,
    total_detections_count: int,
    frames_with_detections: int,
    has_verify: bool,
) -> dict:
    """
    Build the final detection results dictionary.

    Args:
        video_info: dict with total_frames, fps, duration, width, height.
        state: TrackerState instance.
        class_lists: dict mapping class_name → sorted list of detections.
        frames: list of frame data dicts (from NDJSON read-back).
        total_detections_count: total detection entries in frames.
        frames_with_detections: how many frames contained detections.
        has_verify: whether verification was enabled.

    Returns:
        Complete results dict ready for JSON serialization.
    """
    frame_count = video_info["total_frames"]
    detection_rate = (
        round((frames_with_detections / frame_count) * 100, 2)
        if frame_count > 0
        else 0
    )

    return {
        "video_id": video_id,
        "video_path": video_path,
        "detection_mode": detection_mode,
        "processed_at": datetime.now().isoformat(),
        "video_info": {
            "total_frames": video_info["total_frames_raw"],
            "fps": round(video_info["fps"], 2),
            "duration": round(video_info["duration"], 2),
            "width": video_info["width"],
            "height": video_info["height"],
            "resolution": f"{video_info['width']}x{video_info['height']}",
        },
        "summary": {
            "total_frames": frame_count,
            "unique_defected_sign_board": len(state.counted_ids["defected_sign_board"]),
            "unique_pothole": len(state.counted_ids["pothole"]),
            "unique_road_crack": len(state.counted_ids["road_crack"]),
            "unique_damaged_road_marking": len(state.counted_ids["damaged_road_marking"]),
            "unique_good_sign_board": len(state.counted_ids["good_sign_board"]),
            "total_road_damage": sum(
                len(state.counted_ids[c]) for c in ROAD_DAMAGE_CLASSES
            ),
            "total_detections": total_detections_count,
            "frames_with_detections": frames_with_detections,
            "detection_rate": detection_rate,
        },
        "rejection_stats": {
            "multi_frame_pending": len(state.rejection_stats["multi_frame_pending"]),
            "spatial_duplicate": state.rejection_stats["spatial_duplicate"],
            "class_mismatch": state.rejection_stats["class_mismatch"],
            "roi_outside": state.rejection_stats["roi_outside"],
            "vl_mismatch": state.rejection_stats["vl_mismatch"],
            "vl_errors": state.rejection_stats["vl_errors"],
        },
        "vl_stats": (
            {
                "enabled": has_verify,
                "total_verified": state.verify_stats["total_verified"],
                "verified_success": state.verify_stats["verified_success"],
                "verified_failed": state.verify_stats["verified_failed"],
                "vl_overrides": state.verify_stats["vl_overrides"],
                "cache_hits": state.verify_stats["skipped"],
            }
            if has_verify
            else None
        ),
        "defected_sign_board_list": class_lists.get("defected_sign_board", []),
        "pothole_list": class_lists.get("pothole", []),
        "road_crack_list": class_lists.get("road_crack", []),
        "damaged_road_marking_list": class_lists.get("damaged_road_marking", []),
        "good_sign_board_list": class_lists.get("good_sign_board", []),
        "frames": frames,
    }


def format_perf_report(
    video_id: str,
    detection_mode: str,
    perf_timings: dict,
    video_info: dict,
    verify_stats: dict,
    total_time: float,
    yolo_time: float,
    drain_time: float,
    num_detections: int,
):
    """
    Format and log the structured performance report.

    Writes to the dedicated perf log only (detection.log stays clean).
    """
    frames_processed = (
        video_info["total_frames"] // FRAME_SKIP if FRAME_SKIP > 1 else video_info["total_frames"]
    )

    def _fmt(stage_key):
        d = perf_timings[stage_key]
        total = d["total"]
        count = d["count"]
        avg_ms = (total / count * 1000) if count > 0 else 0.0
        return total, count, avg_ms

    fd_t, fd_c, fd_avg = _fmt("frame_decode")
    yi_t, yi_c, yi_avg = _fmt("yolo_inference")
    gc_t, gc_c, gc_avg = _fmt("gps_coord")
    vl_t, vl_c, vl_avg = _fmt("verification")
    dg_t, dg_c, dg_avg = _fmt("db_gps_match")
    db_t, db_c, db_avg = _fmt("db_bulk_write")

    def _flag(t):
        return " ⚠️" if total_time > 0 and (t / total_time) > 0.20 else ""

    fps = video_info["fps"]
    total_frames_raw = video_info["total_frames_raw"]
    video_duration = video_info["duration"]

    report_lines = [
        f"{'=' * 78}",
        f"  PERF REPORT — [{video_id}]  mode={detection_mode}",
        f"{'=' * 78}",
        f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}",
        f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
        f"  {'Frame decode (I/O)':<30} {fd_t:>10.3f} {fd_c:>7d} {fd_avg:>15.2f}{_flag(fd_t)}",
        f"  {'YOLO inference':<30} {yi_t:>10.3f} {yi_c:>7d} {yi_avg:>15.2f}{_flag(yi_t)}",
        f"  {'GPS coord lookup':<30} {gc_t:>10.3f} {gc_c:>7d} {gc_avg:>15.2f}{_flag(gc_t)}",
        f"  {'Verification (async)':<30} {vl_t:>10.3f} {vl_c:>7d} {vl_avg:>15.2f}{_flag(vl_t)}",
        f"  {'Verify drain (post-loop)':<30} {drain_time:>10.3f} {'N/A':>7} {'N/A':>15}",
        f"  {'DB GPS matching':<30} {dg_t:>10.3f} {dg_c:>7d} {dg_avg:>15.2f}{_flag(dg_t)}",
        f"  {'DB bulk write':<30} {db_t:>10.3f} {db_c:>7d} {db_avg:>15.2f}{_flag(db_t)}",
        f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
        f"  {'TOTAL pipeline time':<30} {total_time:>10.3f}s",
        f"  {'Video duration':<30} {video_duration:>10.1f}s  ({total_frames_raw} frames @ {fps:.0f} FPS)",
        f"  {'Frames processed':<30} {frames_processed:>10d}  (FRAME_SKIP={FRAME_SKIP})",
        f"  {'Detections saved':<30} {num_detections:>10d}",
        f"  {'Verifications done':<30} {verify_stats['total_verified']:>10d}",
        f"{'=' * 78}",
    ]
    report_str = "\n".join(report_lines)
    perf_logger.info(f"\n{report_str}")
