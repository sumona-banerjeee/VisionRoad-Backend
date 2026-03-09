"""
Result builder — final output assembly, GPS enrichment, DB persistence.
"""

import json
import time
import logging
from datetime import datetime

from app.detectors.base.base_detector import BaseDetector
from app.detectors.yolo.config import ROAD_DAMAGE_CLASSES
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.db.crud_hierarchy import find_location_by_gps

logger = logging.getLogger(__name__)


def read_ndjson_frames(ndjson_fd, ndjson_path: str, expected_count: int = 0) -> list:
    """Close the NDJSON temp file and read all frame lines back as a list."""
    if not ndjson_fd.closed:
        ndjson_fd.close()
    frames = []
    try:
        with open(ndjson_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Failed to read back NDJSON frames: {e}")
    logger.info(
        f"NDJSON read-back complete — {len(frames)} frames loaded "
        f"(expected {expected_count}) from {ndjson_path}"
    )
    return frames


def build_class_list(
    confirmed: dict, class_name: str, gps_points: list, perf_timings: dict | None = None
) -> list:
    """Build a sorted list of confirmed detections for a given class, with GPS enrichment."""
    lst = []
    gps_timestamps = [p.get("timestamp", 0) for p in gps_points] if gps_points else []
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


def build_results_dict(
    video_id: str,
    video_path: str,
    detection_mode: str,
    tracker_state,
    gps_points: list,
    perf_timings: dict,
    video_info: dict,
    total_detections_count: int,
    frames_written: int,
    frame_count: int,
    ndjson_fd,
    ndjson_path: str,
    has_verify: bool,
) -> dict:
    """Assemble the final results dictionary."""
    defected_sign_board_list = build_class_list(
        tracker_state.confirmed, "defected_sign_board", gps_points, perf_timings
    )
    pothole_list = build_class_list(
        tracker_state.confirmed, "pothole", gps_points, perf_timings
    )
    road_crack_list = build_class_list(
        tracker_state.confirmed, "road_crack", gps_points, perf_timings
    )
    damaged_road_marking_list = build_class_list(
        tracker_state.confirmed, "damaged_road_marking", gps_points, perf_timings
    )
    good_sign_board_list = build_class_list(
        tracker_state.confirmed, "good_sign_board", gps_points, perf_timings
    )

    detection_rate = (
        round((frames_written / frame_count) * 100, 2) if frame_count > 0 else 0
    )

    results = {
        "video_id": video_id,
        "video_path": video_path,
        "detection_mode": detection_mode,
        "processed_at": datetime.now().isoformat(),
        "video_info": video_info,
        "summary": {
            "total_frames": frame_count,
            "unique_defected_sign_board": len(
                tracker_state.counted_ids["defected_sign_board"]
            ),
            "unique_pothole": len(tracker_state.counted_ids["pothole"]),
            "unique_road_crack": len(tracker_state.counted_ids["road_crack"]),
            "unique_damaged_road_marking": len(
                tracker_state.counted_ids["damaged_road_marking"]
            ),
            "unique_good_sign_board": len(tracker_state.counted_ids["good_sign_board"]),
            "total_road_damage": sum(
                len(tracker_state.counted_ids[c]) for c in ROAD_DAMAGE_CLASSES
            ),
            "total_detections": total_detections_count,
            "frames_with_detections": frames_written,
            "detection_rate": detection_rate,
        },
        "rejection_stats": {
            "multi_frame_pending": len(
                tracker_state.rejection_stats["multi_frame_pending"]
            ),
            "spatial_duplicate": tracker_state.rejection_stats["spatial_duplicate"],
            "class_mismatch": tracker_state.rejection_stats["class_mismatch"],
            "roi_outside": tracker_state.rejection_stats["roi_outside"],
            "vl_mismatch": tracker_state.rejection_stats["vl_mismatch"],
            "vl_errors": tracker_state.rejection_stats["vl_errors"],
        },
        "vl_stats": (
            {
                "enabled": has_verify,
                "total_verified": tracker_state.verify_stats["total_verified"],
                "verified_success": tracker_state.verify_stats["verified_success"],
                "verified_failed": tracker_state.verify_stats["verified_failed"],
                "vl_overrides": tracker_state.verify_stats["vl_overrides"],
                "cache_hits": tracker_state.verify_stats["skipped"],
            }
            if has_verify
            else None
        ),
        "defected_sign_board_list": defected_sign_board_list,
        "pothole_list": pothole_list,
        "road_crack_list": road_crack_list,
        "damaged_road_marking_list": damaged_road_marking_list,
        "good_sign_board_list": good_sign_board_list,
        "frames": read_ndjson_frames(ndjson_fd, ndjson_path, frames_written),
    }

    all_detections_flat = (
        defected_sign_board_list
        + pothole_list
        + road_crack_list
        + damaged_road_marking_list
        + good_sign_board_list
    )

    return results, all_detections_flat


def save_to_db(video_id: str, all_detections: list, perf_timings: dict | None = None):
    """Persist confirmed detections to the database with GPS-based location matching."""
    db = SessionLocal()
    try:
        db_detections = []
        for det in all_detections:
            lat, lng = det.get("lat"), det.get("lng")
            project_id, package_id, location_id = None, None, None
            if lat and lng:
                _t0 = time.perf_counter()
                location = find_location_by_gps(db, lat, lng)
                _gps_db_elapsed = time.perf_counter() - _t0
                if perf_timings is not None:
                    perf_timings["db_gps_match"]["total"] += _gps_db_elapsed
                    perf_timings["db_gps_match"]["count"] += 1
                if location:
                    location_id, package_id = location.id, location.package_id
                    if location.package:
                        project_id = location.package.project_id

            db_det = Detection(
                video_id=video_id,
                frame_number=det["first_detected_frame"],
                timestamp_ms=int(det["first_detected_time"] * 1000),
                confidence=det["confidence"],
                detection_type=det["type"],
                class_name=det["type"],
                latitude=lat,
                longitude=lng,
                project_id=project_id,
                package_id=package_id,
                location_id=location_id,
            )
            db_det.set_bounding_box(det.get("bbox", {}))
            db_detections.append(db_det)

        if db_detections:
            _t0 = time.perf_counter()
            crud.create_detections_bulk(db, db_detections)
            _bulk_elapsed = time.perf_counter() - _t0
            if perf_timings is not None:
                perf_timings["db_bulk_write"]["total"] += _bulk_elapsed
                perf_timings["db_bulk_write"]["count"] += 1
            logger.info(
                f"Saved {len(db_detections)} detections to database for {video_id}"
            )
    except Exception as e:
        logger.error(f"Database save error: {e}")
    finally:
        db.close()
