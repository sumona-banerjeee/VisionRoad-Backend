"""
Database persistence for detection results.
"""

import time
import logging

from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.db.crud_hierarchy import find_location_by_gps

logger = logging.getLogger(__name__)


def save_detections_to_db(video_id: str, all_detections: list, perf_timings: dict = None):
    """
    Persist detections to the database with GPS-based location matching.

    Args:
        video_id: identifier of the processed video.
        all_detections: flat list of detection dicts (each with lat, lng, bbox, etc.).
        perf_timings: optional dict for timing accumulation.
    """
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
