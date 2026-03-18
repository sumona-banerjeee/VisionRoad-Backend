"""Chainage mapping service for GPS-based detection assignment (replaces location_mapper.py)"""

from sqlalchemy.orm import Session
from typing import Optional
from app.models.chainage import Chainage


def get_chainage_hierarchy(db: Session, chainage_id: str) -> Optional[dict]:
    """
    Get full hierarchy (Project → Package → Chainage) for a chainage.

    Returns:
        Dictionary with project, package, and chainage details
    """
    chainage = db.query(Chainage).filter(Chainage.id == chainage_id).first()
    if not chainage:
        return None

    package = chainage.package
    project = package.project if package else None

    return {
        "project": (
            {
                "id": project.id,
                "name": project.name,
                "corridor_name": project.corridor_name,
            }
            if project
            else None
        ),
        "package": (
            {
                "id": package.id,
                "name": package.name,
                "region": package.region,
                "chainage_start_km": package.chainage_start_km,
                "chainage_end_km": package.chainage_end_km,
            }
            if package
            else None
        ),
        "chainage": {
            "id": chainage.id,
            "segment_name": chainage.segment_name,
            "chainage_start_km": chainage.chainage_start_km,
            "chainage_end_km": chainage.chainage_end_km,
            "direction": chainage.direction,
        },
    }


def validate_gps_bounds(
    start_lat: float, start_lng: float, end_lat: float, end_lng: float
) -> bool:
    """
    Validate GPS bounding box coordinates.

    NOTE: start/end don't need to be in strict order since roads can travel
    in any direction. We just validate that all coordinates are in valid ranges.

    Args:
        start_lat: Starting latitude
        start_lng: Starting longitude
        end_lat: Ending latitude
        end_lng: Ending longitude

    Returns:
        True if bounds are valid, False otherwise
    """
    # Basic validation - all must be numbers
    if not all(
        isinstance(x, (int, float)) for x in [start_lat, start_lng, end_lat, end_lng]
    ):
        return False

    # Latitude range: -90 to 90
    if not (-90 <= start_lat <= 90 and -90 <= end_lat <= 90):
        return False

    # Longitude range: -180 to 180
    if not (-180 <= start_lng <= 180 and -180 <= end_lng <= 180):
        return False

    # Ensure bounding box has some area (not a single point)
    if start_lat == end_lat and start_lng == end_lng:
        return False

    return True
