"""Location mapping service for GPS-based detection assignment"""

from sqlalchemy.orm import Session
from typing import Optional
from app.models.location import Location
from app.models.package import Package
from app.models.project import Project


def find_location_by_gps(
    db: Session, lat: float, lng: float, package_id: Optional[str] = None
) -> Optional[Location]:
    """
    Find location that contains the given GPS coordinate.

    Uses bounding box matching with min/max to handle roads in any direction.

    Args:
        db: Database session
        lat: Latitude coordinate
        lng: Longitude coordinate
        package_id: Optional package filter to narrow down search

    Returns:
        Location object if GPS falls within a location's bounds, None otherwise

    Example:
        >>> location = find_location_by_gps(db, 22.5726, 88.3639)
        >>> if location:
        >>>     print(f"Detection in {location.segment_name}")
    """
    """
    Find location that contains the given GPS coordinate.

    Uses database query with OR conditions to handle road segments defined in either direction.
    """
    # Create filters for latitude and longitude (handling both directions)
    # Lat between start and end OR between end and start
    lat_filter = ((Location.start_lat <= lat) & (lat <= Location.end_lat)) | \
                 ((Location.end_lat <= lat) & (lat <= Location.start_lat))
    
    # Lng between start and end OR between end and start
    lng_filter = ((Location.start_lng <= lng) & (lng <= Location.end_lng)) | \
                 ((Location.end_lng <= lng) & (lng <= Location.start_lng))
    
    query = db.query(Location).filter(lat_filter, lng_filter)
    
    if package_id:
        query = query.filter(Location.package_id == package_id)

    # Return the first matching location
    return query.first()


def get_location_hierarchy(db: Session, location_id: str) -> Optional[dict]:
    """
    Get full hierarchy (Project → Package → Location) for a location.

    Returns:
        Dictionary with project, package, and location details
    """
    location = db.query(Location).filter(Location.id == location_id).first()
    if not location:
        return None

    package = location.package
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
            {"id": package.id, "name": package.name, "region": package.region}
            if package
            else None
        ),
        "location": {
            "id": location.id,
            "segment_name": location.segment_name,
            "chainage_start_km": location.chainage_start_km,
            "chainage_end_km": location.chainage_end_km,
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
