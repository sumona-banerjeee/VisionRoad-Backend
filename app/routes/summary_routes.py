"""Summary and analytics routes for hierarchical project views"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.database import get_db
from app.models.detection import Detection
from app.models.location import Location
from app.models.package import Package
from app.models.project import Project


router = APIRouter(prefix="/summary", tags=["Analytics"])


@router.get("/projects/{project_id}")
async def get_project_summary(
    project_id: str, video_id: str = None, db: Session = Depends(get_db)
):
    """
    Get detection summary for a project, grouped by package and location.

    Response format:
    {
        "project": {...},
        "packages": {
            "Package Name": {
                "locations": {
                    "Segment Name": {
                        "detection_count": 10,
                        "detections": [...]
                    }
                }
            }
        }
    }
    """
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get all detections for this project with locations and packages
    # Build query
    query = (
        db.query(Detection, Location, Package)
        .join(Location, Detection.location_id == Location.id)
        .join(Package, Location.package_id == Package.id)
        .filter(Detection.project_id == project_id)
    )

    # Optional video_id filter
    if video_id:
        query = query.filter(Detection.video_id == video_id)

    detections = query.all()

    # Build hierarchical response
    summary = {
        "project": {
            "id": project.id,
            "name": project.name,
            "corridor_name": project.corridor_name,
            "state": project.state,
        },
        "packages": {},
    }

    for detection, location, package in detections:
        # Initialize package if not exists
        if package.name not in summary["packages"]:
            summary["packages"][package.name] = {
                "package_id": package.id,
                "region": package.region,
                "locations": {},
            }

        # Initialize location if not exists
        if location.segment_name not in summary["packages"][package.name]["locations"]:
            summary["packages"][package.name]["locations"][location.segment_name] = {
                "location_id": location.id,
                "chainage": (
                    f"{location.chainage_start_km}-{location.chainage_end_km} km"
                    if location.chainage_start_km
                    else None
                ),
                "detection_count": 0,
                "detections": [],
            }

        # Add detection
        summary["packages"][package.name]["locations"][location.segment_name][
            "detection_count"
        ] += 1
        summary["packages"][package.name]["locations"][location.segment_name][
            "detections"
        ].append(
            {
                "id": detection.id,
                "video_id": detection.video_id,
                "type": detection.detection_type,
                "class": detection.class_name,
                "confidence": detection.confidence,
                "latitude": detection.latitude,
                "longitude": detection.longitude,
                "frame_number": detection.frame_number,
                "timestamp_ms": detection.timestamp_ms,
                "bounding_box": detection.get_bounding_box(),
            }
        )

    return summary


@router.get("/packages/{package_id}")
async def get_package_summary(
    package_id: str, video_id: str = None, db: Session = Depends(get_db)
):
    """
    Get detection summary for a package, grouped by location.
    """
    # Verify package exists
    package = db.query(Package).filter(Package.id == package_id).first()
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")

    # Build query
    query = (
        db.query(Detection, Location)
        .join(Location, Detection.location_id == Location.id)
        .filter(Detection.package_id == package_id)
    )

    # Optional video_id filter
    if video_id:
        query = query.filter(Detection.video_id == video_id)

    detections = query.all()

    summary = {
        "package": {
            "id": package.id,
            "name": package.name,
            "region": package.region,
            "project_id": package.project_id,
        },
        "locations": {},
    }

    for detection, location in detections:
        if location.segment_name not in summary["locations"]:
            summary["locations"][location.segment_name] = {
                "location_id": location.id,
                "chainage": (
                    f"{location.chainage_start_km}-{location.chainage_end_km} km"
                    if location.chainage_start_km
                    else None
                ),
                "detection_count": 0,
                "detections": [],
            }

        summary["locations"][location.segment_name]["detection_count"] += 1
        summary["locations"][location.segment_name]["detections"].append(
            {
                "id": detection.id,
                "video_id": detection.video_id,
                "type": detection.detection_type,
                "class": detection.class_name,
                "confidence": detection.confidence,
                "latitude": detection.latitude,
                "longitude": detection.longitude,
                "frame_number": detection.frame_number,
                "timestamp_ms": detection.timestamp_ms,
                "bounding_box": detection.get_bounding_box(),
            }
        )

    return summary


@router.get("/locations/{location_id}")
async def get_location_summary(
    location_id: str, video_id: str = None, db: Session = Depends(get_db)
):
    """
    Get detection summary for a specific location.
    """
    # Verify location exists
    location = db.query(Location).filter(Location.id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    # Get all detections for this location
    query = db.query(Detection).filter(Detection.location_id == location_id)

    # Optional video_id filter
    if video_id:
        query = query.filter(Detection.video_id == video_id)

    detections = query.all()

    # Count by type
    detection_counts = (
        db.query(Detection.detection_type, func.count(Detection.id))
        .filter(Detection.location_id == location_id)
        .group_by(Detection.detection_type)
        .all()
    )

    summary = {
        "location": {
            "id": location.id,
            "segment_name": location.segment_name,
            "chainage": (
                f"{location.chainage_start_km}-{location.chainage_end_km} km"
                if location.chainage_start_km
                else None
            ),
            "package_id": location.package_id,
        },
        "statistics": {
            "total_detections": len(detections),
            "by_type": {dtype: count for dtype, count in detection_counts},
        },
        "detections": [
            {
                "id": d.id,
                "video_id": d.video_id,
                "type": d.detection_type,
                "class": d.class_name,
                "confidence": d.confidence,
                "latitude": d.latitude,
                "longitude": d.longitude,
                "frame_number": d.frame_number,
                "timestamp_ms": d.timestamp_ms,
                "bounding_box": d.get_bounding_box(),
            }
            for d in detections
        ],
    }

    return summary


@router.get("/projects/{project_id}/statistics")
async def get_project_statistics(project_id: str, db: Session = Depends(get_db)):
    """Get high-level statistics for a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Count packages
    package_count = (
        db.query(func.count(Package.id))
        .filter(Package.project_id == project_id)
        .scalar()
    )

    # Count locations
    location_count = (
        db.query(func.count(Location.id))
        .join(Package, Location.package_id == Package.id)
        .filter(Package.project_id == project_id)
        .scalar()
    )

    # Count detections by type
    detection_stats = (
        db.query(Detection.detection_type, func.count(Detection.id))
        .filter(Detection.project_id == project_id)
        .group_by(Detection.detection_type)
        .all()
    )

    return {
        "project_id": project_id,
        "project_name": project.name,
        "package_count": package_count,
        "location_count": location_count,
        "total_detections": sum(count for _, count in detection_stats),
        "detections_by_type": {dtype: count for dtype, count in detection_stats},
    }
