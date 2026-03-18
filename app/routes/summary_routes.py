"""Summary and analytics routes for hierarchical project views"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.database import get_db
from app.models.detection import Detection
from app.models.chainage import Chainage
from app.models.lane import Lane
from app.models.package import Package
from app.models.project import Project


router = APIRouter(prefix="/summary", tags=["Analytics"])


@router.get("/projects/{project_id}")
async def get_project_summary(
    project_id: str, video_id: str = None, db: Session = Depends(get_db)
):
    """
    Get detection summary for a project, grouped by package → chainage → lane.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    query = (
        db.query(Detection, Chainage, Lane, Package)
        .join(Chainage, Detection.chainage_id == Chainage.id)
        .join(Lane, Detection.lane_id == Lane.id)
        .join(Package, Chainage.package_id == Package.id)
        .filter(Detection.project_id == project_id)
    )

    if video_id:
        query = query.filter(Detection.video_id == video_id)

    detections = query.all()

    summary = {
        "project": {
            "id": project.id,
            "name": project.name,
            "corridor_name": project.corridor_name,
            "state": project.state,
        },
        "packages": {},
    }

    for detection, chainage, lane, package in detections:
        # Package level
        if package.name not in summary["packages"]:
            summary["packages"][package.name] = {
                "package_id": package.id,
                "region": package.region,
                "chainages": {},
            }

        # Chainage level
        ch_key = chainage.segment_name
        if ch_key not in summary["packages"][package.name]["chainages"]:
            summary["packages"][package.name]["chainages"][ch_key] = {
                "chainage_id": chainage.id,
                "chainage_km": f"{chainage.chainage_start_km}–{chainage.chainage_end_km} km",
                "lanes": {},
            }

        # Lane level
        lane_key = lane.lane_code
        if lane_key not in summary["packages"][package.name]["chainages"][ch_key]["lanes"]:
            summary["packages"][package.name]["chainages"][ch_key]["lanes"][lane_key] = {
                "lane_id": lane.id,
                "lane_type": lane.lane_type,
                "detection_count": 0,
                "detections": [],
            }

        _append_detection(
            summary["packages"][package.name]["chainages"][ch_key]["lanes"][lane_key],
            detection,
        )

    return summary


@router.get("/packages/{package_id}")
async def get_package_summary(
    package_id: str, video_id: str = None, db: Session = Depends(get_db)
):
    """
    Get detection summary for a package, grouped by chainage → lane.
    """
    package = db.query(Package).filter(Package.id == package_id).first()
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")

    query = (
        db.query(Detection, Chainage, Lane)
        .join(Chainage, Detection.chainage_id == Chainage.id)
        .join(Lane, Detection.lane_id == Lane.id)
        .filter(Detection.package_id == package_id)
    )

    if video_id:
        query = query.filter(Detection.video_id == video_id)

    detections = query.all()

    summary = {
        "package": {
            "id": package.id,
            "name": package.name,
            "region": package.region,
            "project_id": package.project_id,
            "chainage_km": (
                f"{package.chainage_start_km}–{package.chainage_end_km} km"
                if package.chainage_start_km is not None
                else None
            ),
        },
        "chainages": {},
    }

    for detection, chainage, lane in detections:
        ch_key = chainage.segment_name
        if ch_key not in summary["chainages"]:
            summary["chainages"][ch_key] = {
                "chainage_id": chainage.id,
                "chainage_km": f"{chainage.chainage_start_km}–{chainage.chainage_end_km} km",
                "lanes": {},
            }

        lane_key = lane.lane_code
        if lane_key not in summary["chainages"][ch_key]["lanes"]:
            summary["chainages"][ch_key]["lanes"][lane_key] = {
                "lane_id": lane.id,
                "lane_type": lane.lane_type,
                "detection_count": 0,
                "detections": [],
            }

        _append_detection(summary["chainages"][ch_key]["lanes"][lane_key], detection)

    return summary


@router.get("/chainages/{chainage_id}")
async def get_chainage_summary(
    chainage_id: str, video_id: str = None, db: Session = Depends(get_db)
):
    """
    Get detection summary for a specific chainage, grouped by lane.
    """
    chainage = db.query(Chainage).filter(Chainage.id == chainage_id).first()
    if not chainage:
        raise HTTPException(status_code=404, detail="Chainage not found")

    query = (
        db.query(Detection, Lane)
        .join(Lane, Detection.lane_id == Lane.id)
        .filter(Detection.chainage_id == chainage_id)
    )

    if video_id:
        query = query.filter(Detection.video_id == video_id)

    detections = query.all()

    detection_counts = (
        db.query(Detection.detection_type, func.count(Detection.id))
        .filter(Detection.chainage_id == chainage_id)
        .group_by(Detection.detection_type)
        .all()
    )

    summary = {
        "chainage": {
            "id": chainage.id,
            "segment_name": chainage.segment_name,
            "chainage_km": f"{chainage.chainage_start_km}–{chainage.chainage_end_km} km",
            "package_id": chainage.package_id,
        },
        "statistics": {
            "total_detections": sum(c for _, c in detection_counts),
            "by_type": {dtype: count for dtype, count in detection_counts},
        },
        "lanes": {},
    }

    for detection, lane in detections:
        lane_key = lane.lane_code
        if lane_key not in summary["lanes"]:
            summary["lanes"][lane_key] = {
                "lane_id": lane.id,
                "lane_type": lane.lane_type,
                "detection_count": 0,
                "detections": [],
            }
        _append_detection(summary["lanes"][lane_key], detection)

    return summary


@router.get("/projects/{project_id}/statistics")
async def get_project_statistics(project_id: str, db: Session = Depends(get_db)):
    """Get high-level statistics for a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    package_count = (
        db.query(func.count(Package.id))
        .filter(Package.project_id == project_id)
        .scalar()
    )

    chainage_count = (
        db.query(func.count(Chainage.id))
        .join(Package, Chainage.package_id == Package.id)
        .filter(Package.project_id == project_id)
        .scalar()
    )

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
        "chainage_count": chainage_count,
        "total_detections": sum(count for _, count in detection_stats),
        "detections_by_type": {dtype: count for dtype, count in detection_stats},
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _append_detection(container: dict, detection: Detection) -> None:
    """Append a detection record to a lane-level summary container."""
    container["detection_count"] += 1
    container["detections"].append(
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
