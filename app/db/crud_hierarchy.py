"""CRUD operations for Project, Package, Chainage, and Lane models"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from typing import Optional, List
from app.models.project import Project
from app.models.package import Package
from app.models.chainage import Chainage
from app.models.lane import Lane
import uuid


# ==================== Project CRUD ====================


def create_project(
    db: Session,
    name: str,
    state: Optional[str] = None,
    corridor_name: Optional[str] = None,
    start_lat: Optional[float] = None,
    start_lng: Optional[float] = None,
    end_lat: Optional[float] = None,
    end_lng: Optional[float] = None,
) -> Project:
    """Create a new project"""
    project = Project(
        id=str(uuid.uuid4()),
        name=name,
        state=state,
        corridor_name=corridor_name,
        start_lat=start_lat,
        start_lng=start_lng,
        end_lat=end_lat,
        end_lng=end_lng,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def get_project(db: Session, project_id: str) -> Optional[Project]:
    """Get a project by ID"""
    return db.query(Project).filter(Project.id == project_id).first()


def list_projects(db: Session, skip: int = 0, limit: int = 100) -> List[Project]:
    """Get all projects ordered by creation time (newest first)"""
    return (
        db.query(Project)
        .order_by(desc(Project.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_project(db: Session, project_id: str, **kwargs) -> Optional[Project]:
    """Update project fields"""
    project = get_project(db, project_id)
    if project:
        for key, value in kwargs.items():
            if hasattr(project, key):
                setattr(project, key, value)
        db.commit()
        db.refresh(project)
    return project


def delete_project(db: Session, project_id: str) -> bool:
    """Delete a project (cascades to packages, chainages, lanes)"""
    project = get_project(db, project_id)
    if project:
        db.delete(project)
        db.commit()
        return True
    return False


# ==================== Package CRUD ====================


def create_package(
    db: Session,
    project_id: str,
    name: str,
    region: Optional[str] = None,
    chainage_start_km: Optional[float] = None,
    chainage_end_km: Optional[float] = None,
) -> Package:
    """Create a new package in a project"""
    package = Package(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name=name,
        region=region,
        chainage_start_km=chainage_start_km,
        chainage_end_km=chainage_end_km,
    )
    db.add(package)
    db.commit()
    db.refresh(package)
    return package


def get_package(db: Session, package_id: str) -> Optional[Package]:
    """Get a package by ID"""
    return db.query(Package).filter(Package.id == package_id).first()


def list_packages(
    db: Session, project_id: Optional[str] = None, skip: int = 0, limit: int = 100
) -> List[Package]:
    """Get all packages, optionally filtered by project"""
    query = db.query(Package)
    if project_id:
        query = query.filter(Package.project_id == project_id)
    return query.order_by(desc(Package.created_at)).offset(skip).limit(limit).all()


def update_package(db: Session, package_id: str, **kwargs) -> Optional[Package]:
    """Update package fields"""
    package = get_package(db, package_id)
    if package:
        for key, value in kwargs.items():
            if hasattr(package, key):
                setattr(package, key, value)
        db.commit()
        db.refresh(package)
    return package


def delete_package(db: Session, package_id: str) -> bool:
    """Delete a package (cascades to chainages, lanes, videos)"""
    package = get_package(db, package_id)
    if package:
        db.delete(package)
        db.commit()
        return True
    return False


# ==================== Chainage CRUD ====================


def create_chainage(
    db: Session,
    package_id: str,
    segment_name: str,
    chainage_start_km: float,
    chainage_end_km: float,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
) -> Chainage:
    """Create a new chainage in a package"""
    chainage = Chainage(
        id=str(uuid.uuid4()),
        package_id=package_id,
        segment_name=segment_name,
        chainage_start_km=chainage_start_km,
        chainage_end_km=chainage_end_km,
        start_lat=start_lat,
        start_lng=start_lng,
        end_lat=end_lat,
        end_lng=end_lng,
    )
    db.add(chainage)
    db.commit()
    db.refresh(chainage)
    return chainage


def get_chainage(db: Session, chainage_id: str) -> Optional[Chainage]:
    """Get a chainage by ID"""
    return db.query(Chainage).filter(Chainage.id == chainage_id).first()


def list_chainages(
    db: Session, package_id: Optional[str] = None, skip: int = 0, limit: int = 100
) -> List[Chainage]:
    """Get all chainages, optionally filtered by package"""
    query = db.query(Chainage)
    if package_id:
        query = query.filter(Chainage.package_id == package_id)
    return (
        query.order_by(Chainage.chainage_start_km).offset(skip).limit(limit).all()
    )


def update_chainage(db: Session, chainage_id: str, **kwargs) -> Optional[Chainage]:
    """Update chainage fields"""
    chainage = get_chainage(db, chainage_id)
    if chainage:
        for key, value in kwargs.items():
            if hasattr(chainage, key):
                setattr(chainage, key, value)
        db.commit()
        db.refresh(chainage)
    return chainage


def delete_chainage(db: Session, chainage_id: str) -> bool:
    """Delete a chainage (cascades to lanes, videos)"""
    chainage = get_chainage(db, chainage_id)
    if chainage:
        db.delete(chainage)
        db.commit()
        return True
    return False


# ==================== Chainage Lookup ====================


def find_chainage_by_gps(
    db: Session, lat: float, lng: float, package_id: Optional[str] = None
) -> Optional[Chainage]:
    """
    Find chainage that contains the given GPS coordinates.
    Uses bounding box matching.

    Args:
        db: Database session
        lat: Latitude coordinate
        lng: Longitude coordinate
        package_id: Optional package filter for faster lookup

    Returns:
        Chainage if found, None otherwise
    """
    query = db.query(Chainage).filter(
        or_(
            and_(Chainage.start_lat <= lat, Chainage.end_lat >= lat),
            and_(Chainage.end_lat <= lat, Chainage.start_lat >= lat),
        ),
        or_(
            and_(Chainage.start_lng <= lng, Chainage.end_lng >= lng),
            and_(Chainage.end_lng <= lng, Chainage.start_lng >= lng),
        ),
    )

    if package_id:
        query = query.filter(Chainage.package_id == package_id)

    return query.first()


# ==================== Lane CRUD ====================


def create_lane(
    db: Session,
    chainage_id: str,
    lane_code: str,
    lane_type: Optional[str] = None,
    direction: Optional[str] = None,
) -> Lane:
    """Create a new lane in a chainage"""
    lane = Lane(
        id=str(uuid.uuid4()),
        chainage_id=chainage_id,
        lane_code=lane_code,
        lane_type=lane_type,
        direction=direction,
    )
    db.add(lane)
    db.commit()
    db.refresh(lane)
    return lane


def get_lane(db: Session, lane_id: str) -> Optional[Lane]:
    """Get a lane by ID"""
    return db.query(Lane).filter(Lane.id == lane_id).first()


def list_lanes(
    db: Session, chainage_id: Optional[str] = None, skip: int = 0, limit: int = 100
) -> List[Lane]:
    """Get all lanes, optionally filtered by chainage"""
    query = db.query(Lane)
    if chainage_id:
        query = query.filter(Lane.chainage_id == chainage_id)
    return query.order_by(Lane.lane_code).offset(skip).limit(limit).all()


def update_lane(db: Session, lane_id: str, **kwargs) -> Optional[Lane]:
    """Update lane fields"""
    lane = get_lane(db, lane_id)
    if lane:
        for key, value in kwargs.items():
            if hasattr(lane, key):
                setattr(lane, key, value)
        db.commit()
        db.refresh(lane)
    return lane


def delete_lane(db: Session, lane_id: str) -> bool:
    """Delete a lane (cascades to videos)"""
    lane = get_lane(db, lane_id)
    if lane:
        db.delete(lane)
        db.commit()
        return True
    return False
