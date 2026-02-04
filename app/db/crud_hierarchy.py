"""CRUD operations for Project, Package, and Location models"""

from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional, List
from app.models.project import Project
from app.models.package import Package
from app.models.location import Location
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
    """Delete a project (cascades to packages, locations)"""
    project = get_project(db, project_id)
    if project:
        db.delete(project)
        db.commit()
        return True
    return False


# ==================== Package CRUD ====================


def create_package(
    db: Session, project_id: str, name: str, region: Optional[str] = None
) -> Package:
    """Create a new package in a project"""
    package = Package(
        id=str(uuid.uuid4()), project_id=project_id, name=name, region=region
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
    """Delete a package (cascades to locations)"""
    package = get_package(db, package_id)
    if package:
        db.delete(package)
        db.commit()
        return True
    return False


# ==================== Location CRUD ====================


def create_location(
    db: Session,
    package_id: str,
    segment_name: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    chainage_start_km: Optional[float] = None,
    chainage_end_km: Optional[float] = None,
) -> Location:
    """Create a new location in a package"""
    location = Location(
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
    db.add(location)
    db.commit()
    db.refresh(location)
    return location


def get_location(db: Session, location_id: str) -> Optional[Location]:
    """Get a location by ID"""
    return db.query(Location).filter(Location.id == location_id).first()


def list_locations(
    db: Session, package_id: Optional[str] = None, skip: int = 0, limit: int = 100
) -> List[Location]:
    """Get all locations, optionally filtered by package"""
    query = db.query(Location)
    if package_id:
        query = query.filter(Location.package_id == package_id)
    return query.order_by(desc(Location.created_at)).offset(skip).limit(limit).all()


def update_location(db: Session, location_id: str, **kwargs) -> Optional[Location]:
    """Update location fields"""
    location = get_location(db, location_id)
    if location:
        for key, value in kwargs.items():
            if hasattr(location, key):
                setattr(location, key, value)
        db.commit()
        db.refresh(location)
    return location


def delete_location(db: Session, location_id: str) -> bool:
    """Delete a location (cascades to videos)"""
    location = get_location(db, location_id)
    if location:
        db.delete(location)
        db.commit()
        return True
    return False


# ==================== Location Lookup ====================


def find_location_by_gps(
    db: Session, lat: float, lng: float, package_id: Optional[str] = None
) -> Optional[Location]:
    """
    Find location that contains the given GPS coordinates.
    Uses bounding box matching.

    Args:
        db: Database session
        lat: Latitude coordinate
        lng: Longitude coordinate
        package_id: Optional package filter for faster lookup

    Returns:
        Location if found, None otherwise
    """
    query = db.query(Location).filter(
        Location.start_lat <= lat,
        Location.end_lat >= lat,
        Location.start_lng <= lng,
        Location.end_lng >= lng,
    )

    if package_id:
        query = query.filter(Location.package_id == package_id)

    return query.first()
