"""API routes for Package management"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List
from app.db.database import get_db
from app.db import crud_hierarchy


router = APIRouter(prefix="/packages", tags=["Packages"])


# Request/Response schemas
class PackageCreate(BaseModel):
    project_id: str = Field(..., min_length=36, max_length=36)
    name: str = Field(..., min_length=1, max_length=255)
    region: Optional[str] = Field(None, max_length=255)
    chainage_start_km: Optional[float] = Field(None, ge=0, description="Absolute NHAI start KM of this package")
    chainage_end_km: Optional[float] = Field(None, ge=0, description="Absolute NHAI end KM of this package")


class PackageUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    region: Optional[str] = Field(None, max_length=255)
    chainage_start_km: Optional[float] = Field(None, ge=0)
    chainage_end_km: Optional[float] = Field(None, ge=0)


class PackageResponse(BaseModel):
    id: str
    project_id: str
    name: str
    region: Optional[str]
    chainage_start_km: Optional[float]
    chainage_end_km: Optional[float]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class PaginatedPackageResponse(BaseModel):
    items: List[PackageResponse]
    totalItems: int


# Routes
@router.post("/", response_model=PackageResponse, status_code=201)
async def create_package(package: PackageCreate, db: Session = Depends(get_db)):
    """Create a new package in a project"""
    # Verify project exists
    project = crud_hierarchy.get_project(db, package.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    new_package = crud_hierarchy.create_package(
        db=db,
        project_id=package.project_id,
        name=package.name,
        region=package.region,
        chainage_start_km=package.chainage_start_km,
        chainage_end_km=package.chainage_end_km,
    )
    return _to_response(new_package)


@router.get("/", response_model=PaginatedPackageResponse)
async def list_packages(
    project_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List all packages, optionally filtered by project"""
    packages = crud_hierarchy.list_packages(
        db, project_id=project_id, skip=skip, limit=limit
    )
    total = crud_hierarchy.count_packages(db, project_id=project_id)

    return PaginatedPackageResponse(
        items=[_to_response(p) for p in packages],
        totalItems=total,
    )



@router.get("/{package_id}", response_model=PackageResponse)
async def get_package(package_id: str, db: Session = Depends(get_db)):
    """Get a specific package by ID"""
    package = crud_hierarchy.get_package(db, package_id)
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
    return _to_response(package)


@router.put("/{package_id}", response_model=PackageResponse)
async def update_package(
    package_id: str, package_data: PackageUpdate, db: Session = Depends(get_db)
):
    """Update a package"""
    update_dict = package_data.model_dump(exclude_unset=True)
    updated_package = crud_hierarchy.update_package(db, package_id, **update_dict)

    if not updated_package:
        raise HTTPException(status_code=404, detail="Package not found")

    return _to_response(updated_package)


@router.delete("/{package_id}", status_code=204)
async def delete_package(package_id: str, db: Session = Depends(get_db)):
    """Delete a package (cascades to chainages, lanes, videos)"""
    success = crud_hierarchy.delete_package(db, package_id)
    if not success:
        raise HTTPException(status_code=404, detail="Package not found")
    return None


def _to_response(p) -> PackageResponse:
    return PackageResponse(
        id=p.id,
        project_id=p.project_id,
        name=p.name,
        region=p.region,
        chainage_start_km=p.chainage_start_km,
        chainage_end_km=p.chainage_end_km,
        created_at=p.created_at.isoformat(),
        updated_at=p.updated_at.isoformat(),
    )
