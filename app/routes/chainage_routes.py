"""API routes for Chainage management (replaces location_routes.py)"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List
from app.db.database import get_db
from app.db import crud_hierarchy
from app.services.chainage_mapper import validate_gps_bounds
from app.models.chainage import DirectionEnum


router = APIRouter(prefix="/chainages", tags=["Chainages"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChainageCreate(BaseModel):
    package_id: str = Field(..., min_length=36, max_length=36)
    segment_name: str = Field(..., min_length=1, max_length=255)
    chainage_start_km: float = Field(..., ge=0, description="Absolute NHAI start KM")
    chainage_end_km: float = Field(..., ge=0, description="Absolute NHAI end KM")
    start_lat: float = Field(..., ge=-90, le=90)
    start_lng: float = Field(..., ge=-180, le=180)
    end_lat: float = Field(..., ge=-90, le=90)
    end_lng: float = Field(..., ge=-180, le=180)
    direction: DirectionEnum = Field(..., description="Direction of travel: UP or DOWN")


class ChainageUpdate(BaseModel):
    segment_name: Optional[str] = Field(None, min_length=1, max_length=255)
    chainage_start_km: Optional[float] = Field(None, ge=0)
    chainage_end_km: Optional[float] = Field(None, ge=0)
    start_lat: Optional[float] = Field(None, ge=-90, le=90)
    start_lng: Optional[float] = Field(None, ge=-180, le=180)
    end_lat: Optional[float] = Field(None, ge=-90, le=90)
    end_lng: Optional[float] = Field(None, ge=-180, le=180)
    direction: Optional[DirectionEnum] = Field(None, description="Direction of travel: UP or DOWN")


class ChainageResponse(BaseModel):
    id: str
    package_id: str
    segment_name: str
    chainage_start_km: float
    chainage_end_km: float
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    direction: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class PaginatedChainageResponse(BaseModel):
    items: List[ChainageResponse]
    totalItems: int


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=ChainageResponse, status_code=201)
async def create_chainage(chainage: ChainageCreate, db: Session = Depends(get_db)):
    """Create a new chainage in a package"""
    # Verify package exists
    package = crud_hierarchy.get_package(db, chainage.package_id)
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")

    # Validate KM range
    if chainage.chainage_end_km <= chainage.chainage_start_km:
        raise HTTPException(
            status_code=400,
            detail="chainage_end_km must be greater than chainage_start_km",
        )

    # Validate GPS bounds
    if not validate_gps_bounds(
        chainage.start_lat, chainage.start_lng, chainage.end_lat, chainage.end_lng
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid GPS coordinates: must be within valid lat/lng ranges",
        )

    new_chainage = crud_hierarchy.create_chainage(
        db=db,
        package_id=chainage.package_id,
        segment_name=chainage.segment_name,
        chainage_start_km=chainage.chainage_start_km,
        chainage_end_km=chainage.chainage_end_km,
        start_lat=chainage.start_lat,
        start_lng=chainage.start_lng,
        end_lat=chainage.end_lat,
        end_lng=chainage.end_lng,
        direction=chainage.direction.value,
    )
    return _to_response(new_chainage)


@router.get("/", response_model=PaginatedChainageResponse)
async def list_chainages(
    package_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List all chainages, optionally filtered by package"""
    chainages = crud_hierarchy.list_chainages(
        db, package_id=package_id, skip=skip, limit=limit
    )
    total = crud_hierarchy.count_chainages(db,package_id=package_id)
    return PaginatedChainageResponse(
        items=[_to_response(c) for c in chainages],
        totalItems=total,
    )


@router.get("/{chainage_id}", response_model=ChainageResponse)
async def get_chainage(chainage_id: str, db: Session = Depends(get_db)):
    """Get a specific chainage by ID"""
    chainage = crud_hierarchy.get_chainage(db, chainage_id)
    if not chainage:
        raise HTTPException(status_code=404, detail="Chainage not found")
    return _to_response(chainage)


@router.put("/{chainage_id}", response_model=ChainageResponse)
async def update_chainage(
    chainage_id: str, chainage_data: ChainageUpdate, db: Session = Depends(get_db)
):
    """Update a chainage"""
    update_dict = chainage_data.model_dump(exclude_unset=True)

    # Validate GPS bounds if any GPS field is being updated
    if any(k in update_dict for k in ["start_lat", "start_lng", "end_lat", "end_lng"]):
        current = crud_hierarchy.get_chainage(db, chainage_id)
        if not current:
            raise HTTPException(status_code=404, detail="Chainage not found")
        start_lat = update_dict.get("start_lat", current.start_lat)
        start_lng = update_dict.get("start_lng", current.start_lng)
        end_lat = update_dict.get("end_lat", current.end_lat)
        end_lng = update_dict.get("end_lng", current.end_lng)
        if not validate_gps_bounds(start_lat, start_lng, end_lat, end_lng):
            raise HTTPException(
                status_code=400,
                detail="Invalid GPS coordinates: must be within valid lat/lng ranges",
            )

    updated = crud_hierarchy.update_chainage(db, chainage_id, **update_dict)
    if not updated:
        raise HTTPException(status_code=404, detail="Chainage not found")
    return _to_response(updated)


@router.delete("/{chainage_id}", status_code=204)
async def delete_chainage(chainage_id: str, db: Session = Depends(get_db)):
    """Delete a chainage (cascades to videos)"""
    success = crud_hierarchy.delete_chainage(db, chainage_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chainage not found")
    return None


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_response(c) -> ChainageResponse:
    return ChainageResponse(
        id=c.id,
        package_id=c.package_id,
        segment_name=c.segment_name,
        chainage_start_km=c.chainage_start_km,
        chainage_end_km=c.chainage_end_km,
        start_lat=c.start_lat,
        start_lng=c.start_lng,
        end_lat=c.end_lat,
        end_lng=c.end_lng,
        direction=c.direction,
        created_at=c.created_at.isoformat(),
        updated_at=c.updated_at.isoformat(),
    )
