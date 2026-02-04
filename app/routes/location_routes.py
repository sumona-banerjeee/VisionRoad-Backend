"""API routes for Location management"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List
from app.db.database import get_db
from app.db import crud_hierarchy
from app.services.location_mapper import validate_gps_bounds


router = APIRouter(prefix="/locations", tags=["Locations"])


# Request/Response schemas
class LocationCreate(BaseModel):
    package_id: str = Field(..., min_length=36, max_length=36)
    segment_name: str = Field(..., min_length=1, max_length=255)
    chainage_start_km: Optional[float] = Field(None, ge=0)
    chainage_end_km: Optional[float] = Field(None, ge=0)
    start_lat: float = Field(..., ge=-90, le=90)
    start_lng: float = Field(..., ge=-180, le=180)
    end_lat: float = Field(..., ge=-90, le=90)
    end_lng: float = Field(..., ge=-180, le=180)


class LocationUpdate(BaseModel):
    segment_name: Optional[str] = Field(None, min_length=1, max_length=255)
    chainage_start_km: Optional[float] = Field(None, ge=0)
    chainage_end_km: Optional[float] = Field(None, ge=0)
    start_lat: Optional[float] = Field(None, ge=-90, le=90)
    start_lng: Optional[float] = Field(None, ge=-180, le=180)
    end_lat: Optional[float] = Field(None, ge=-90, le=90)
    end_lng: Optional[float] = Field(None, ge=-180, le=180)


class LocationResponse(BaseModel):
    id: str
    package_id: str
    segment_name: str
    chainage_start_km: Optional[float]
    chainage_end_km: Optional[float]
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


# Routes
@router.post("/", response_model=LocationResponse, status_code=201)
async def create_location(location: LocationCreate, db: Session = Depends(get_db)):
    """Create a new location in a package"""
    # Verify package exists
    package = crud_hierarchy.get_package(db, location.package_id)
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")

    # Validate GPS bounds
    if not validate_gps_bounds(
        location.start_lat, location.start_lng, location.end_lat, location.end_lng
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid GPS coordinates: must be within valid lat/lng ranges",
        )

    new_location = crud_hierarchy.create_location(
        db=db,
        package_id=location.package_id,
        segment_name=location.segment_name,
        start_lat=location.start_lat,
        start_lng=location.start_lng,
        end_lat=location.end_lat,
        end_lng=location.end_lng,
        chainage_start_km=location.chainage_start_km,
        chainage_end_km=location.chainage_end_km,
    )
    return LocationResponse(
        id=new_location.id,
        package_id=new_location.package_id,
        segment_name=new_location.segment_name,
        chainage_start_km=new_location.chainage_start_km,
        chainage_end_km=new_location.chainage_end_km,
        start_lat=new_location.start_lat,
        start_lng=new_location.start_lng,
        end_lat=new_location.end_lat,
        end_lng=new_location.end_lng,
        created_at=new_location.created_at.isoformat(),
        updated_at=new_location.updated_at.isoformat(),
    )


@router.get("/", response_model=List[LocationResponse])
async def list_locations(
    package_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List all locations, optionally filtered by package"""
    locations = crud_hierarchy.list_locations(
        db, package_id=package_id, skip=skip, limit=limit
    )
    return [
        LocationResponse(
            id=loc.id,
            package_id=loc.package_id,
            segment_name=loc.segment_name,
            chainage_start_km=loc.chainage_start_km,
            chainage_end_km=loc.chainage_end_km,
            start_lat=loc.start_lat,
            start_lng=loc.start_lng,
            end_lat=loc.end_lat,
            end_lng=loc.end_lng,
            created_at=loc.created_at.isoformat(),
            updated_at=loc.updated_at.isoformat(),
        )
        for loc in locations
    ]


@router.get("/{location_id}", response_model=LocationResponse)
async def get_location(location_id: str, db: Session = Depends(get_db)):
    """Get a specific location by ID"""
    location = crud_hierarchy.get_location(db, location_id)
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    return LocationResponse(
        id=location.id,
        package_id=location.package_id,
        segment_name=location.segment_name,
        chainage_start_km=location.chainage_start_km,
        chainage_end_km=location.chainage_end_km,
        start_lat=location.start_lat,
        start_lng=location.start_lng,
        end_lat=location.end_lat,
        end_lng=location.end_lng,
        created_at=location.created_at.isoformat(),
        updated_at=location.updated_at.isoformat(),
    )


@router.put("/{location_id}", response_model=LocationResponse)
async def update_location(
    location_id: str, location_data: LocationUpdate, db: Session = Depends(get_db)
):
    """Update a location"""
    # If GPS bounds are being updated, validate them
    if any(
        k in location_data.model_dump(exclude_unset=True)
        for k in ["start_lat", "start_lng", "end_lat", "end_lng"]
    ):
        # Get current location to fill in missing values
        current_loc = crud_hierarchy.get_location(db, location_id)
        if not current_loc:
            raise HTTPException(status_code=404, detail="Location not found")

        update_dict = location_data.model_dump(exclude_unset=True)
        start_lat = update_dict.get("start_lat", current_loc.start_lat)
        start_lng = update_dict.get("start_lng", current_loc.start_lng)
        end_lat = update_dict.get("end_lat", current_loc.end_lat)
        end_lng = update_dict.get("end_lng", current_loc.end_lng)

        if not validate_gps_bounds(start_lat, start_lng, end_lat, end_lng):
            raise HTTPException(
                status_code=400,
                detail="Invalid GPS coordinates: must be within valid lat/lng ranges",
            )

    update_dict = location_data.model_dump(exclude_unset=True)
    updated_location = crud_hierarchy.update_location(db, location_id, **update_dict)

    if not updated_location:
        raise HTTPException(status_code=404, detail="Location not found")

    return LocationResponse(
        id=updated_location.id,
        package_id=updated_location.package_id,
        segment_name=updated_location.segment_name,
        chainage_start_km=updated_location.chainage_start_km,
        chainage_end_km=updated_location.chainage_end_km,
        start_lat=updated_location.start_lat,
        start_lng=updated_location.start_lng,
        end_lat=updated_location.end_lat,
        end_lng=updated_location.end_lng,
        created_at=updated_location.created_at.isoformat(),
        updated_at=updated_location.updated_at.isoformat(),
    )


@router.delete("/{location_id}", status_code=204)
async def delete_location(location_id: str, db: Session = Depends(get_db)):
    """Delete a location (cascades to videos)"""
    success = crud_hierarchy.delete_location(db, location_id)
    if not success:
        raise HTTPException(status_code=404, detail="Location not found")
    return None
