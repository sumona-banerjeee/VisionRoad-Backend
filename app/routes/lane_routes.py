"""API routes for Lane management"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List
from app.db.database import get_db
from app.db import crud_hierarchy


router = APIRouter(prefix="/lanes", tags=["Lanes"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class LaneCreate(BaseModel):
    chainage_id: str = Field(..., min_length=36, max_length=36)
    lane_code: str = Field(
        ..., min_length=1, max_length=20,
        description="e.g. LHS, RHS, UP, DOWN"
    )
    lane_type: Optional[str] = Field(
        None, max_length=50,
        description="e.g. driving, shoulder, service"
    )
    direction: Optional[str] = Field(None, max_length=100)


class LaneUpdate(BaseModel):
    lane_code: Optional[str] = Field(None, min_length=1, max_length=20)
    lane_type: Optional[str] = Field(None, max_length=50)
    direction: Optional[str] = Field(None, max_length=100)


class LaneResponse(BaseModel):
    id: str
    chainage_id: str
    lane_code: str
    lane_type: Optional[str]
    direction: Optional[str]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=LaneResponse, status_code=201)
async def create_lane(lane: LaneCreate, db: Session = Depends(get_db)):
    """Create a new lane in a chainage"""
    # Verify chainage exists
    chainage = crud_hierarchy.get_chainage(db, lane.chainage_id)
    if not chainage:
        raise HTTPException(status_code=404, detail="Chainage not found")

    new_lane = crud_hierarchy.create_lane(
        db=db,
        chainage_id=lane.chainage_id,
        lane_code=lane.lane_code.upper(),
        lane_type=lane.lane_type,
        direction=lane.direction,
    )
    return _to_response(new_lane)


@router.get("/", response_model=List[LaneResponse])
async def list_lanes(
    chainage_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List all lanes, optionally filtered by chainage"""
    lanes = crud_hierarchy.list_lanes(db, chainage_id=chainage_id, skip=skip, limit=limit)
    return [_to_response(l) for l in lanes]


@router.get("/{lane_id}", response_model=LaneResponse)
async def get_lane(lane_id: str, db: Session = Depends(get_db)):
    """Get a specific lane by ID"""
    lane = crud_hierarchy.get_lane(db, lane_id)
    if not lane:
        raise HTTPException(status_code=404, detail="Lane not found")
    return _to_response(lane)


@router.put("/{lane_id}", response_model=LaneResponse)
async def update_lane(
    lane_id: str, lane_data: LaneUpdate, db: Session = Depends(get_db)
):
    """Update a lane"""
    update_dict = lane_data.model_dump(exclude_unset=True)
    if "lane_code" in update_dict:
        update_dict["lane_code"] = update_dict["lane_code"].upper()

    updated = crud_hierarchy.update_lane(db, lane_id, **update_dict)
    if not updated:
        raise HTTPException(status_code=404, detail="Lane not found")
    return _to_response(updated)


@router.delete("/{lane_id}", status_code=204)
async def delete_lane(lane_id: str, db: Session = Depends(get_db)):
    """Delete a lane (cascades to videos)"""
    success = crud_hierarchy.delete_lane(db, lane_id)
    if not success:
        raise HTTPException(status_code=404, detail="Lane not found")
    return None


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_response(l) -> LaneResponse:
    return LaneResponse(
        id=l.id,
        chainage_id=l.chainage_id,
        lane_code=l.lane_code,
        lane_type=l.lane_type,
        direction=l.direction,
        created_at=l.created_at.isoformat(),
        updated_at=l.updated_at.isoformat(),
    )
