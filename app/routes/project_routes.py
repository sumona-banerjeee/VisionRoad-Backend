"""API routes for Project management"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List
from app.db.database import get_db
from app.db import crud_hierarchy


router = APIRouter(prefix="/projects", tags=["Projects"])


# Request/Response schemas
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    state: Optional[str] = Field(None, max_length=255)
    corridor_name: Optional[str] = Field(None, max_length=255)
    start_lat: Optional[float] = Field(None, ge=-90, le=90)
    start_lng: Optional[float] = Field(None, ge=-180, le=180)
    end_lat: Optional[float] = Field(None, ge=-90, le=90)
    end_lng: Optional[float] = Field(None, ge=-180, le=180)


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    state: Optional[str] = Field(None, max_length=255)
    corridor_name: Optional[str] = Field(None, max_length=255)
    start_lat: Optional[float] = Field(None, ge=-90, le=90)
    start_lng: Optional[float] = Field(None, ge=-180, le=180)
    end_lat: Optional[float] = Field(None, ge=-90, le=90)
    end_lng: Optional[float] = Field(None, ge=-180, le=180)


class ProjectResponse(BaseModel):
    id: str
    name: str
    state: Optional[str]
    corridor_name: Optional[str]
    start_lat: Optional[float]
    start_lng: Optional[float]
    end_lat: Optional[float]
    end_lng: Optional[float]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class PaginatedProjectResponse(BaseModel):
    items: List[ProjectResponse]
    totalItems: int


# Routes
@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project"""
    new_project = crud_hierarchy.create_project(
        db=db,
        name=project.name,
        state=project.state,
        corridor_name=project.corridor_name,
        start_lat=project.start_lat,
        start_lng=project.start_lng,
        end_lat=project.end_lat,
        end_lng=project.end_lng,
    )
    return ProjectResponse(
        id=new_project.id,
        name=new_project.name,
        state=new_project.state,
        corridor_name=new_project.corridor_name,
        start_lat=new_project.start_lat,
        start_lng=new_project.start_lng,
        end_lat=new_project.end_lat,
        end_lng=new_project.end_lng,
        created_at=new_project.created_at.isoformat(),
        updated_at=new_project.updated_at.isoformat(),
    )


@router.get("/", response_model=PaginatedProjectResponse)
async def list_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all projects with total count for pagination"""
    projects = crud_hierarchy.list_projects(db, skip=skip, limit=limit)
    total = crud_hierarchy.count_projects(db)
    return PaginatedProjectResponse(
        items=[
            ProjectResponse(
                id=p.id,
                name=p.name,
                state=p.state,
                corridor_name=p.corridor_name,
                start_lat=p.start_lat,
                start_lng=p.start_lng,
                end_lat=p.end_lat,
                end_lng=p.end_lng,
                created_at=p.created_at.isoformat(),
                updated_at=p.updated_at.isoformat(),
            )
            for p in projects
        ],
        totalItems=total,
    )



@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get a specific project by ID"""
    project = crud_hierarchy.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectResponse(
        id=project.id,
        name=project.name,
        state=project.state,
        corridor_name=project.corridor_name,
        start_lat=project.start_lat,
        start_lng=project.start_lng,
        end_lat=project.end_lat,
        end_lng=project.end_lng,
        created_at=project.created_at.isoformat(),
        updated_at=project.updated_at.isoformat(),
    )


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str, project_data: ProjectUpdate, db: Session = Depends(get_db)
):
    """Update a project"""
    update_dict = project_data.model_dump(exclude_unset=True)
    updated_project = crud_hierarchy.update_project(db, project_id, **update_dict)

    if not updated_project:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectResponse(
        id=updated_project.id,
        name=updated_project.name,
        state=updated_project.state,
        corridor_name=updated_project.corridor_name,
        start_lat=updated_project.start_lat,
        start_lng=updated_project.start_lng,
        end_lat=updated_project.end_lat,
        end_lng=updated_project.end_lng,
        created_at=updated_project.created_at.isoformat(),
        updated_at=updated_project.updated_at.isoformat(),
    )


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a project (cascades to packages and locations)"""
    success = crud_hierarchy.delete_project(db, project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return None
