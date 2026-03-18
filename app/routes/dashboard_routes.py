"""Dashboard home route — single endpoint for the projects overview page"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, case

from app.db.database import get_db
from app.models.project import Project
from app.models.package import Package
from app.models.chainage import Chainage
from app.models.detection import Detection
from app.models.video import Video

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/home")
async def get_dashboard_home(db: Session = Depends(get_db)):
    """
    Return all data needed for the Road Infrastructure Projects home page.

    Response includes:
    - total_projects, total_detections, last_scan (global)
    - per-project: detection breakdown by class_name, package sub-totals,
      last scan date
    """

    # ── 1. All projects ──────────────────────────────────────────────────
    projects = db.query(Project).order_by(Project.created_at).all()
    total_projects = len(projects)

    # ── 2. Global detection total ────────────────────────────────────────
    total_detections: int = db.query(func.count(Detection.id)).scalar() or 0

    # ── 3. Per-project detection counts grouped by class_name ────────────
    proj_class_counts = (
        db.query(
            Detection.project_id,
            Detection.class_name,
            func.count(Detection.id).label("cnt"),
        )
        .group_by(Detection.project_id, Detection.class_name)
        .all()
    )
    # Build a nested dict:  { project_id: { class_name: count } }
    det_by_project: dict[str, dict[str, int]] = {}
    for project_id, class_name, cnt in proj_class_counts:
        det_by_project.setdefault(project_id, {})[class_name] = cnt

    # ── 4. Per-package detection totals ──────────────────────────────────
    pkg_counts = (
        db.query(
            Detection.package_id,
            func.count(Detection.id).label("cnt"),
        )
        .group_by(Detection.package_id)
        .all()
    )
    det_by_package: dict[str, int] = {
        pkg_id: cnt for pkg_id, cnt in pkg_counts if pkg_id
    }

    # ── 5. All packages keyed by project_id ──────────────────────────────
    all_packages = db.query(Package).order_by(Package.created_at).all()
    pkgs_by_project: dict[str, list] = {}
    for pkg in all_packages:
        pkgs_by_project.setdefault(pkg.project_id, []).append(pkg)

    # ── 6. Last scan (most recent video upload_at) per project ───────────
    last_scan_rows = (
        db.query(
            Package.project_id,
            func.max(Video.uploaded_at).label("last_upload"),
        )
        .join(Chainage, Chainage.package_id == Package.id)
        .join(Video, Video.chainage_id == Chainage.id)
        .group_by(Package.project_id)
        .all()
    )
    last_scan_by_project: dict[str, str | None] = {
        row.project_id: row.last_upload.isoformat() if row.last_upload else None
        for row in last_scan_rows
    }

    # ── 7. Global last scan ──────────────────────────────────────────────
    global_last_scan_row = (
        db.query(
            Video.uploaded_at,
            Package.project_id,
        )
        .join(Chainage, Video.chainage_id == Chainage.id)
        .join(Package, Chainage.package_id == Package.id)
        .order_by(Video.uploaded_at.desc())
        .first()
    )

    global_last_scan = None
    if global_last_scan_row:
        # Find the project name for the most recently scanned project
        last_project = (
            db.query(Project.name)
            .filter(Project.id == global_last_scan_row.project_id)
            .scalar()
        )
        global_last_scan = {
            "date": global_last_scan_row.uploaded_at.isoformat()
            if global_last_scan_row.uploaded_at
            else None,
            "project_name": last_project,
        }

    # ── 8. Assemble per-project response ─────────────────────────────────
    project_cards = []
    for proj in projects:
        class_counts = det_by_project.get(proj.id, {})
        proj_total = sum(class_counts.values())

        packages_list = []
        for pkg in pkgs_by_project.get(proj.id, []):
            packages_list.append(
                {
                    "id": pkg.id,
                    "name": pkg.name,
                    "region": pkg.region,
                    "total_detections": det_by_package.get(pkg.id, 0),
                }
            )

        project_cards.append(
            {
                "id": proj.id,
                "name": proj.name,
                "corridor_name": proj.corridor_name,
                "state": proj.state,
                "detections": class_counts,
                "total_detections": proj_total,
                "packages": packages_list,
                "last_scan": last_scan_by_project.get(proj.id),
            }
        )

    return {
        "total_projects": total_projects,
        "total_detections": total_detections,
        "last_scan": global_last_scan,
        "projects": project_cards,
    }
