from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes.upload_process_routes import router as upload_router
from app.routes.project_routes import router as project_router
from app.routes.package_routes import router as package_router
from app.routes.location_routes import router as location_router
from app.routes.summary_routes import router as summary_router
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup: Initialize database
    init_db()
    print("✓ Database initialized")
    yield
    # Shutdown: cleanup if needed
    print("✓ Application shutdown")


def create_app():
    app = FastAPI(title="VisionRoad API", version="1.0.0", lifespan=lifespan)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_prefix = "/api/v1"

    @app.get("/")
    async def root():
        return {
            "message": "VisionRoad API",
            "version": "1.0.0",
            "endpoints": {
                "projects": "/api/v1/projects",
                "packages": "/api/v1/packages",
                "locations": "/api/v1/locations",
                "summary": "/api/v1/summary",
                "upload": "/api/upload",
                "status": "/api/status/{video_id}",
                "results": "/api/results/{video_id}",
                "websocket": "/ws/{video_id}",
                "list_videos": "/api/videos",
            },
        }

    # Include routers
    app.include_router(project_router, prefix=f"{api_prefix}", tags=["Projects"])
    app.include_router(package_router, prefix=f"{api_prefix}", tags=["Packages"])
    app.include_router(location_router, prefix=f"{api_prefix}", tags=["Locations"])
    app.include_router(summary_router, prefix=f"{api_prefix}", tags=["Analytics"])
    app.include_router(upload_router, prefix=f"{api_prefix}", tags=["Detection"])

    return app
