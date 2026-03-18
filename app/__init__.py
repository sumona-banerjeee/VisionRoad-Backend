import time
import logging
import builtins
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes.upload_process_routes import router as upload_router
from app.routes.project_routes import router as project_router
from app.routes.package_routes import router as package_router
from app.routes.chainage_routes import router as chainage_router
from app.routes.lane_routes import router as lane_router
from app.routes.summary_routes import router as summary_router
from app.db.database import init_db
import app.core.config  # noqa: F401 — imported first to run load_dotenv() before any os.getenv() calls
from app.core.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    start_time = time.time()

    # Startup: Initialize logging first
    setup_logging()

    # Initialize database
    init_db()
    print("✓ Database initialized")

    elapsed = time.time() - start_time
    total = time.time() - getattr(builtins, "_boot_start", start_time)
    print(f"✓ Lifespan init in {elapsed:.2f}s | Total boot time: {total:.2f}s")
    yield
    # Shutdown: drain and close thread pools
    logger.info("Shutting down executor pools …")
    from app.detectors.base.base_detector import get_executor
    from app.detectors.yolo.detector import get_verify_executor

    get_executor().shutdown(wait=True, cancel_futures=False)
    get_verify_executor().shutdown(wait=True, cancel_futures=False)
    logger.info("✓ Executor pools shut down cleanly")
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
                "chainages": "/api/v1/chainages",
                "lanes": "/api/v1/lanes",
                "summary": "/api/v1/summary",
                "upload": "/api/v1/upload",
                "status": "/api/v1/status/{video_id}",
                "results": "/api/v1/results/{video_id}",
                "websocket": "/api/v1/ws/{video_id}",
                "list_videos": "/api/v1/videos",
            },
        }

    # Include routers
    app.include_router(project_router, prefix=f"{api_prefix}", tags=["Projects"])
    app.include_router(package_router, prefix=f"{api_prefix}", tags=["Packages"])
    app.include_router(chainage_router, prefix=f"{api_prefix}", tags=["Chainages"])
    app.include_router(lane_router, prefix=f"{api_prefix}", tags=["Lanes"])
    app.include_router(summary_router, prefix=f"{api_prefix}", tags=["Analytics"])
    app.include_router(upload_router, prefix=f"{api_prefix}", tags=["Detection"])

    return app
