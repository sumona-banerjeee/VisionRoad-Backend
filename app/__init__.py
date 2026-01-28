from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.upload_process_routes import router as upload_router

def create_app():
    app = FastAPI(title="Pothole Detection API", version="1.0.0")

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
            "message": "Pothole Detection API",
            "version": "1.0.0",
            "endpoints": {
                "upload": "/api/upload",
                "status": "/api/status/{video_id}",
                "results": "/api/results/{video_id}",
                "websocket": "/ws/{video_id}",
                "list_videos": "/api/videos"
            }
        }
    # Include routers
    app.include_router(upload_router, prefix=f"{api_prefix}", tags=["Detection"])

    return app

    