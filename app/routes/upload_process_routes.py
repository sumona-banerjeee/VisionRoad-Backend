from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends
)
from sqlalchemy.orm import Session
import asyncio
from app.services.upload_service import UploadService, DetectionType
from app.services.video_processor import VideoProcessor
from app.ws.websocket_manager import manager
from app.core.storage import processing_status, detection_results
from app.db.database import get_db


router = APIRouter()

upload_service = UploadService()
video_processor = VideoProcessor()


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    json_file: UploadFile = File(...),
    detection_type: DetectionType = Form(DetectionType.POTHOLE_DETECTION),
    speed_kmh: int = Form(30),
    db: Session = Depends(get_db)
):
    """Upload video and start background processing"""
    return await upload_service.upload_video(
        file=file,
        json_file=json_file,
        detection_type=detection_type,
        speed_kmh=speed_kmh,
        db=db
    )


@router.get("/status/{video_id}")
async def get_status(video_id: str):
    """Get current processing status"""
    return await video_processor.get_status(video_id)


@router.get("/results/{video_id}")
async def get_results(video_id: str):
    """Get detection results for a processed video"""
    return await video_processor.get_results(video_id)


@router.get("/videos")
async def list_videos():
    """List all processed videos"""
    videos = []
    for video_id, status in processing_status.items():
        video_info = {
            "video_id": video_id,
            "status": status["status"],
            "progress": status["progress"],
        }

        if video_id in detection_results:
            video_info["summary"] = detection_results[video_id]["summary"]

        videos.append(video_info)

    return {"videos": videos}


@router.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    """WebSocket for real-time processing updates"""

    await manager.connect(video_id, websocket)

    try:
        # Send initial status if available
        if video_id in processing_status:
            await websocket.send_json({"type": "status", **processing_status[video_id]})

        # Keep connection alive and wait for processing to complete
        while True:
            # Check if processing is done
            if video_id in processing_status:
                status = processing_status[video_id]["status"]
                if status in ["completed", "error"]:
                    # Send final status and close gracefully
                    await websocket.send_json(
                        {"type": "status", **processing_status[video_id]}
                    )
                    break

            # Keep connection alive with a ping/pong mechanism
            try:
                # Wait for any message from client (like ping) with timeout
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a heartbeat to keep connection alive
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except:
                    break
            except:
                break

            # Small delay to prevent busy loop
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error for {video_id}: {e}")
    finally:
        manager.disconnect(video_id)
