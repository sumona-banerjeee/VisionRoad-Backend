from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends,
)
from sqlalchemy.orm import Session
import asyncio
import json
from app.services.upload_service import UploadService, DetectionMode
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.db.database import get_db


router = APIRouter()

upload_service = UploadService()


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    json_file: UploadFile = File(...),
    detection_mode: DetectionMode = Form(DetectionMode.YOLO_VL),
    speed_kmh: int = Form(30),
    chainage_id: str = Form(None),
    db: Session = Depends(get_db),
):
    """Upload video and start background processing.

    detection_mode options (road hazard):
      - yolo             : YOLO-only (fast, no VL verification)
      - yolo_vl          : YOLO + VL verification (default)
      - sam3             : YOLO + SAM3 verification
      - yoloe            : YOLOE open-vocabulary detection (text-prompted)
      - yoloe_trained_vl : YOLOE fine-tuned + VL verification (all classes except drain)
      - combined         : yoloe_trained_vl for all classes + YOLO for drain_issue only
                           (single-pass dual-model; best overall coverage)

    Culvert detection:
      - culvert_detection : Culvert-specific model (culvert_best.pt) + BotSORT tracking.
                            When this mode is selected the detection_mode field is
                            ignored — CulvertDetector is always used.
                            Classes: good_culvert · defective_culvert
    """
    return await upload_service.upload_video(
        file=file,
        json_file=json_file,
        detection_mode=detection_mode,
        speed_kmh=speed_kmh,
        chainage_id=chainage_id,
        db=db,
    )


@router.get("/status/{video_id}")
async def get_status(video_id: str):
    """Get current processing status"""
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video ID not found")
    return processing_status[video_id]


@router.get("/results/{video_id}")
async def get_results(video_id: str):
    """Get detection results for a processed video"""
    if video_id not in detection_results:
        result_file = RESULTS_DIR / f"{video_id}.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                detection_results[video_id] = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Results not found")
    return detection_results[video_id]


@router.get("/videos")
async def list_videos():
    """List all processed videos"""
    videos = []
    for video_id, status in processing_status.items():
        video_info = {
            "video_id": video_id,
            "status": status.get("status", "unknown"),
            "progress": status.get("progress", 0),
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
