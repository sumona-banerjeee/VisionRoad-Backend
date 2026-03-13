from pathlib import Path
from fastapi import UploadFile, HTTPException, Depends
import anyio
import uuid
import asyncio
import logging
from enum import Enum
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db import crud
from app.models.processing import ProcessingStatusEnum
from app.detectors.registry import DETECTOR_REGISTRY
from app.core.config import processing_status, UPLOAD_DIR
from app.ws.websocket_manager import manager

logger = logging.getLogger(__name__)


class DetectionMode(str, Enum):
    """
    Available detection modes.

    - YOLO     : YOLO-only inference (fast, no VL API calls)
    - YOLO_VL  : YOLO inference + async VL cross-verification
    - SAM3     : SAM3-based detection (not yet implemented — stub)
    """

    YOLO = "yolo"
    YOLO_VL = "yolo_vl"
    SAM3 = "sam3"
    YOLOE = "yoloe"


# Keep backward-compatible alias so other internal code can still import DetectionType
DetectionType = DetectionMode


_CHUNK = 1 << 20  # 1 MB (1048576 bytes) — yields event loop every chunk 


async def _save_upload_async(upload: UploadFile, dest: Path) -> None:
    """Stream an UploadFile to disk in 1 MB chunks without blocking the event loop."""
    async with await anyio.open_file(dest, "wb") as f:
        while True:
            chunk = await upload.read(_CHUNK)
            if not chunk:
                break
            await f.write(chunk)


class UploadService:
    def __init__(self):
        # Lazy-loaded detector instances, keyed by DetectionMode value
        self._processors: dict = {}

    def _get_processor(self, mode: DetectionMode):
        """Lazy-load the detector only when first requested via registry."""
        factory = DETECTOR_REGISTRY.get(mode.value)
        if factory is None:
            raise HTTPException(
                status_code=400, detail=f"Unknown detection mode: {mode}"
            )
        if mode.value not in self._processors:
            logger.info(f"Loading detector for mode: {mode.value}")
            self._processors[mode.value] = factory()
        return self._processors[mode.value]

    async def upload_video(
        self,
        file: UploadFile,
        json_file: UploadFile,
        detection_mode: DetectionMode,
        speed_kmh: int = 30,
        db: Session = Depends(get_db),
    ):
        """Upload video and start background processing"""

        # Validate file type
        if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            raise HTTPException(
                status_code=400, detail="Invalid file type. Please upload a video file."
            )
        if not json_file.filename.lower().endswith((".json")):
            raise HTTPException(
                status_code=400, detail="Invalid file type. Please upload a json file."
            )

        # Generate unique video ID
        video_id = str(uuid.uuid4())

        file_extension = Path(file.filename).suffix
        video_path = UPLOAD_DIR / f"{video_id}{file_extension}"
        
        json_file_extension = Path(json_file.filename).suffix
        json_path = UPLOAD_DIR / f"{video_id}{json_file_extension}"

        try:
            await _save_upload_async(file, video_path)
            await _save_upload_async(json_file, json_path)

            logger.info(f"Video uploaded: {video_id} - {file.filename}")
            logger.info(f"JSON file uploaded: {video_id} - {json_file.filename}")

            # Create video record in database
            video = crud.create_video(
                db=db,
                video_id=video_id,
                filename=file.filename,
                original_path=str(video_path),
                json_file_path=str(json_path),
                detection_type=detection_mode,
                speed_kmh=speed_kmh,
            )

            # Create initial processing status
            db_processing_status = crud.create_processing_status(
                db=db, video_id=video_id, status=ProcessingStatusEnum.PENDING
            )

        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save video file")

        # Initialize processing status
        processing_status[video_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Video uploaded, waiting to process...",
        }

        # Get processor (lazy-loaded on first use)
        processor = self._get_processor(detection_mode)

        # Start background processing
        asyncio.create_task(
            processor.process_video(
                video_id=video_id,
                video_path=str(video_path),
                json_path=str(json_path),
                speed_kmh=speed_kmh,
            )
        )

        # Give a brief moment for WebSocket to potentially connect
        await asyncio.sleep(0.1)

        # Send initial message to any connected WebSocket
        await manager.send_message(
            video_id,
            {
                "type": "status",
                "status": "queued",
                "progress": 0,
                "message": "Video uploaded, starting processing...",
            },
        )

        return {
            "video_id": video_id,
            "filename": file.filename,
            "detection_mode": detection_mode,
            "message": "Video uploaded successfully. Processing started.",
            "status": "queued",
        }
