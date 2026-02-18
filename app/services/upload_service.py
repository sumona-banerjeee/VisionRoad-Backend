from pathlib import Path
from fastapi import UploadFile, HTTPException, Depends
import shutil
import uuid
import asyncio
import logging
from enum import Enum
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db import crud
from app.models.processing import ProcessingStatusEnum
from app.services.video_processor import VideoProcessor
from app.services.signboard_detector import SignBoardDetector
from app.services.pot_sign_detector import PotSignDetector
from app.core.config import processing_status, UPLOAD_DIR
from app.ws.websocket_manager import manager

logger = logging.getLogger(__name__)


class DetectionType(str, Enum):
    POTHOLE_DETECTION = "pothole-detection"
    SIGN_BOARD_DETECTION = "sign-board-detection"
    POT_SIGN_DETECTION = "pot-sign-detection"


class UploadService:
    def __init__(self):
        self._pothole_processor = None
        self._signboard_processor = None
        self._pot_sign_processor = None

    def _get_processor(self, detection_type: DetectionType):
        """Lazy-load the detector only when first requested"""
        if detection_type == DetectionType.POTHOLE_DETECTION:
            if self._pothole_processor is None:
                logger.info("Loading pothole detector...")
                self._pothole_processor = VideoProcessor()
            return self._pothole_processor
        elif detection_type == DetectionType.SIGN_BOARD_DETECTION:
            if self._signboard_processor is None:
                logger.info("Loading signboard detector...")
                self._signboard_processor = SignBoardDetector()
            return self._signboard_processor
        elif detection_type == DetectionType.POT_SIGN_DETECTION:
            if self._pot_sign_processor is None:
                logger.info("Loading pot-sign detector...")
                self._pot_sign_processor = PotSignDetector()
            return self._pot_sign_processor
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid detection type: {detection_type}"
            )

    async def upload_video(
        self,
        file: UploadFile,
        json_file: UploadFile,
        detection_type: DetectionType,
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

        # Save uploaded video file
        file_extension = Path(file.filename).suffix
        video_path = UPLOAD_DIR / f"{video_id}{file_extension}"
        # save json file
        json_file_extension = Path(json_file.filename).suffix
        json_path = UPLOAD_DIR / f"{video_id}{json_file_extension}"

        try:
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            with open(json_path, "wb") as buffer:
                shutil.copyfileobj(json_file.file, buffer)

            logger.info(f"Video uploaded: {video_id} - {file.filename}")
            logger.info(f"JSON file uploaded: {video_id} - {json_file.filename}")

            # Create video record in database
            video = crud.create_video(
                db=db,
                video_id=video_id,
                filename=file.filename,
                original_path=str(video_path),
                json_file_path=str(json_path),
                detection_type=detection_type,
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
        processor = self._get_processor(detection_type)

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
            "message": "Video uploaded successfully. Processing started.",
            "status": "queued",
        }
