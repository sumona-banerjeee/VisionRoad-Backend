# app/services/upload_service.py

from pathlib import Path
from fastapi import UploadFile, HTTPException
import shutil
import uuid
import asyncio
import logging

from app.services.video_processor import VideoProcessor
from app.core.storage import processing_status
from app.ws.websocket_manager import manager

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class UploadService:
    def __init__(self):
        self.video_processor = VideoProcessor()

    async def upload_video(self, file: UploadFile, speed_kmh: int = 30):
        """Upload video and start background processing"""
        
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload a video file."
            )
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_extension = Path(file.filename).suffix
        video_path = UPLOAD_DIR / f"{video_id}{file_extension}"
        
        try:
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"Video uploaded: {video_id} - {file.filename}")
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save video file")
        
        # Initialize processing status
        processing_status[video_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Video uploaded, waiting to process..."
        }
        
        # Start background processing
        asyncio.create_task(
            self.video_processor.process_video(video_id, str(video_path), speed_kmh)
        )
        
        # Give a brief moment for WebSocket to potentially connect
        await asyncio.sleep(0.1)
        
        # Send initial message to any connected WebSocket
        await manager.send_message(video_id, {
            "type": "status",
            "status": "queued",
            "progress": 0,
            "message": "Video uploaded, starting processing..."
        })
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "message": "Video uploaded successfully. Processing started.",
            "status": "queued"
        }