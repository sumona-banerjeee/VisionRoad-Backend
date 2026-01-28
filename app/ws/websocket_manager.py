# app/ws/websocket_manager.py

from fastapi import WebSocket
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, video_id: str, websocket: WebSocket):
        """Accept and store WebSocket connection"""
        await websocket.accept()
        self.active_connections[video_id] = websocket
        logger.info(f"WebSocket connected for video: {video_id}")

    def disconnect(self, video_id: str):
        """Remove WebSocket connection"""
        if video_id in self.active_connections:
            del self.active_connections[video_id]
            logger.info(f"WebSocket disconnected for video: {video_id}")

    async def send_message(self, video_id: str, message: dict):
        """Send message to specific video's WebSocket"""
        if video_id in self.active_connections:
            try:
                await self.active_connections[video_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {video_id}: {str(e)}")
                self.disconnect(video_id)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected WebSockets"""
        for video_id in list(self.active_connections.keys()):
            await self.send_message(video_id, message)


# Global manager instance
manager = ConnectionManager()