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
        """Send message to WebSocket"""
        # Log state changes regardless of whether a client is connected
        msg_type = message.get("type")
        if msg_type == "complete":
            logger.info(
                f"WS [{video_id}] → complete | "
                f"detections={message.get('counts', {}).get('total_road_damage', '?')} "
            )
        elif msg_type == "error":
            logger.error(
                f"WS [{video_id}] → error | {message.get('message', 'unknown')}"
            )
        elif msg_type == "progress":
            pct = message.get("progress", -1)
            # Log only at milestones to avoid flooding — every 5% would be ~20 lines/video
            if pct in (0, 25, 50, 75, 100):
                logger.info(
                    f"WS [{video_id}] → progress {pct}% | job={message.get('job_status')}"
                )

        if video_id in self.active_connections:
            try:
                await self.active_connections[video_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending WS message to {video_id}: {str(e)}")
                self.disconnect(video_id)
        elif msg_type not in ("progress", "heartbeat"):
            logger.debug(
                f"WS [{video_id}] → no active connection, message not delivered"
            )

    async def broadcast(self, message: dict):
        """Broadcast message to all connected WebSockets"""
        for video_id in list(self.active_connections.keys()):
            await self.send_message(video_id, message)


# Global manager instance
manager = ConnectionManager()
