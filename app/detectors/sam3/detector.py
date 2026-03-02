"""
SAM3 Detector — Stub Implementation

This detector is a placeholder for future SAM3-based road damage detection.
The actual implementation will be added in a future release.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class Sam3Detector:
    """
    SAM3-based detection engine.

    Currently a stub — not yet implemented. Attempting to use this detector
    will raise NotImplementedError both at the async and blocking levels.
    The server will not crash; the video job will transition to an 'error' state
    with a descriptive message.
    """

    def __init__(self):
        logger.warning(
            "Sam3Detector instantiated — this mode is not yet implemented. "
            "Any process_video call will fail gracefully."
        )

    async def process_video(
        self, video_id: str, video_path: str, json_path: str, speed_kmh: int
    ):
        from app.core.config import processing_status
        from app.ws.websocket_manager import manager

        msg = "SAM3 detection is not yet implemented."
        processing_status[video_id] = {"status": "error", "message": msg}
        await manager.send_message(video_id, {"type": "error", "message": msg})
        raise NotImplementedError(msg)

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        raise NotImplementedError("SAM3 detection is not yet implemented.")
