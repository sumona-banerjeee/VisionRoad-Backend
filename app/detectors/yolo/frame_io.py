"""
NDJSON frame I/O for the YOLO detection pipeline.

Handles temporary file creation, post-drain filtering, writing,
and read-back of per-frame detection data.
"""

import json
import os
import logging
import tempfile

from app.core.config import RESULTS_DIR

logger = logging.getLogger(__name__)


def create_ndjson_file(video_id: str):
    """
    Create a temporary NDJSON file for streaming frame data.

    Returns:
        (fd, path): the open file descriptor and its filesystem path.
    """
    fd = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ndjson",
        prefix=f"frames_{video_id}_",
        dir=str(RESULTS_DIR),
        delete=False,
    )
    path = fd.name
    logger.info(f"NDJSON (deferred-write) temp file: {path}")
    return fd, path


def filter_and_write_frames(pending_frames: list, confirmed_ids: set, ndjson_fd):
    """
    Filter buffered frame data to remove rejected tids, then write
    surviving frames to NDJSON.

    Args:
        pending_frames: list of frame_data dicts (each with 'detections' list)
        confirmed_ids: set of confirmed tracker ids
        ndjson_fd: open file descriptor to write to

    Returns:
        (frames_written: int, total_detections_count: int)
    """
    frames_written = 0
    total_detections_count = 0

    for fdata in pending_frames:
        surviving = [
            d for d in fdata["detections"]
            if d["detection_id"] in confirmed_ids
        ]
        if surviving:
            fdata["detections"] = surviving
            ndjson_fd.write(json.dumps(fdata) + "\n")
            frames_written += 1
            total_detections_count += len(surviving)

    logger.info(
        f"NDJSON post-drain filter — {frames_written} frames, "
        f"{total_detections_count} detections written "
        f"(rejected tids filtered from frame data)"
    )
    return frames_written, total_detections_count


def read_ndjson_frames(ndjson_fd, ndjson_path: str, expected_count: int = 0) -> list:
    """Close the NDJSON temp file and read all frame lines back as a list."""
    if not ndjson_fd.closed:
        ndjson_fd.close()
    frames = []
    try:
        with open(ndjson_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Failed to read back NDJSON frames: {e}")
    logger.info(
        f"NDJSON read-back complete — {len(frames)} frames loaded "
        f"(expected {expected_count}) from {ndjson_path}"
    )
    return frames


def cleanup_ndjson(ndjson_fd, ndjson_path: str):
    """Close and remove the NDJSON temp file (for use in finally blocks)."""
    try:
        if ndjson_fd and not ndjson_fd.closed:
            ndjson_fd.close()
        if ndjson_path and os.path.exists(ndjson_path):
            os.remove(ndjson_path)
            logger.info(f"NDJSON temp file cleaned up: {ndjson_path}")
    except OSError:
        pass
