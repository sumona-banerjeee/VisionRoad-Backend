"""
Logging configuration for VisionRoad Backend
Configures both console and file logging with rotation
"""

import logging
import logging.handlers
import time
from pathlib import Path

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
MAIN_LOG_FILE = LOGS_DIR / "visionroad.log"
ERROR_LOG_FILE = LOGS_DIR / "error.log"
DETECTION_LOG_FILE = LOGS_DIR / "detection.log"
PERF_LOG_FILE = LOGS_DIR / "perf.log"  # New performance log file

# Logging format with file, function, and line number for precision
LOG_FORMAT = (
    "%(asctime)s - %(levelname)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Performance logger — propagate=False so timing entries don't bleed into visionroad.log
perf_logger = logging.getLogger("visionroad.perf")
perf_logger.setLevel(logging.INFO)
perf_logger.propagate = False


class PerfTimer:
    """
    Context manager for timing pipeline stages.

    Usage:
        with PerfTimer("YOLO inference", video_id) as t:
            results = model.track(...)
        # t.elapsed  → duration in seconds
        # t.elapsed_ms → duration in milliseconds

    Writes one line to perf_logger on exit:
        [<video_id>] <stage> | <elapsed>s
    """

    def __init__(self, stage: str, video_id: str = ""):
        self.stage = stage
        self.video_id = video_id
        self.elapsed: float = 0.0  # seconds
        self.elapsed_ms: float = 0.0  # milliseconds
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        self.elapsed_ms = self.elapsed * 1000
        prefix = f"[{self.video_id}] " if self.video_id else ""
        perf_logger.info(f"{prefix}{self.stage} | {self.elapsed:.4f}s")
        return False  # don't suppress exceptions


def setup_logging():
    """Configure application-wide logging"""

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Main log file handler with rotation (10MB per file, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        MAIN_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Error log file handler (ERROR and above only)
    error_handler = logging.handlers.RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)

    # Detection-specific log file (for video_processor, signboard_detector, pot_sign_detector)
    detection_handler = logging.handlers.RotatingFileHandler(
        DETECTION_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10,  # Keep more detection logs
        encoding="utf-8",
    )
    detection_handler.setLevel(logging.DEBUG)
    detection_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    detection_handler.setFormatter(detection_formatter)

    # Add detection handler to specific loggers
    for logger_name in [
        "app.services.video_processor",
        "app.services.signboard_detector",
        "app.services.pot_sign_detector",
    ]:
        specific_logger = logging.getLogger(logger_name)
        specific_logger.addHandler(detection_handler)

    # Performance log file handler — clean pipe-delimited format for easy parsing
    perf_handler = logging.handlers.RotatingFileHandler(
        PERF_LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5,
        encoding="utf-8",
    )
    perf_handler.setLevel(logging.INFO)
    perf_formatter = logging.Formatter("%(asctime)s | %(message)s", DATE_FORMAT)
    perf_handler.setFormatter(perf_formatter)
    # Only add if handler not already present (avoid duplicates on reload)
    if not perf_logger.handlers:
        perf_logger.addHandler(perf_handler)

    # Suppress noisy libraries
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logging.info("=" * 60)
    logging.info("VisionRoad Backend Logging Initialized")
    logging.info(f"Main log: {MAIN_LOG_FILE}")
    logging.info(f"Error log: {ERROR_LOG_FILE}")
    logging.info(f"Detection log: {DETECTION_LOG_FILE}")
    logging.info(f"Performance log: {PERF_LOG_FILE}")  # New log file info
    logging.info("=" * 60)
