"""
Logging configuration for VisionRoad Backend
Configures both console and file logging with rotation
"""

import logging
import logging.handlers
import os
import time
from pathlib import Path

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
MAIN_LOG_FILE = LOGS_DIR / "visionroad.log"
ERROR_LOG_FILE = LOGS_DIR / "error.log"
DETECTION_LOG_FILE = LOGS_DIR / "detection.log"
PERF_LOG_FILE = LOGS_DIR / "perf.log"

# Logging format with file, function, and line number for precision
LOG_FORMAT = (
    "%(asctime)s - %(levelname)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Performance logger — propagate=False so timing entries don't bleed into visionroad.log
perf_logger = logging.getLogger("visionroad.perf")
perf_logger.setLevel(logging.INFO)
perf_logger.propagate = False


# Namespace filter ──────────────────────────────────────────────────
# Keeps visionroad.log focused on app-level logs only.
# Blocks noisy third-party namespaces from flooding the main log file.
_EXCLUDED_PREFIXES = (
    "uvicorn",
    "starlette",
    "fastapi",
    "multipart",
    "asyncio",
    "concurrent",
)


class _AppOnlyFilter(logging.Filter):
    """
    Allow only records whose logger name starts with 'app.' or 'visionroad.',
    or have no specific namespace (root-level logging.info() calls).
    Blocks third-party library noise from visionroad.log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        # Always allow root-level and app-namespace logs
        if name == "root" or name.startswith("app.") or name.startswith("visionroad."):
            return True
        # Block known noisy third-party prefixes
        if name.startswith(_EXCLUDED_PREFIXES):
            return False
        return True  # Allow anything else by default (e.g., __main__)


def _has_handler_for_file(logger: logging.Logger, filepath: Path) -> bool:
    """
    Check if a logger already has a FileHandler pointing to the given path.
    Prevents doubling up handlers on hot-reload
    """
    target = str(filepath.resolve())
    for h in logger.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            if str(Path(h.baseFilename).resolve()) == target:
                return True
    return False


class PerfTimer:
    """
    Context manager for timing pipeline stages.

    Usage:
        with PerfTimer("YOLO inference", video_id) as t:
            results = model.track(...)
        # t.elapsed    → duration in seconds
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
        return False  # don't suppress exceptions


def setup_logging():
    """Configure application-wide logging. Safe to call more than once."""

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear root handlers each time so we start fresh without stacking
    root_logger.handlers.clear()

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # Main log file — app-namespace only (_AppOnlyFilter)
    file_handler = logging.handlers.RotatingFileHandler(
        MAIN_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    file_handler.addFilter(_AppOnlyFilter())
    root_logger.addHandler(file_handler)

    # Error log file (ERROR and above only — no filter needed, errors always matter)
    error_handler = logging.handlers.RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root_logger.addHandler(error_handler)

    # ── Detection-specific loggers ───────────────────────────────────────────
    # Attach to the *parent* namespace loggers so every module added under
    # app.detectors.*, app.helpers.*, or app.services.* is automatically
    # captured — no need to list individual files here.
    # propagate=False so entries don't also appear in visionroad.log.
    _DETECTION_NAMESPACES = [
        "app.detectors",   # all detectors (yolo, yoloe, base, registry…)
        "app.helpers",     # all helpers  (vl_helper, sam3_helper, yoloe_helper…)
        "app.services",    # all services (upload_service, location_mapper…)
    ]

    detection_handler = logging.handlers.RotatingFileHandler(
        DETECTION_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
        encoding="utf-8",
    )
    detection_handler.setLevel(logging.DEBUG)
    detection_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    for ns in _DETECTION_NAMESPACES:
        ns_logger = logging.getLogger(ns)
        ns_logger.setLevel(logging.DEBUG)
        ns_logger.propagate = False  # no bleed into visionroad.log

        # only add handler if not already attached (survives reload)
        if not _has_handler_for_file(ns_logger, DETECTION_LOG_FILE):
            ns_logger.addHandler(detection_handler)

        # Forward ERRORs to shared error.log (propagation is off, so attach directly)
        if not _has_handler_for_file(ns_logger, ERROR_LOG_FILE):
            ns_logger.addHandler(error_handler)

    # In development, also stream detection logs to console for real-time
    # monitoring of YOLO/VL/YOLOE progress without tailing a file.
    if os.getenv("APP_ENV", "development") != "production":
        for ns in _DETECTION_NAMESPACES:
            ns_logger = logging.getLogger(ns)
            if console_handler not in ns_logger.handlers:
                ns_logger.addHandler(console_handler)

    # ── Performance logger ───────────────────────────────────────────────────
    perf_handler = logging.handlers.RotatingFileHandler(
        PERF_LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", DATE_FORMAT)
    )

    # guard against duplicate handlers on reload
    if not _has_handler_for_file(perf_logger, PERF_LOG_FILE):
        perf_logger.addHandler(perf_handler)

    # ── Suppress noisy third-party libraries ─────────────────────────────────
    for lib in ("ultralytics", "torch", "PIL", "httpx", "httpcore"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.info("=" * 60)
    logging.info("VisionRoad Backend Logging Initialized")
    logging.info(f"Main log:       {MAIN_LOG_FILE}")
    logging.info(f"Error log:      {ERROR_LOG_FILE}")
    logging.info(f"Detection log:  {DETECTION_LOG_FILE}")
    logging.info(f"Performance log:{PERF_LOG_FILE}")
    logging.info("=" * 60)
