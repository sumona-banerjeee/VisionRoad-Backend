# app/core/storage.py

from pathlib import Path
from typing import Dict

# Directory setup
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory storage for processing status and results
processing_status: Dict[str, dict] = {}
detection_results: Dict[str, dict] = {}