import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Single, centralized load_dotenv() for the entire app.
# This populates os.environ so os.getenv() works everywhere.
# Must be called before any module reads env vars at import time.
load_dotenv()


class Settings(BaseSettings):
    """Application settings and configuration"""

    # Application
    PROJECT_NAME: str = "VisionRoad API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "sqlite:///./visionroad.db"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    RESULTS_DIR: Path = BASE_DIR / "results"
    MODELS_DIR: Path = BASE_DIR / "models"

    # Processing
    DEFAULT_SPEED_KMH: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.RESULTS_DIR.mkdir(exist_ok=True)

# Convenience aliases
UPLOAD_DIR = settings.UPLOAD_DIR
RESULTS_DIR = settings.RESULTS_DIR

# In-memory storage for processing status and results
processing_status: Dict[str, dict] = {}
detection_results: Dict[str, dict] = {}
