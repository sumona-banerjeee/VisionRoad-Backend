"""Database engine and session configuration"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from typing import Generator


# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False,  # Set to True for SQL query logging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    Use with FastAPI's Depends() for automatic session management.

    Example:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables"""
    from app.db.base import Base

    # Import all models here to ensure they are registered with SQLAlchemy
    from app.models import video, detection, processing  # noqa

    Base.metadata.create_all(bind=engine)
