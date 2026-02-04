"""Processing status model for tracking video processing progress"""

from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base
import enum


class ProcessingStatusEnum(str, enum.Enum):
    """Enum for processing status values"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class ProcessingStatus(Base):
    """Model for tracking video processing status and progress"""

    __tablename__ = "processing_status"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to video (one-to-one relationship)
    video_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Status information
    status: Mapped[ProcessingStatusEnum] = mapped_column(
        SQLEnum(ProcessingStatusEnum),
        nullable=False,
        default=ProcessingStatusEnum.PENDING,
    )

    # Progress tracking
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Error information
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationship to video
    video: Mapped["Video"] = relationship("Video", back_populates="processing_status")

    def __repr__(self) -> str:
        return f"<ProcessingStatus(video_id={self.video_id}, status={self.status}, progress={self.progress}%)>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "video_id": self.video_id,
            "status": self.status.value,
            "progress": self.progress,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }
