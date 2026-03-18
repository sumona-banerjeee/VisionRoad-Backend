"""Video model for storing uploaded video metadata"""

from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin
import uuid


class Video(Base, TimestampMixin):
    """Model for storing video metadata"""

    __tablename__ = "videos"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Video metadata
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_path: Mapped[str] = mapped_column(String(512), nullable=False)
    processed_video_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    json_file_path: Mapped[str] = mapped_column(String(512), nullable=False)

    # Chainage + Lane linkage (replaces location_id)
    chainage_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("chainages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    lane_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("lanes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Detection parameters
    detection_type: Mapped[str] = mapped_column(String(50), nullable=False)
    speed_kmh: Mapped[int] = mapped_column(Integer, nullable=False, default=30)

    # Upload timestamp
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    chainage: Mapped["Chainage"] = relationship("Chainage", back_populates="videos")
    lane: Mapped["Lane"] = relationship("Lane", back_populates="videos")

    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="video", cascade="all, delete-orphan"
    )

    processing_status: Mapped["ProcessingStatus"] = relationship(
        "ProcessingStatus",
        back_populates="video",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Video(id={self.id}, filename={self.filename}, type={self.detection_type})>"
