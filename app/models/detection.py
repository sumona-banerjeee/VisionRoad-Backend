"""Detection model for storing individual detection results"""

from sqlalchemy import String, Integer, Float, ForeignKey, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base
import json


class Detection(Base):
    """Model for storing individual detection results"""

    __tablename__ = "detections"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to video
    video_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Project hierarchy (denormalized for efficient querying)
    # These allow filtering detections by project/package/location without joins
    project_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    package_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    location_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )

    # Frame information
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # GPS coordinates
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Detection information
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detection_type: Mapped[str] = mapped_column(String(50), nullable=False)
    class_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Bounding box stored as JSON string
    bounding_box: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationship to video
    video: Mapped["Video"] = relationship("Video", back_populates="detections")

    def __repr__(self) -> str:
        return f"<Detection(id={self.id}, video_id={self.video_id}, frame={self.frame_number}, class={self.class_name})>"

    def get_bounding_box(self) -> dict:
        """Parse bounding box JSON string to dict"""
        return json.loads(self.bounding_box)

    def set_bounding_box(self, bbox: dict) -> None:
        """Convert bounding box dict to JSON string"""
        self.bounding_box = json.dumps(bbox)


# Create composite index for video_id and frame_number for faster queries
Index("idx_video_frame", Detection.video_id, Detection.frame_number)
