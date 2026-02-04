"""Location model for GPS-based road segments"""

import uuid
from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin


class Location(Base, TimestampMixin):
    """
    GPS-based road segment within a package.
    Defined by chainage markers and GPS bounding box.
    Example: "Segment KM 10-15", "Section A-B (CH 0+000 to 5+000)"
    """

    __tablename__ = "locations"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Foreign key to package
    package_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("packages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Location details
    segment_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Chainage (kilometer markers)
    chainage_start_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    chainage_end_km: Mapped[float | None] = mapped_column(Float, nullable=True)

    # GPS bounding box for this segment
    # Videos/detections within these bounds are mapped to this location
    start_lat: Mapped[float] = mapped_column(Float, nullable=False)
    start_lng: Mapped[float] = mapped_column(Float, nullable=False)
    end_lat: Mapped[float] = mapped_column(Float, nullable=False)
    end_lng: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    package: Mapped["Package"] = relationship("Package", back_populates="locations")

    videos: Mapped[list["Video"]] = relationship(
        "Video", back_populates="location", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Location(id={self.id}, name={self.segment_name}, package_id={self.package_id})>"

    def contains_point(self, lat: float, lng: float) -> bool:
        """
        Check if a GPS point falls within this location's bounding box.
        Uses min/max to handle roads traveling in any direction.
        """
        min_lat = min(self.start_lat, self.end_lat)
        max_lat = max(self.start_lat, self.end_lat)
        min_lng = min(self.start_lng, self.end_lng)
        max_lng = max(self.start_lng, self.end_lng)

        return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng
