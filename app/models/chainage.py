"""Chainage model for NHAI-style linear road segments (replaces Location)"""

import uuid
from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin


class Chainage(Base, TimestampMixin):
    """
    NHAI-style linear road segment defined by absolute kilometer markers.
    Belongs to a Package. Contains one or more Lanes.
    Example: KM 100–120 of NH-19, KM 45–90 of NH-44
    """

    __tablename__ = "chainages"

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

    # Segment label (human-readable name)
    segment_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Absolute NHAI kilometer markers — required, not nullable
    chainage_start_km: Mapped[float] = mapped_column(Float, nullable=False)
    chainage_end_km: Mapped[float] = mapped_column(Float, nullable=False)

    # GPS bounding box for auto-mapping GPS tracks to this chainage
    start_lat: Mapped[float] = mapped_column(Float, nullable=False)
    start_lng: Mapped[float] = mapped_column(Float, nullable=False)
    end_lat: Mapped[float] = mapped_column(Float, nullable=False)
    end_lng: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    package: Mapped["Package"] = relationship("Package", back_populates="chainages")

    lanes: Mapped[list["Lane"]] = relationship(
        "Lane", back_populates="chainage", cascade="all, delete-orphan"
    )

    videos: Mapped[list["Video"]] = relationship(
        "Video", back_populates="chainage", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Chainage(id={self.id}, name={self.segment_name}, "
            f"km={self.chainage_start_km}-{self.chainage_end_km}, "
            f"package_id={self.package_id})>"
        )

    def contains_point(self, lat: float, lng: float) -> bool:
        """
        Check if a GPS point falls within this chainage's bounding box.
        Uses min/max to handle roads traveling in any direction.
        """
        min_lat = min(self.start_lat, self.end_lat)
        max_lat = max(self.start_lat, self.end_lat)
        min_lng = min(self.start_lng, self.end_lng)
        max_lng = max(self.start_lng, self.end_lng)

        return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng
