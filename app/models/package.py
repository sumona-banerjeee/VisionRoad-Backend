"""Package model for project subdivisions"""

import uuid
from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin


class Package(Base, TimestampMixin):
    """
    Project subdivision representing regional sections.
    Covers an absolute NHAI kilometer range on the highway.
    Example: "Package A — KM 100-200", "Kolkata-Patna Section"
    """

    __tablename__ = "packages"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Foreign key to project
    project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Package details
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    region: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Absolute NHAI kilometer range covered by this package (optional)
    # e.g. chainage_start_km=100.0, chainage_end_km=200.0
    chainage_start_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    chainage_end_km: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="packages")

    chainages: Mapped[list["Chainage"]] = relationship(
        "Chainage", back_populates="package", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Package(id={self.id}, name={self.name}, project_id={self.project_id})>"
        )
