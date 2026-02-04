"""Package model for project subdivisions"""

import uuid
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin


class Package(Base, TimestampMixin):
    """
    Project subdivision representing regional sections.
    Example: "Kolkata-Patna Section", "Package A - West Bengal"
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

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="packages")

    locations: Mapped[list["Location"]] = relationship(
        "Location", back_populates="package", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Package(id={self.id}, name={self.name}, project_id={self.project_id})>"
        )
