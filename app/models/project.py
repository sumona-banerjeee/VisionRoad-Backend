"""Project model for organizing road infrastructure projects"""

import uuid
from sqlalchemy import String, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin


class Project(Base, TimestampMixin):
    """
    Top-level organizational unit for road infrastructure projects.
    Example: "Kolkata-Jaipur Highway", "NH-19 Expressway"
    """

    __tablename__ = "projects"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Project details
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    state: Mapped[str | None] = mapped_column(String(255), nullable=True)
    corridor_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Overall project GPS bounds (optional)
    start_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_lng: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    packages: Mapped[list["Package"]] = relationship(
        "Package", back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name})>"
