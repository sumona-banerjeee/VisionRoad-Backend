"""Lane model for directional sides of a Chainage (NHAI-style)"""

import uuid
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin


class Lane(Base, TimestampMixin):
    """
    Directional side / carriageway of a Chainage.
    A chainage typically has at least two lanes: LHS and RHS.
    Example: LHS (Left Hand Side), RHS (Right Hand Side), UP, DOWN
    """

    __tablename__ = "lanes"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Foreign key to chainage
    chainage_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chainages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Lane code — e.g. "LHS", "RHS", "UP", "DOWN"
    lane_code: Mapped[str] = mapped_column(String(20), nullable=False)

    # Optional: lane type — e.g. "driving", "shoulder", "service"
    lane_type: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Optional: normalized direction description
    direction: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationships
    chainage: Mapped["Chainage"] = relationship("Chainage", back_populates="lanes")

    videos: Mapped[list["Video"]] = relationship(
        "Video", back_populates="lane", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Lane(id={self.id}, code={self.lane_code}, "
            f"chainage_id={self.chainage_id})>"
        )
