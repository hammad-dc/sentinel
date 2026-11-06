import uuid
from datetime import datetime, timezone as dt_timezone
from sqlalchemy import Column, String, Integer, Float, BigInteger, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base


class Camera(Base):
    """Represents a physical or virtual CCTV camera."""

    __tablename__ = "cameras"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    location = Column(String(255), nullable=True)
    timezone = Column(String(50), default="UTC", nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(dt_timezone.utc),
        nullable=False,
    )

    videos = relationship(
        "Video", back_populates="camera", cascade="all, delete-orphan"
    )


class Video(Base):
    """Represents an uploaded surveillance recording."""

    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id"), nullable=True)

    filename = Column(String(255), nullable=False)
    storage_path = Column(String(512), nullable=False)
    sha256 = Column(String(64), nullable=False, index=True)

    # Forensic & temporal metadata
    duration_ms = Column(BigInteger, nullable=False)
    fps = Column(Float, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    time_base = Column(String(32), nullable=True)

    status = Column(String(32), default="UPLOADED", nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(dt_timezone.utc),
        nullable=False,
    )

    camera = relationship("Camera", back_populates="videos")
