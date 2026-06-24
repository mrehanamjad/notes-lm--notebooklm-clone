import uuid
import enum
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Enum, func, Integer, UUID
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.database.base import Base


class SourceType(str, enum.Enum):
    UPLOAD = "upload"
    WEBSITE = "website"
    YOUTUBE = "youtube"
    TOPIC = "topic"
    NOTE = "note"


class SourceStatus(str, enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class Source(Base):
    __tablename__ = "sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    notebook_id = Column(UUID(as_uuid=True), ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    source_id = Column(String(32), index=True, nullable=False)
    source_type = Column(Enum(SourceType), nullable=False)
    title = Column(String(500), nullable=False)

    status = Column(Enum(SourceStatus), default=SourceStatus.READY)
    error_message = Column(Text, nullable=True)

    source_data = Column(JSONB, nullable=False, default=dict, server_default='{}')

    total_chunks = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    notebook = relationship("Notebook", back_populates="sources")

    def __repr__(self) -> str:
        return f"<Source(id={self.id}, type={self.source_type}, title='{self.title[:50]}')>"