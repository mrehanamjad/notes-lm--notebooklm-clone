# model.py
"""Artifact database model with support for all artifact types."""

from sqlalchemy import Index
import uuid
import enum
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Enum, func, UUID
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.database.base import Base


class ArtifactType(str, enum.Enum):
    """Supported artifact types."""
    QUIZ = "quiz"
    FLASHCARDS = "flashcards"
    FAQ = "faq"
    STUDY_GUIDE = "study_guide"
    SUMMARY = "summary"
    MINDMAP = "mindmap"
    SLIDE_DECK = "slide_deck"


class ArtifactStatus(str, enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class Artifact(Base):
    """Artifact model for storing generated learning materials."""
    
    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True )
    notebook_id = Column(UUID(as_uuid=True), ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False, index=True )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True )    
    artifact_type = Column(Enum(ArtifactType), nullable=False)
    status = Column(Enum(ArtifactStatus), default=ArtifactStatus.PROCESSING)
    title = Column(String(500), nullable=False)    
    #  {"question_count": 10, "difficulty": "mixed", "prompt": "Focus on neural networks"}
    options_json = Column(JSONB, nullable=False, default=dict, server_default='{}')
    included_sources = Column(
        JSONB,  # Using JSONB for array storage
        nullable=False, 
        default=list, 
        server_default='[]'
    )
    content_json = Column(JSONB, nullable=False, default=dict, server_default='{}')
    
    # Context building metadata 
    # Stores: mode_used, total_chunks, total_estimated_tokens
    context_metadata = Column(JSONB, nullable=False, default=dict, server_default='{}')
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    
    __table_args__ = (
        Index('ix_artifacts_included_sources', 'included_sources', postgresql_using='gin'),
    )

    notebook = relationship("Notebook", back_populates="artifacts")

    def __repr__(self) -> str:
        return f"<Artifact(id={self.id}, type={self.artifact_type.value}, title='{self.title[:50]}')>"