import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, func, UUID
from sqlalchemy.orm import relationship
from app.database.base import Base


class Notebook(Base):
    __tablename__ = "notebooks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="notebooks")
    sources = relationship("Source", back_populates="notebook", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="notebook", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="notebook", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Notebook(id={self.id}, title='{self.title}', user_id={self.user_id})>"