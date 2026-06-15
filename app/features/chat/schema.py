from datetime import datetime
import uuid
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ChatSessionCreate(BaseModel):
    notebook_id: uuid.UUID
    title: Optional[str] = "New Chat"


class MessageRequest(BaseModel):
    question: str = Field(..., min_length=1)


class CitationDetail(BaseModel):
    file_name: str
    page_number: int
    chunk_index: int
    similarity_score: float
    doc_id: str
    is_table: bool = False
    chunk_text: str = ""


class ChatSessionResponse(BaseModel):
    id: uuid.UUID
    notebook_id: uuid.UUID
    user_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionResponse]
    total: int
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1)
    has_more: bool = False


class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    citations: list[CitationDetail] = []  # FIXED: List → list
    used_memory: bool = False
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageListResponse(BaseModel):
    messages: list[ChatMessageResponse]  # FIXED: List → list
    total: int
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1)
    has_more: bool = False


class AskResponse(BaseModel):
    human_message: ChatMessageResponse
    assistant_message: ChatMessageResponse


class MemoryStatusResponse(BaseModel):
    session_id: uuid.UUID
    total_messages: int
    has_summary: bool
    summary_preview: Optional[str] = None