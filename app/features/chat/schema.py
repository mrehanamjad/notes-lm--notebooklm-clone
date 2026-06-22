from datetime import datetime
import uuid
from typing import Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict

class ChatSessionCreate(BaseModel):
    notebook_id: uuid.UUID
    title: Optional[str] = "New Chat"

class MessageRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=15000)
    excluded_source_ids: list[str] = Field(default_factory=list)

class CitationDetail(BaseModel):
    file_name: str
    page_number: int
    chunk_index: int
    similarity_score: float
    source_id: str
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


class BaseChatMessageResponse(BaseModel):
    """Shared fields for all messages"""
    id: uuid.UUID
    session_id: uuid.UUID
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class HumanMessageResponse(BaseChatMessageResponse):
    """Human messages only need the base fields"""
    role: Literal["human"]

class AssistantMessageResponse(BaseChatMessageResponse):
    """Assistant messages include extra metadata"""
    role: Literal["assistant"]
    citations: list[CitationDetail] = Field(default_factory=list)
    used_memory: bool = False

# This tells Pydantic to look at the "role" field to figure out 
# which model to use when parsing a list of messages.
ChatMessageResponse = Annotated[
    Union[HumanMessageResponse, AssistantMessageResponse], 
    Field(discriminator="role")
]

# ---------------------------------------------------------

class MessageListResponse(BaseModel):
    messages: list[ChatMessageResponse]  # Now safely handles both types!
    total: int
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1)
    has_more: bool = False

class AskResponse(BaseModel):
    # Strictly define which is which for the Ask endpoint
    human_message: HumanMessageResponse
    assistant_message: AssistantMessageResponse

class MemoryStatusResponse(BaseModel):
    session_id: uuid.UUID
    total_messages: int
    has_summary: bool
    summary_preview: Optional[str] = None