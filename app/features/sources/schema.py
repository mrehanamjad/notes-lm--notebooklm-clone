from pydantic import HttpUrl
from typing import Tuple
from datetime import datetime
import uuid
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


class SourceType(str, Enum):
    UPLOAD = "upload"
    WEBSITE = "website"
    YOUTUBE = "youtube"
    TOPIC = "topic"
    NOTE = "note"


class SourceStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class TargetURL(BaseModel):
    url: HttpUrl

    @field_validator('url', mode='before')
    @classmethod
    def ensure_scheme(cls, v: Any) -> Any:
        if isinstance(v, str):
            # Clean up accidental spaces
            v = v.strip()
            # If it doesn't start with http or https, assume https
            if not v.startswith(('http://', 'https://')):
                return f"https://{v}"
        return v


# ---- Source-specific data models ----

class UploadSourceData(BaseModel):
    file_name: str
    file_type: str
    file_size_bytes: int
    imagekit_file_id: str
    imagekit_url: str
    thumbnail_url: Optional[str] = None


class WebsiteSourceData(BaseModel):
    url: TargetURL
    title: str
    content: str = ""
    word_count: int = 0


class YouTubeSourceData(BaseModel):
    url: TargetURL
    video_id: str  
    thumbnail_url: Optional[str] = None
    language: str = "en"
    content: str = ""
    


class TopicSourceData(BaseModel):
    topic: str
    content: str = ""
    source_urls: list[str] = []


class NoteSourceData(BaseModel):
    content: str

    @field_validator('content')
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        from app.core.config import settings
        if len(v) > settings.MAX_PLAIN_TEXT_CHARS:
            raise ValueError(f"Text content exceeds maximum length of {settings.MAX_PLAIN_TEXT_CHARS} characters")
        return v


# ---- Request/Response schemas ----

class SourceCreate(BaseModel):
    notebook_id: uuid.UUID
    source_type: SourceType
    title: str
    source_data: dict[str, Any]


class NoteCreateRequest(BaseModel):
    notebook_id: uuid.UUID
    title: str = Field(..., max_length=255)
    content: str = Field(..., max_length=20000)

class SourceUploadResponse(BaseModel):
    id: uuid.UUID
    source_id: str
    title: str
    source_type: SourceType
    status: SourceStatus


class SourceResponse(BaseModel):
    id: uuid.UUID
    notebook_id: uuid.UUID
    user_id: uuid.UUID
    source_id: str
    source_type: SourceType
    title: str
    status: SourceStatus
    error_message: Optional[str] = None
    source_data: dict[str, Any]
    total_chunks: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SourceListResponse(BaseModel):
    sources: list[SourceResponse]
    total: int
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1)
    has_more: bool = False


class SourceDeleteResponse(BaseModel):
    message: str
    source_id: str


class SourceStatusResponse(BaseModel):
    source_id: str
    status: SourceStatus
    error_message: Optional[str] = None
    total_chunks: int