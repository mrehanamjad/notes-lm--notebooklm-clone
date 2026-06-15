from datetime import datetime
import uuid
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class DocumentChunkResponse(BaseModel):
    # FIXED: int → uuid.UUID
    id: uuid.UUID
    chunk_index: int
    chunk_text: str
    page_number: int
    is_table: bool
    qdrant_point_id: str

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: uuid.UUID
    notebook_id: uuid.UUID
    user_id: uuid.UUID
    doc_id: str
    file_name: str
    file_type: str
    total_pages: int
    total_chunks: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    id: uuid.UUID
    doc_id: str
    file_name: str
    status: str = "processing"
    message: str = "Document uploaded. Indexing in background."


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1)
    has_more: bool = False


class DocumentDetailResponse(DocumentResponse):
    chunks: list[DocumentChunkResponse] = []


class DocumentDeleteResponse(BaseModel):
    message: str
    doc_id: str
    chunks_deleted: int = 0


class DocumentStatusResponse(BaseModel):
    doc_id: str
    status: str
    total_chunks: int = 0
    error_message: Optional[str] = None