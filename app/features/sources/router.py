from typing import Optional, List
import uuid
from fastapi import APIRouter, Depends, Query, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.core.deps import get_db, get_current_user
from app.features.users.model import User
from app.features.sources.service import SourceService
from app.features.sources.schema import (
    SourceUploadResponse, SourceListResponse, SourceResponse,
    SourceDeleteResponse, SourceStatusResponse, NoteCreateRequest,
)

router = APIRouter(tags=["Sources"])

@router.post("/upload", response_model=APIResponse[SourceUploadResponse], status_code=status.HTTP_202_ACCEPTED)
async def upload_source(
    notebook_id: uuid.UUID = Query(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    source = await service.create_upload_source(file, notebook_id, current_user.id)
    return APIResponse(
        message="Source uploaded. Indexing in background.",
        data=source 
    )

@router.post("/website", response_model=APIResponse[List[SourceUploadResponse]], status_code=status.HTTP_202_ACCEPTED)
async def add_website(
    notebook_id: uuid.UUID = Query(...),
    urls: list[str] = Query(default=list, alias="url"), 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    # Fixed method name: create_website_sources
    sources = await service.create_website_sources(urls, notebook_id, current_user.id)
    return APIResponse(
        message="Websites added successfully",
        data=sources
    )

@router.post("/youtube", response_model=APIResponse[List[SourceUploadResponse]], status_code=status.HTTP_202_ACCEPTED)
async def add_youtube(
    notebook_id: uuid.UUID = Query(...),
    urls: list[str] = Query(default=list, alias="url"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    # Fixed method name: create_youtube_sources
    sources = await service.create_youtube_sources(urls, notebook_id, current_user.id)
    return APIResponse(
        message="YouTube videos added successfully",
        data=sources
    )

@router.post("/topic", response_model=APIResponse[List[SourceUploadResponse]], status_code=status.HTTP_202_ACCEPTED)
async def add_topic(
    notebook_id: uuid.UUID = Query(...),
    topic: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    sources = await service.create_topic_source(topic, notebook_id, current_user.id)
    return APIResponse(
        message="Topic processed successfully and web sources added",
        data=sources
    )

@router.post("/note", response_model=APIResponse[SourceUploadResponse], status_code=status.HTTP_202_ACCEPTED)
async def add_note(
    note: NoteCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    source = await service.create_note_source(note, current_user.id)
    return APIResponse(
        message="Note added successfully",
        data=source
    )

@router.post("/{source_id}/retry", response_model=APIResponse[SourceStatusResponse], status_code=status.HTTP_202_ACCEPTED)
async def retry_source(
    source_id: str,
    notebook_id: uuid.UUID = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    result = await service.retry_indexing(source_id, current_user.id, notebook_id)
    return APIResponse(
        message="Source indexing queued for retry",
        data=result
    )

@router.get("/", response_model=APIResponse[SourceListResponse])
async def list_sources(
    notebook_id: uuid.UUID = Query(...),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    sources = await service.list_sources(notebook_id, current_user.id, page, size)
    return APIResponse(
        message="Sources fetched successfully",
        data=sources
    )

@router.get("/{source_id}", response_model=APIResponse[SourceResponse])
async def get_source(
    source_id: str,
    notebook_id: uuid.UUID = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    source = await service.get_source(source_id, current_user.id, notebook_id)
    return APIResponse(
        message="Source fetched successfully",
        data=source
    )

# Fixed response_model to match the wrapped APIResponse return format
@router.get("/{source_id}/status", response_model=APIResponse[SourceStatusResponse])
async def get_source_status(
    source_id: str,
    notebook_id: uuid.UUID = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    status_data = await service.get_status(source_id, current_user.id, notebook_id)
    return APIResponse(
        message="Status fetched successfully",
        data=status_data
    )

# Standardized response wrapper and added the critical notebook_id requirement
@router.delete("/{source_id}", response_model=APIResponse[SourceDeleteResponse], status_code=status.HTTP_202_ACCEPTED)
async def delete_source(
    source_id: str,
    notebook_id: uuid.UUID = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    result = await service.delete_source(source_id, current_user.id, notebook_id)
    return APIResponse(
        message="Source deleted successfully",
        data=result
    )