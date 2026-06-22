from app.core.schemas import APIResponse
from typing import Optional
import uuid
from fastapi import APIRouter, Depends, Query, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
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


@router.post("/website", response_model=APIResponse[SourceUploadResponse], status_code=status.HTTP_202_ACCEPTED)
async def add_website(
    notebook_id: uuid.UUID = Query(...),
    url: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    source = await service.create_website_source(url, notebook_id, current_user.id)
    return APIResponse(
        message="Website added successfully",
        data=source
    )


@router.post("/youtube", response_model=APIResponse[list[SourceUploadResponse]], status_code=status.HTTP_202_ACCEPTED)
async def add_youtube(
    notebook_id: uuid.UUID = Query(...),
    url: list[str] = Query(default=list),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    sources = await service.create_youtube_source(url, notebook_id, current_user.id)
    return APIResponse(
        message="Youtube added successfully",
        data=sources
    )


@router.post("/topic", response_model=APIResponse[list[SourceUploadResponse]], status_code=status.HTTP_202_ACCEPTED)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    source = await service.get_source(source_id, current_user.id)
    return APIResponse(
        message="Source fetched successfully",
        data=source
    )


@router.get("/{source_id}/status", response_model=SourceStatusResponse)
async def get_source_status(
    source_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    status = await service.get_status(source_id, current_user.id)
    return APIResponse(
        message="Status fetched successfully",
        data=status
    )


@router.delete("/{source_id}", response_model=SourceDeleteResponse, status_code=status.HTTP_202_ACCEPTED)
async def delete_source(
    source_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = SourceService(db)
    return await service.delete_source(source_id, current_user.id)