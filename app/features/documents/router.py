import uuid
from fastapi import APIRouter, Depends, Query, UploadFile, File, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.deps import get_db, get_current_user
from app.features.users.model import User
from app.features.documents.service import DocumentService, index_document_background
from app.features.documents.schema import (
    DocumentUploadResponse, DocumentListResponse,
    DocumentDetailResponse, DocumentDeleteResponse, DocumentStatusResponse,
)

router = APIRouter(tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    notebook_id: uuid.UUID = Query(..., description="Notebook to upload into"),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a document. Returns immediately — indexing runs in background."""
    service = DocumentService(db)
    result = await service.upload_document(file, notebook_id, current_user.id)

    if result.status == "processing":
        file_content = await file.read()
        background_tasks.add_task(
            index_document_background,
            doc_id=result.doc_id,
            file_content=file_content,
            file_name=result.file_name,
            user_id=current_user.id,
            notebook_id=notebook_id,
        )

    return result


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    notebook_id: uuid.UUID = Query(..., description="Filter by notebook"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List documents in a notebook (paginated)."""
    service = DocumentService(db)
    return await service.list_documents(notebook_id, current_user.id, page=page, size=size)


@router.get("/{doc_id}", response_model=DocumentDetailResponse)
async def get_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get document details including chunks."""
    service = DocumentService(db)
    return await service.get_document(doc_id, current_user.id)


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Check indexing status of a document."""
    service = DocumentService(db)
    return await service.get_status(doc_id, current_user.id)


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a document from both Qdrant and the database."""
    service = DocumentService(db)
    return await service.delete_document(doc_id, current_user.id)