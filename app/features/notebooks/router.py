from app.core.schemas import APIResponse
import uuid
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.deps import get_db, get_current_user
from app.features.users.model import User
from app.features.notebooks.service import NotebookService
from app.features.notebooks.schema import (
    NotebookCreate, NotebookUpdate, NotebookResponse, NotebookListResponse,
)

router = APIRouter(tags=["Notebooks"])


@router.post("/", response_model=APIResponse[NotebookResponse], status_code=status.HTTP_201_CREATED)
async def create_notebook(
    data: NotebookCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new notebook."""
    service = NotebookService(db)
    notebook = await service.create_notebook(data, current_user.id)  # current_user.id is UUID
    return APIResponse(
        message="Notebook created successfully",
        data=notebook,
    )


@router.get("/", response_model=APIResponse[NotebookListResponse])
async def list_notebooks(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all notebooks for the current user."""
    service = NotebookService(db)
    notebooks = await service.list_notebooks(current_user.id, page=page, size=size)
    return APIResponse(
        message="Notebooks fetched successfully",
        data=notebooks
    )


@router.get("/{notebook_id}", response_model=APIResponse[NotebookResponse])
async def get_notebook(
    notebook_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get notebook details."""
    service = NotebookService(db)
    notebook = await service.get_notebook(notebook_id, current_user.id)
    return APIResponse(
        message="Notebook fetched successfully",
        data=notebook,
    )

@router.patch("/{notebook_id}", response_model=APIResponse[NotebookResponse])
async def update_notebook(
    notebook_id: uuid.UUID,
    data: NotebookUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update notebook title or description."""
    service = NotebookService(db)
    notebook = await service.update_notebook(notebook_id, data, current_user.id)
    return APIResponse(
        message="Notebook updated successfully",
        data=notebook,
    )


@router.delete("/{notebook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notebook(
    notebook_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a notebook and all its contents."""
    service = NotebookService(db)
    await service.delete_notebook(notebook_id, current_user.id)