import uuid
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.deps import get_db, get_current_user
from app.features.users.model import User
from app.features.chat.service import ChatService
from app.features.chat.schema import (
    ChatSessionCreate, ChatSessionResponse, ChatSessionListResponse,
    MessageRequest, MessageListResponse, AskResponse, MemoryStatusResponse,
)

router = APIRouter(tags=["Chat"])


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    data: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new chat session in a notebook."""
    service = ChatService(db)
    return await service.create_session(data, current_user.id)


@router.get("/sessions", response_model=ChatSessionListResponse)
async def list_sessions(
    notebook_id: uuid.UUID = Query(..., description="Filter by notebook"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List chat sessions in a notebook (paginated)."""
    service = ChatService(db)
    return await service.list_sessions(notebook_id, current_user.id, page=page, size=size)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a chat session and all its messages."""
    service = ChatService(db)
    await service.delete_session(session_id, current_user.id)


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_messages(
    session_id: uuid.UUID,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all messages in a chat session (paginated)."""
    service = ChatService(db)
    return await service.get_messages(session_id, current_user.id, page=page, size=size)


@router.post("/sessions/{session_id}/messages", response_model=AskResponse)
async def send_message(
    session_id: uuid.UUID,
    data: MessageRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Send a question and get an AI answer with citations."""
    service = ChatService(db)
    return await service.send_message(session_id, data.question, current_user.id)


@router.post("/sessions/{session_id}/clear-memory")
async def clear_memory(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Clear AI memory summaries. Messages are preserved."""
    service = ChatService(db)
    return await service.clear_memory(session_id, current_user.id)


@router.get("/sessions/{session_id}/memory-status", response_model=MemoryStatusResponse)
async def memory_status(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get memory status for a chat session."""
    service = ChatService(db)
    return await service.get_memory_status(session_id, current_user.id)