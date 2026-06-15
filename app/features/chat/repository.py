import uuid
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.features.chat.model import ChatSession, ChatMessage, MemorySummary


class ChatSessionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, session: ChatSession) -> ChatSession:
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_by_id(self, session_id: uuid.UUID, user_id: uuid.UUID) -> Optional[ChatSession]:
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def list_by_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID,
                               skip: int = 0, limit: int = 20) -> list[ChatSession]:
        result = await self.db.execute(
            select(ChatSession)
            .where(ChatSession.notebook_id == notebook_id, ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(ChatSession)
            .where(ChatSession.notebook_id == notebook_id, ChatSession.user_id == user_id)
        )
        return result.scalar_one()

    async def delete(self, session: ChatSession) -> None:
        await self.db.delete(session)
        await self.db.commit()


class ChatMessageRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, message: ChatMessage) -> ChatMessage:
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def get_messages(self, session_id: uuid.UUID, user_id: uuid.UUID,
                           skip: int = 0, limit: int = 50) -> list[ChatMessage]:
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.asc())
            .offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count_messages(self, session_id: uuid.UUID, user_id: uuid.UUID) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(ChatMessage)
            .where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
        )
        return result.scalar_one()

    async def get_recent_messages(self, session_id: uuid.UUID, user_id: uuid.UUID, limit: int = 12) -> list[ChatMessage]:
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        messages = list(result.scalars().all())
        messages.reverse()
        return messages

    async def get_active_messages(self, session_id: uuid.UUID, user_id: uuid.UUID, 
                                   after_message_id: Optional[uuid.UUID] = None) -> list[ChatMessage]:
        query = select(ChatMessage).where(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user_id
        )
        if after_message_id is not None:
            query = query.where(ChatMessage.id > after_message_id)  # Works with UUID
        query = query.order_by(ChatMessage.created_at.asc())
        result = await self.db.execute(query)
        return list(result.scalars().all())


class MemorySummaryRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, summary: MemorySummary) -> MemorySummary:
        self.db.add(summary)
        await self.db.commit()
        await self.db.refresh(summary)
        return summary

    async def get_latest(self, session_id: uuid.UUID, user_id: uuid.UUID) -> Optional[MemorySummary]:
        result = await self.db.execute(
            select(MemorySummary)
            .where(MemorySummary.session_id == session_id, MemorySummary.user_id == user_id)
            .order_by(MemorySummary.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def delete_by_session(self, session_id: uuid.UUID, user_id: uuid.UUID) -> None:
        result = await self.db.execute(
            select(MemorySummary)
            .where(MemorySummary.session_id == session_id, MemorySummary.user_id == user_id)
        )
        for summary in result.scalars().all():
            await self.db.delete(summary)
        await self.db.commit()