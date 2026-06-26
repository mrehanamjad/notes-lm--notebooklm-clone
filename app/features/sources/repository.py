from app.features.sources.model import SourceType
import uuid
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.features.sources.model import Source, SourceStatus


class SourceRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, source: Source) -> Source:
        self.db.add(source)
        await self.db.commit()
        await self.db.refresh(source)
        return source

    async def get_in_notebook(self, source_id: str, user_id: uuid.UUID, notebook_id: uuid.UUID) -> Optional[Source]:
        """Scope Validation: Checks if file exists in the specific notebook."""
        result = await self.db.execute(
            select(Source).where(
                Source.source_id == source_id,
                Source.user_id == user_id,
                Source.notebook_id == notebook_id
            )
        )
        return result.scalar_one_or_none()

    async def get_first_by_source_id(self, source_id: str, user_id: uuid.UUID) -> Optional[Source]:
        """Cross-Notebook Validation: Checks if file exists globally for the user."""
        result = await self.db.execute(
            select(Source).where(
                Source.source_id == source_id,
                Source.user_id == user_id
            ).limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_source_by_source_id_notebook_id(self, source_id: str, user_id: uuid.UUID, notebook_id: uuid.UUID) -> Optional[Source]:
        """Scope Validation: Checks if file exists in the specific notebook or globally for the user."""
        result = await self.db.execute(
            select(Source).where(
                Source.source_id == source_id,
                Source.user_id == user_id,
                Source.notebook_id == notebook_id
            )
        )
        return result.scalar_one_or_none()

    async def get_in_notebook_bulk(self, source_ids: list[str], user_id: uuid.UUID, notebook_id: uuid.UUID) -> list[Source]:
        """Scope Validation Bulk: Checks if multiple files exist in the specific notebook."""
        if not source_ids:
            return []
        result = await self.db.execute(
            select(Source).where(
                Source.source_id.in_(source_ids),
                Source.user_id == user_id,
                Source.notebook_id == notebook_id
            )
        )
        return list(result.scalars().all())

    async def get_global_bulk(self, source_ids: list[str], user_id: uuid.UUID) -> list[Source]:
        """Cross-Notebook Validation Bulk: Fetches existing global sources for cross-notebook linking."""
        if not source_ids:
            return []
        result = await self.db.execute(
            select(Source).where(
                Source.source_id.in_(source_ids),
                Source.user_id == user_id
            )
        )
        return list(result.scalars().all())

    async def list_by_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID,
                               skip: int = 0, limit: int = 20) -> list[Source]:
        result = await self.db.execute(
            select(Source)
            .where(Source.notebook_id == notebook_id, Source.user_id == user_id)
            .order_by(Source.created_at.desc())
            .offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(Source)
            .where(Source.notebook_id == notebook_id, Source.user_id == user_id)
        )
        return result.scalar_one()

    async def get_source_ids_for_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID) -> list[str]:
        """Get all source_ids belonging to a notebook for scoped RAG queries."""
        result = await self.db.execute(
            select(Source.source_id)
            .where(
                Source.notebook_id == notebook_id,
                Source.user_id == user_id,
                Source.status == SourceStatus.READY,
            )
        )
        return list(result.scalars().all())

    async def count_by_source_id(self, source_id: str, user_id: uuid.UUID) -> int:
        """Counts how many notebooks are using this file (critical for safe deletion)."""
        result = await self.db.execute(
            select(func.count()).select_from(Source)
            .where(Source.source_id == source_id, Source.user_id == user_id)
        )
        return result.scalar_one()

    # UPDATED: Must include notebook_id to prevent updating other notebooks
    async def update_status(self, source_id: str, notebook_id: uuid.UUID, status: SourceStatus,
                            total_chunks: int = 0, error_message: str | None = None) -> None:
        result = await self.db.execute(
            select(Source).where(
                Source.source_id == source_id, 
                Source.notebook_id == notebook_id
            )
        )
        source = result.scalar_one_or_none()
        if source:
            source.status = status
            source.total_chunks = total_chunks
            source.error_message = error_message
            await self.db.commit()

    async def delete(self, source: Source) -> None:
        await self.db.delete(source)
        await self.db.commit()

    async def create_many(self, sources: list[Source]) -> list[Source]:
        self.db.add_all(sources)
        await self.db.commit()
        return sources