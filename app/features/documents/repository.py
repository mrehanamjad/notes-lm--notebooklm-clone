import uuid
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional
from app.features.documents.model import Document, DocumentChunk


class DocumentRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, document: Document) -> Document:
        self.db.add(document)
        await self.db.commit()
        await self.db.refresh(document)
        return document

    async def get_by_doc_id(self, doc_id: str, user_id: uuid.UUID) -> Optional[Document]:
        result = await self.db.execute(
            select(Document).where(Document.doc_id == doc_id, Document.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_doc_id_with_chunks(self, doc_id: str, user_id: uuid.UUID) -> Optional[Document]:
        result = await self.db.execute(
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.doc_id == doc_id, Document.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def list_by_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID,
                               skip: int = 0, limit: int = 20) -> list[Document]:
        result = await self.db.execute(
            select(Document)
            .where(Document.notebook_id == notebook_id, Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(Document)
            .where(Document.notebook_id == notebook_id, Document.user_id == user_id)
        )
        return result.scalar_one()

    async def get_doc_ids_for_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID) -> list[str]:
        """Get all doc_ids belonging to a notebook — used for scoped RAG queries."""
        result = await self.db.execute(
            select(Document.doc_id)
            .where(
                Document.notebook_id == notebook_id,
                Document.user_id == user_id,
                Document.status == "ready",
            )
        )
        return list(result.scalars().all())

    async def update_status(self, doc_id: str, status: str,
                            total_pages: int = 0, total_chunks: int = 0,
                            error_message: str | None = None) -> None:
        result = await self.db.execute(
            select(Document).where(Document.doc_id == doc_id)
        )
        doc = result.scalar_one_or_none()
        if doc:
            doc.status = status
            doc.total_pages = total_pages
            doc.total_chunks = total_chunks
            doc.error_message = error_message
            await self.db.commit()

    async def delete(self, document: Document) -> None:
        await self.db.delete(document)
        await self.db.commit()


class DocumentChunkRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_many(self, chunks: list[DocumentChunk]) -> None:
        self.db.add_all(chunks)
        await self.db.commit()

    async def get_by_document(self, document_id: uuid.UUID, user_id: uuid.UUID) -> list[DocumentChunk]:
        result = await self.db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id, DocumentChunk.user_id == user_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def delete_by_document(self, document_id: uuid.UUID) -> int:
        result = await self.db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        chunks = list(result.scalars().all())
        count = len(chunks)
        for chunk in chunks:
            await self.db.delete(chunk)
        await self.db.commit()
        return count