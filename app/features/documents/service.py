"""Document service — upload, index (background), list, delete."""

import uuid
import hashlib
import tempfile
import os
from typing import List

import asyncio

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, PayloadSchemaType

from app.core.config import settings
from app.core.ai_clients import get_qdrant_client, get_embeddings, get_vector_size
from app.core.logger import logger
from app.core.exceptions import NotFoundException

from app.features.notebooks.service import NotebookService
from app.features.documents.repository import DocumentRepository, DocumentChunkRepository
from app.features.documents.model import Document, DocumentChunk
from app.features.documents.schema import (
    DocumentUploadResponse, DocumentListResponse, DocumentResponse,
    DocumentDeleteResponse, DocumentStatusResponse,
)
from app.features.documents.loader import load_file, is_supported
from app.features.documents.chunker import generate_markdown, smart_chunk

from app.database.session import AsyncSessionLocal


def _ensure_collection(client, collection_name: str, vector_size: int) -> None:
    """Create Qdrant collection and payload indexes if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {collection_name}")
    
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD,  # UUID as string
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="notebook_id",
            field_schema=PayloadSchemaType.KEYWORD,  # UUID as string
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception as e:
        logger.error(f"Failed to create Qdrant payload indexes: {e}")


def _collection_exists(client, collection_name: str) -> bool:
    collections = [c.name for c in client.get_collections().collections]
    return collection_name in collections


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = DocumentChunkRepository(db)
        self.notebook_service = NotebookService(db)

    # FIXED: notebook_id: int → uuid.UUID, user_id: int → uuid.UUID
    async def upload_document(
        self, file: UploadFile, notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> DocumentUploadResponse:
        """Save metadata to DB and return immediately. Indexing runs in background."""
        # Validate ownership
        await self.notebook_service.get_notebook(notebook_id, user_id)

        file_name = file.filename or "unknown"
        if not is_supported(file_name):
            raise ValueError(f"Unsupported file type: {file_name}")

        # Generate doc_id
        content = await file.read()
        doc_id = hashlib.md5(content).hexdigest()
        await file.seek(0)

        # Check for duplicate
        existing = await self.doc_repo.get_by_doc_id(doc_id, user_id)
        if existing:
            return DocumentUploadResponse(
                id=existing.id, doc_id=doc_id, file_name=file_name,
                status=existing.status,
                message="Document already uploaded.",
            )

        # Save to DB as "processing"
        ext = os.path.splitext(file_name)[1].lstrip('.')
        document = Document(
            notebook_id=notebook_id,
            user_id=user_id,
            doc_id=doc_id,
            file_name=file_name,
            file_type=ext,
            status="processing",
        )
        document = await self.doc_repo.create(document)

        return DocumentUploadResponse(
            id=document.id, doc_id=doc_id, file_name=file_name,
            status="processing",
            message="Document uploaded. Indexing in background.",
        )

    async def list_documents(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, page: int = 1, size: int = 20
    ) -> DocumentListResponse:
        await self.notebook_service.get_notebook(notebook_id, user_id)
        skip = (page - 1) * size
        docs = await self.doc_repo.list_by_notebook(notebook_id, user_id, skip=skip, limit=size)
        total = await self.doc_repo.count_by_notebook(notebook_id, user_id)
        return DocumentListResponse(
            documents=[DocumentResponse.model_validate(d) for d in docs],
            total=total, page=page, size=size,
            has_more=(skip + size) < total,
        )

    async def get_document(self, doc_id: str, user_id: uuid.UUID) -> Document:
        doc = await self.doc_repo.get_by_doc_id_with_chunks(doc_id, user_id)
        if not doc:
            raise NotFoundException(f"Document {doc_id} not found")
        return doc

    async def get_status(self, doc_id: str, user_id: uuid.UUID) -> DocumentStatusResponse:
        doc = await self.doc_repo.get_by_doc_id(doc_id, user_id)
        if not doc:
            raise NotFoundException(f"Document {doc_id} not found")
        return DocumentStatusResponse(
            doc_id=doc_id, status=doc.status,
            total_chunks=doc.total_chunks, error_message=doc.error_message,
        )

    async def delete_document(self, doc_id: str, user_id: uuid.UUID) -> DocumentDeleteResponse:
        doc = await self.doc_repo.get_by_doc_id_with_chunks(doc_id, user_id)
        if not doc:
            raise NotFoundException(f"Document {doc_id} not found")

        chunks_deleted = 0
        try:
            client = get_qdrant_client()
            collection = settings.QDRANT_COLLECTION
            if _collection_exists(client, collection):
                # FIXED: Convert UUID to string for Qdrant filter
                filt = Filter(must=[
                    FieldCondition(key='doc_id', match=MatchValue(value=doc_id)),
                    FieldCondition(key='user_id', match=MatchValue(value=str(user_id))),
                ])
                count = client.count(collection_name=collection, count_filter=filt).count
                if count > 0:
                    client.delete(collection_name=collection, points_selector=filt)
                    chunks_deleted = count
        except Exception as e:
            logger.error(f"Qdrant cleanup error for doc {doc_id}: {e}")

        await self.doc_repo.delete(doc)
        logger.info(f"Document deleted: {doc_id}, {chunks_deleted} Qdrant points removed")

        return DocumentDeleteResponse(
            message="Document deleted successfully",
            doc_id=doc_id, chunks_deleted=chunks_deleted,
        )


# ── Background indexing task ──────────────────────────────────────────────────

async def index_document_background(
    doc_id: str,
    file_content: bytes,
    file_name: str,
    user_id: uuid.UUID,
    notebook_id: uuid.UUID,
) -> None:
    """Run indexing in background — saves chunks to Qdrant + Postgres."""
    async with AsyncSessionLocal() as db:
        doc_repo = DocumentRepository(db)
        chunk_repo = DocumentChunkRepository(db)
        tmp_path = None

        try:
            ext = os.path.splitext(file_name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            docs = load_file(tmp_path, file_name)
            markdown_text = generate_markdown(docs, file_name)
            chunks = smart_chunk(markdown_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

            if not chunks:
                await doc_repo.update_status(doc_id, "ready", total_pages=len(docs), total_chunks=0)
                return

            total_pages = max(c['page_number'] for c in chunks)

            embeddings = get_embeddings()
            client = get_qdrant_client()
            collection = settings.QDRANT_COLLECTION

            points: List[PointStruct] = []
            db_chunks: List[DocumentChunk] = []

            doc_record = await doc_repo.get_by_doc_id(doc_id, user_id)
            if not doc_record:
                logger.error(f"Document record not found for {doc_id}")
                return

            for idx, chunk in enumerate(chunks):
                vector = await asyncio.to_thread(embeddings.embed_query, chunk['text'])

                if idx == 0:
                    _ensure_collection(client, collection, len(vector))

                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        'file_name': file_name,
                        'file_type': ext.lstrip('.'),
                        'page_number': chunk['page_number'],
                        'chunk_index': idx,
                        'doc_id': doc_id,
                        'chunk_text': chunk['text'],
                        'is_table': chunk['is_table'],
                        'total_pages': total_pages,
                        'user_id': str(user_id),  # FIXED: UUID → string for Qdrant
                        'notebook_id': str(notebook_id),  # FIXED: UUID → string
                    },
                ))
                db_chunks.append(DocumentChunk(
                    document_id=doc_record.id,
                    user_id=user_id,
                    qdrant_point_id=point_id,
                    chunk_index=idx,
                    chunk_text=chunk['text'],
                    page_number=chunk['page_number'],
                    is_table=chunk['is_table'],
                ))

            client.upsert(collection_name=collection, points=points)
            await chunk_repo.create_many(db_chunks)
            await doc_repo.update_status(
                doc_id, "ready", total_pages=total_pages, total_chunks=len(points),
            )
            logger.info(f"Indexed {len(points)} chunks for doc {doc_id} ({file_name})")

        except Exception as e:
            logger.error(f"Indexing failed for {doc_id}: {e}", exc_info=True)
            await doc_repo.update_status(doc_id, "error", error_message=str(e))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass