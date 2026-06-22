"""Background tasks for source indexing."""

import asyncio
import os
import tempfile
import uuid
from typing import List, Optional
from datetime import datetime

from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from app.core.ai_clients import get_qdrant_client, get_embeddings
from app.core.config import settings
from app.core.logger import logger
from app.database.session import AsyncSessionLocal
from app.features.sources.model import SourceType, SourceStatus
from app.features.sources.repository import SourceRepository
from app.features.sources.loader import load_file
from app.features.sources.chunker import generate_markdown, smart_chunk
from app.features.sources.helpers import ensure_collection



async def index_source_background(
    source_id: str,
    source_type: SourceType,
    user_id: uuid.UUID,
    notebook_id: uuid.UUID,
    **kwargs,
) -> None:
    """Generic background indexer for all source types."""
    async with AsyncSessionLocal() as db:
        source_repo = SourceRepository(db)

        try:
            # Extract text depending on source type
            if source_type == SourceType.UPLOAD:
                file_path = kwargs.get("file_path")
                file_name = kwargs.get("file_name")
                if not file_path or not os.path.exists(file_path):
                    raise ValueError("File path missing or invalid for upload")
                try:
                    docs = load_file(file_path, file_name)
                    markdown_text = generate_markdown(docs, file_name)
                finally:
                    os.unlink(file_path)  # clean up temporary file
                title = file_name

            elif source_type == SourceType.WEBSITE or source_type == SourceType.YOUTUBE:
                url = kwargs.get("url")
                content = kwargs.get("content")
                title = kwargs.get("title")
                markdown_text = content

            elif source_type == SourceType.TOPIC:
                topic = kwargs.get("topic")
                # Placeholder: generate research via LLM
                markdown_text = f"Research on {topic} (to be implemented)"
                title = topic

            elif source_type == SourceType.NOTE:
                text = kwargs.get("text")
                markdown_text = text
                title = kwargs.get("title", "Note")

            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            # Chunk the text
            chunks = smart_chunk(markdown_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            if not chunks:
                await source_repo.update_status(source_id, SourceStatus.READY, total_chunks=0)
                return

            embeddings = get_embeddings()
            client = get_qdrant_client()
            collection = settings.QDRANT_COLLECTION

            points: List[PointStruct] = []
            for idx, chunk in enumerate(chunks):
                vector = await asyncio.to_thread(embeddings.embed_query, chunk['text'])
                if idx == 0:
                    ensure_collection(client, collection, len(vector))

                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        'source_id': source_id,
                        'source_type': source_type.value,
                        'title': title,
                        'chunk_index': idx,
                        'chunk_text': chunk['text'],
                        'page_number': chunk.get('page_number', 1),
                        'is_table': chunk.get('is_table', False),
                        'user_id': str(user_id),
                        'notebook_id': str(notebook_id),
                        'file_name': kwargs.get('file_name', title),
                    },
                ))

            client.upsert(collection_name=collection, points=points)
            await source_repo.update_status(
                source_id,notebook_id, SourceStatus.READY,
                total_chunks=len(points),
            )
            logger.info(f"Indexed {len(points)} chunks for source {source_id} ({source_type})")

        except Exception as e:
            logger.error(f"Indexing failed for {source_id}: {e}", exc_info=True)
            await source_repo.update_status(source_id, notebook_id, SourceStatus.ERROR, error_message=str(e))