"""Source service — supports upload, website, YouTube, topic, note."""

from imagekitio.types import video_transformation_accepted_event
from imagekitio.types import video_transformation_accepted_event
from app.features.sources.loader import load_yt
from app.features.sources.loader import load_yt_bulk
from app.features.sources.loader import load_topic
from app.core.validator import StringValidator
from app.features.sources.loader import load_web
from app.core.exceptions import BadRequestException
from app.core.exceptions import ConflictException
from app.features.sources.tasks import index_source_background
import uuid
import hashlib
import tempfile
import aiofiles  
import os
from typing import List, Optional
from datetime import datetime

import asyncio
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, PayloadSchemaType

from app.core.config import settings
from app.core.ai_clients import get_qdrant_client, get_embeddings, get_vector_size
from app.core.logger import logger
from app.core.exceptions import NotFoundException, ValidationError
from app.core.providers.storage import get_storage_provider

from app.features.notebooks.service import NotebookService
from app.features.sources.repository import SourceRepository
from app.features.sources.model import Source, SourceType, SourceStatus
from app.features.sources.schema import (
    SourceUploadResponse, SourceListResponse, SourceResponse,
    SourceDeleteResponse, SourceStatusResponse, NoteCreateRequest,
)
from app.features.sources.loader import load_file, is_supported
from app.features.sources.chunker import generate_markdown, smart_chunk
from app.features.sources.helpers import ensure_collection, collection_exists
from googleapiclient.discovery import build
import re
from app.database.session import AsyncSessionLocal

class SourceService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.source_repo = SourceRepository(db)
        self.notebook_service = NotebookService(db)
        self.storage = get_storage_provider()
        self.empty_validator = StringValidator.validate_non_empty

        self.youtube_client = None
        if hasattr(settings, "YOUTUBE_DATA_API_KEY") and settings.YOUTUBE_DATA_API_KEY:
            self.youtube_client = build('youtube', 'v3', developerKey=settings.YOUTUBE_DATA_API_KEY)

    @staticmethod
    def _handle_bg_task_result(t: asyncio.Task, s_id: str) -> None:
        """
        Generic callback to catch and log silent crashes in background workers.
        """
        try:
            t.result()
        except Exception as e:
            logger.error(f"Background indexing crashed for {s_id}: {e}", exc_info=True)

    @staticmethod
    def _extract_video_id(url: str) -> str:
        """Helper to extract the 11-character YouTube video ID from any URL format."""
        match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11}).*", url)
        return match.group(1) if match else url

    async def _fetch_youtube_metadata(self, urls: list[str]) -> list[dict]:
        """
        Fetches official metadata from YouTube Data API v3.
        Gracefully handles partial failures (e.g., if 1 out of 5 videos is private/deleted),
        ensuring valid data is kept and missing data gets a safe fallback.
        """
        # 1. Extract video IDs from the URLs using your helper method
        video_ids = [self._extract_video_id(url) for url in urls]

        # The YouTube API limits 'id' parameter to 50 videos per request.
        if len(video_ids) > 50:
            logger.warning(
                f"More than 50 video IDs provided ({len(video_ids)}). "
                "Only the first 50 will be fetched in this single request."
            )
            video_ids = video_ids[:50]

        video_ids_str = ",".join(video_ids)
        youtube_videos_data = []

        # 2. Check if the YouTube client is configured
        if getattr(self, "youtube_client", None) is None:
            logger.debug("YOUTUBE_API_KEY not found or client not initialized. Using fallback metadata.")
            return self._generate_fallback_metadata(video_ids)

        # 3. Fetch from Official Google API
        try:
            request = self.youtube_client.videos().list(
                part="snippet,player",
                id=video_ids_str
            )
            # Execute synchronously in a background thread to avoid blocking the event loop
            response = await asyncio.to_thread(request.execute)

            # 4. Map successfully retrieved videos by their ID for quick lookup
            fetched_items = {item['id']: item for item in response.get('items', [])}

            # 5. Iterate through the ORIGINAL requested list to build the final payload
            for video_id in video_ids:
                standard_link = f"https://www.youtube.com/watch?v={video_id}"
                embed_link = f"https://www.youtube.com/embed/{video_id}"

                if video_id in fetched_items:
                    # Video is public and fetched successfully
                    item = fetched_items[video_id]
                    title = item['snippet'].get('title') or f"YouTube Video: {video_id}"
                    
                    # Safely extract embed HTML or fallback to a standard iframe
                    embed_html = item.get('player', {}).get('embedHtml') 
                    if not embed_html:
                        embed_html = f'<iframe width="560" height="315" src="{embed_link}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>'
                    
                    # Safely extract the highest available quality thumbnail
                    thumbnails = item['snippet'].get('thumbnails', {})
                    thumbnail_url = (
                        thumbnails.get('maxres', {}).get('url') or 
                        thumbnails.get('standard', {}).get('url') or 
                        thumbnails.get('high', {}).get('url') or 
                        thumbnails.get('default', {}).get('url') or 
                        ""
                    )

                else:
                    # Video is private, deleted, or geo-restricted. 
                    # Create safe fallback metadata JUST for this specific video.
                    title = f"YouTube Video: {video_id} (Metadata Unavailable)"
                    embed_html = f'<iframe width="560" height="315" src="{embed_link}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>'
                    thumbnail_url = ""

                # Append the constructed metadata to our final list
                youtube_videos_data.append({
                    "video_id": video_id,
                    "title": title,
                    "url": standard_link,
                    "embed_url": embed_link,
                    "embed_html": embed_html,
                    "thumbnail_url": thumbnail_url,
                })

        except Exception as e:
            # If the entire API call fails (e.g., quota exceeded, network error),
            # log it and fall back to dummy data for the whole batch.
            logger.warning(f"Official YouTube API fetch failed for batch: {e}. Falling back to default metadata.")
            return self._generate_fallback_metadata(video_ids)

        return youtube_videos_data

    def _generate_fallback_metadata(self, video_ids: list[str]) -> list[dict]:
        """Helper to generate dummy metadata when the API fails or is unavailable."""
        fallback_data = []
        for video_id in video_ids:
            standard_link = f"https://www.youtube.com/watch?v={video_id}"
            embed_link = f"https://www.youtube.com/embed/{video_id}"
            fallback_data.append({
                "video_id": video_id,
                "title": f"YouTube Video: {video_id}",
                "url": standard_link,
                "embed_url": embed_link,
                "embed_html": f'<iframe width="560" height="315" src="{embed_link}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>',
                "thumbnail_url": "",
            })
        return fallback_data


    async def _handle_duplication(self, source_id: str, user_id: uuid.UUID, notebook_id: uuid.UUID) -> Optional[Source]:
        """Applies Rule 1 (Scope) and Rule 2 (Cross-Notebook)."""
        # Rule 1: Scope Validation
        existing_in_notebook = await self.source_repo.get_in_notebook(source_id, user_id, notebook_id)
        if existing_in_notebook:
            raise ConflictException(
                message="Source already exists in this notebook.",
                details={
                    "source_id": existing_in_notebook.source_id,
                    "title": existing_in_notebook.title,
                    "status": existing_in_notebook.status,
                }
            )

        # Rule 2: Cross-Notebook Validation
        existing_global = await self.source_repo.get_first_by_source_id(source_id, user_id)
        if existing_global:
            # Create a soft-link row. No new storage upload, no embeddings.
            source = Source(
                notebook_id=notebook_id,
                user_id=user_id,
                source_id=source_id,
                source_type=existing_global.source_type,
                title=existing_global.title,
                status=existing_global.status,
                source_data=existing_global.source_data,
                total_chunks=existing_global.total_chunks,
                error_message=existing_global.error_message
            )
            return await self.source_repo.create(source)
            
        return None

    # ---- Upload source ----
    async def create_upload_source(
        self, file: UploadFile, notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> SourceUploadResponse:
        await self.notebook_service.get_notebook(notebook_id, user_id)

        file_name = file.filename or "unknown"
        if not is_supported(file_name):
            raise ValidationError(f"Unsupported file type: {file_name}")

        # ── 1. Create a SINGLE temp file (streaming) ──────────────────────────
        ext = os.path.splitext(file_name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_path = temp_file.name

        try:
            # Stream the uploaded file to disk asynchronously
            async with aiofiles.open(temp_path, 'wb') as out_file:
                while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                    await out_file.write(chunk)
        except Exception as e:
            os.unlink(temp_path)  # Clean up if streaming fails
            raise ValidationError(f"Failed to save uploaded file: {e}")

        # ── 2. Compute hash from the temp file (avoid reading into memory) ────
        hasher = hashlib.md5()
        async with aiofiles.open(temp_path, 'rb') as f:
            while chunk := await f.read(8192):
                hasher.update(chunk)
        source_id = hasher.hexdigest()

        # ── 3. Duplicate check ──────────────────────────────────────────────────
        try:
            linked_source = await self._handle_duplication(source_id, user_id, notebook_id)
            if linked_source:
                os.unlink(temp_path)  # File linked! Delete temp, bypass background task.
                return SourceUploadResponse(
                    id=linked_source.id,
                    source_id=linked_source.source_id,
                    title=linked_source.title,
                    source_type=linked_source.source_type,
                    status=linked_source.status,
                )
        except ConflictException:
            os.unlink(temp_path)
            raise

        # ── 4. Upload to ImageKit (using the temp file path) ──────────────────
        try:
            upload_result = self.storage.upload_file(
                temp_path,  # Pass the path directly
                file_name,
                folder=f"sources/{user_id}"
            )
        except Exception as e:
            os.unlink(temp_path)  # Clean up if ImageKit fails
            logger.error(f"ImageKit upload failed: {e}")
            raise ValidationError(f"Failed to store file: {e}")

        # ── 5. Save to DB ──────────────────────────────────────────────────────
        file_size = os.path.getsize(temp_path)
        file_type = ext.lstrip('.')
        source_data = {
            "file_name": file_name,
            "file_type": file_type,
            "file_size_bytes": file_size,
            "imagekit_file_id": upload_result.get("file_id"),
            "imagekit_url": upload_result.get("url"),
            "thumbnail_url": upload_result.get("thumbnail_url")
        }

        source = Source(
            notebook_id=notebook_id,
            user_id=user_id,
            source_id=source_id,
            source_type=SourceType.UPLOAD,
            title=file_name,
            status=SourceStatus.PROCESSING,
            source_data=source_data,
        )
        source = await self.source_repo.create(source)

        # ── 6. Spawn background task & pass the temp file path ────────────────
        asyncio.create_task(
            index_source_background(
                source_id=source_id,
                source_type=SourceType.UPLOAD,
                user_id=user_id,
                notebook_id=notebook_id,
                file_path=temp_path,          # Pass the path, not the bytes!
                file_name=file_name,
            )
        )

        # ⚠️ IMPORTANT: Do NOT delete temp_path here. The background task owns it now.

        return SourceUploadResponse(
            id=source.id,
            source_id=source_id,
            title=source.title,
            source_type=SourceType.UPLOAD,
            status=SourceStatus.PROCESSING,
        )


    # ---- Website source ----
    async def create_website_source(
        self, url: str, notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> SourceUploadResponse:
        """Fetch and index a website."""

        self.empty_validator(url, "URL")

        await self.notebook_service.get_notebook(notebook_id, user_id)

        clean_url = url.strip()

        # Generate source_id from URL
        source_id = hashlib.md5(clean_url.encode()).hexdigest()

        linked_source = await self._handle_duplication(source_id, user_id, notebook_id)
        if linked_source:
            return SourceUploadResponse(
                id=linked_source.id,
                source_id=linked_source.source_id,
                title=linked_source.title,
                source_type=linked_source.source_type,
                status=linked_source.status,
            )

        
        web_content = await load_web([url])
        if not web_content:
            raise BadRequestException("Failed to process website content")

        web_data = web_content[0]
        title = web_data.metadata.get("title") or url
        content = web_data.page_content
        source_data = {
            "url": url,
            "title": title,
            "content": content,
            "word_count": len(content),
        }

        source = Source(
            notebook_id=notebook_id,
            user_id=user_id,
            source_id=source_id,
            source_type=SourceType.WEBSITE,
            title=title,
            status=SourceStatus.PROCESSING,
            source_data=source_data,
        )
        source = await self.source_repo.create(source)

        # Background indexing
        asyncio.create_task(
            index_source_background(
                source_id=source_id,
                source_type=SourceType.WEBSITE,
                user_id=user_id,
                notebook_id=notebook_id,
                url=url,
                content=content,
                title=title,
            )
        )

        return SourceUploadResponse(
            id=source.id,
            source_id=source_id,
            title=source.title,
            source_type=SourceType.WEBSITE,
            status=SourceStatus.PROCESSING,
        )



    # ---- YouTube source ----
# ---- Unified YouTube Source (Single or Bulk) ----
    async def create_youtube_sources(
        self, urls: list[str], notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> List[SourceUploadResponse]:
        """
        Accepts a list of URLs (including a single link in a list).
        Concurrently fetches transcripts and official metadata, handles duplication rules, 
        and triggers background indexing.
        """

        # Clean and filter out empty strings
        valid_urls = [u.strip() for u in urls if u and u.strip()]
        if not valid_urls:
            raise BadRequestException("No valid YouTube URLs provided for ingestion.")

        await self.notebook_service.get_notebook(notebook_id, user_id)

        # 1. Fetch metadata AND transcripts concurrently for maximum speed
        metadata_task = self._fetch_youtube_metadata(valid_urls)
        transcripts_task = load_yt_bulk(valid_urls)
        
        metadata_list, raw_docs = await asyncio.gather(metadata_task, transcripts_task)

        if not raw_docs:
            raise BadRequestException("Failed to retrieve transcripts from the provided YouTube targets.")

        # Create a quick lookup dictionary for the metadata using video_id
        metadata_by_vid = {m["video_id"]: m for m in metadata_list}

        # Group returned document chunks back by their original source URL
        docs_by_url = {}
        for doc in raw_docs:
            source_url = doc.metadata.get("source")
            if source_url:
                if source_url not in docs_by_url:
                    docs_by_url[source_url] = []
                docs_by_url[source_url].append(doc.page_content)

        all_new_sources = []
        responses = []

        # 3. Process duplication layers and prepare database rows
        for url, chunks in docs_by_url.items():
            source_id = hashlib.md5(url.encode("utf-8")).hexdigest()
            content = "\n".join(chunks)
            
            # Extract video ID to match with our fetched metadata
            video_id = self._extract_video_id(url)
            meta = metadata_by_vid.get(video_id, {})

            try:
                # Rule 1 & 2 cross-notebook checking
                linked_source = await self._handle_duplication(source_id, user_id, notebook_id)
                if linked_source:
                    responses.append(
                        SourceUploadResponse(
                            id=linked_source.id,
                            source_id=linked_source.source_id,
                            title=linked_source.title,
                            source_type=linked_source.source_type,
                            status=linked_source.status,
                        )
                    )
                    continue 
            except ConflictException:
                # Silent skip if the link already exists in this specific notebook
                continue 

            # Enrich the database payload with official Google API data
            source_data = {
                "url": meta.get("url", url),
                "video_id": video_id,
                "language": "en",
                "content": content,
                "word_count": len(content.split()),
                "embed_url": meta.get("embed_url"),
                "embed_html": meta.get("embed_html"),
                "thumbnail_url": meta.get("thumbnail_url"),
            }
            
            # Use official title if available, otherwise fallback
            title = meta.get("title", f"YouTube Video: {url}")
            
            source = Source(
                notebook_id=notebook_id,
                user_id=user_id,
                source_id=source_id,
                source_type=SourceType.YOUTUBE, 
                title=title,
                status=SourceStatus.PROCESSING,
                source_data=source_data,
            )
            all_new_sources.append(source)

        # 4. Bulk DB insert and dispatch background execution workers
        if all_new_sources:
            inserted_sources = await self.source_repo.create_many(all_new_sources)

            for source in inserted_sources:
                responses.append(
                    SourceUploadResponse(
                        id=source.id,
                        source_id=source.source_id,
                        title=source.title,
                        source_type=SourceType.YOUTUBE,
                        status=SourceStatus.PROCESSING,
                    )
                )
                
                task = asyncio.create_task(
                    index_source_background(
                        source_id=source.source_id,
                        source_type=SourceType.YOUTUBE,
                        user_id=user_id,
                        notebook_id=notebook_id,
                        url=source.source_data["url"],
                        content=source.source_data["content"],
                        title=source.title,
                    )
                )
                
                task.add_done_callback(lambda t, s_id=source.source_id: self._handle_bg_task_result(t, s_id))

        return responses
    
    # ---- Topic source ----
    async def create_topic_source(
    self, topic: str, notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> list[SourceUploadResponse]:
        """
        Searches a topic, resolves duplicates safely, bulk saves new websites, 
        and fires concurrent indexing tasks.
        """
        await self.notebook_service.get_notebook(notebook_id, user_id)
        clean_topic = topic.strip()

        # 1. Fetch the raw web documents
        web_search_results = await load_topic(clean_topic)
        if not web_search_results:
            raise BadRequestException("No web search results found for the given topic.")
        
        all_new_sources = []
        responses = []

        # 2. Process each result and handle duplication rules safely
        for web_result in web_search_results:
            # LangChain AsyncHtmlLoader stores the URL under the "source" metadata key
            url = web_result.metadata.get("source", "unknown_url")
            title = web_result.metadata.get("title", "Untitled Document")
            content = web_result.page_content
            
            # Hash the URL for deterministic deduplication
            source_id = hashlib.md5(url.encode("utf-8", errors="ignore")).hexdigest()

            try:
                # Check Rule 1 (Scope) and Rule 2 (Cross-Notebook)
                linked_source = await self._handle_duplication(source_id, user_id, notebook_id)
                
                if linked_source:
                    # Rule 2 triggered: It was a global duplicate. 
                    # _handle_duplication just saved a soft-link to the DB.
                    # We add it to our response, but DO NOT bulk insert or trigger embedding!
                    responses.append(
                        SourceUploadResponse(
                            id=linked_source.id,
                            source_id=linked_source.source_id,
                            title=linked_source.title,
                            source_type=linked_source.source_type,
                            status=linked_source.status,
                        )
                    )
                    continue 
                    
            except ConflictException:
                # Rule 1 triggered: It's already in this exact notebook.
                # Silently catch and skip it so the batch doesn't crash.
                continue 

            source_data = {
                "url": url,
                "title": title,
                "content": content,
                "origin_topic_query": clean_topic
            }
            
            source = Source(
                notebook_id=notebook_id,
                user_id=user_id,
                source_id=source_id,
                source_type=SourceType.WEBSITE, 
                title=title,
                status=SourceStatus.PROCESSING,
                source_data=source_data,
            )
            all_new_sources.append(source)

        if all_new_sources:
            inserted_sources = await self.source_repo.create_many(all_new_sources)
            for source in inserted_sources:
                responses.append(
                    SourceUploadResponse(
                        id=source.id,
                        source_id=source.source_id,
                        title=source.title,
                        source_type=SourceType.WEBSITE,
                        status=SourceStatus.PROCESSING,
                    )
                )
                
                # Fire the worker thread
                task = asyncio.create_task(
                    index_source_background(
                        source_id=source.source_id,
                        source_type=SourceType.WEBSITE,
                        user_id=user_id,
                        notebook_id=notebook_id,
                        url=source.source_data["url"],
                        content=source.source_data["content"],
                        title=source.title,
                    )
                )
                
                # Use a lambda to pass both the task and the source_id
                task.add_done_callback(lambda t, s_id=source.source_id: self._handle_bg_task_result(t, s_id))

        # Returns the combined list of newly processed URLs + instantly soft-linked globals
        return responses


    # ---- Note source ----
    async def create_note_source(
        self, note_data: NoteCreateRequest, user_id: uuid.UUID
    ) -> SourceUploadResponse:
        """Create a note (plain text) and index it."""

        if not note_data.title.strip() or not note_data.content.strip():
            raise BadRequestException("Note title and content are required.")

        notebook = await self.notebook_service.get_notebook(note_data.notebook_id, user_id)

        if not notebook:
            raise NotFoundException(f"Notebook  not found or Access Denied")

        if not note_data.title or not note_data.content:
            raise BadRequestException("Note title and content are required.")

        # source_id from text hash
        source_id = hashlib.md5(note_data.content.encode()).hexdigest()

        linked_source = await self._handle_duplication(source_id, user_id, note_data.notebook_id)
        if linked_source:
            return SourceUploadResponse(
                id=linked_source.id,
                source_id=linked_source.source_id,
                title=linked_source.title,
                source_type=linked_source.source_type,
                status=linked_source.status,
            )

        source_data = {
            "content": note_data.content,
        }

        source = Source(
            notebook_id=note_data.notebook_id,
            user_id=user_id,
            source_id=source_id,
            source_type=SourceType.NOTE,
            title=note_data.title,
            status=SourceStatus.PROCESSING,
            source_data=source_data,
        )
        source = await self.source_repo.create(source)

        asyncio.create_task(
            index_source_background(
                source_id=source_id,
                source_type=SourceType.NOTE,
                user_id=user_id,
                notebook_id=note_data.notebook_id,
                text=note_data.content,
                title=note_data.title,
            )
        )

        return SourceUploadResponse(
            id=source.id,
            source_id=source_id,
            title=source.title,
            source_type=SourceType.NOTE,
            status=SourceStatus.PROCESSING,
        )

    # ---- Common operations ----
    async def list_sources(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, page: int = 1, size: int = 20
    ) -> SourceListResponse:
        await self.notebook_service.get_notebook(notebook_id, user_id)
        skip = (page - 1) * size
        sources = await self.source_repo.list_by_notebook(notebook_id, user_id, skip=skip, limit=size)
        total = await self.source_repo.count_by_notebook(notebook_id, user_id)
        return SourceListResponse(
            sources=[SourceResponse.model_validate(s) for s in sources],
            total=total, page=page, size=size,
            has_more=(skip + size) < total,
        )

    async def get_source(self, source_id: str, user_id: uuid.UUID) -> Source:
        source = await self.source_repo.get_by_source_id(source_id, user_id)
        if not source:
            raise NotFoundException(f"Source {source_id} not found")
        return source

    async def get_status(self, source_id: str, user_id: uuid.UUID) -> SourceStatusResponse:
        source = await self.source_repo.get_by_source_id(source_id, user_id)
        if not source:
            raise NotFoundException(f"Source {source_id} not found")
        return SourceStatusResponse(
            source_id=source_id,
            status=source.status,
            error_message=source.error_message,
            total_chunks=source.total_chunks,  
        )  

    # ---- Delete source ----

    async def _cleanup_qdrant(self, source_id: str, user_id: uuid.UUID) -> int:
        """Isolated logic for Qdrant cleanup."""
        try:
            client = get_qdrant_client()
            collection = settings.QDRANT_COLLECTION
            if not collection_exists(client, collection):
                return 0
                
            filt = Filter(must=[
                FieldCondition(key='source_id', match=MatchValue(value=source_id)),
                FieldCondition(key='user_id', match=MatchValue(value=str(user_id))),
            ])
            
            count = client.count(collection_name=collection, count_filter=filt).count
            if count > 0:
                client.delete(collection_name=collection, points_selector=filt)
            return count
        except Exception as e:
            logger.error(f"Qdrant cleanup error for source {source_id}: {e}")
            return 0
    
    async def _cleanup_storage(self, file_id: str) -> bool:
        """Isolated logic for storage cleanup."""
        try:
            return self.storage.delete_file(file_id)
        except Exception as e:
            logger.error(f"Storage cleanup error for source {file_id}: {e}")
            return False


    async def delete_source(self, source_id: str, user_id: uuid.UUID, notebook_id: uuid.UUID) -> SourceDeleteResponse:
        source = await self.source_repo.get_in_notebook(source_id, user_id, notebook_id)
        if not source:
            raise NotFoundException(f"Source {source_id} not found in this notebook")

        # CRITICAL: Reference counting
        usage_count = await self.source_repo.count_by_source_id(source_id, user_id)

        tasks = []
        # Only wipe external Qdrant & Storage if this is the LAST notebook using the file
        if usage_count == 1:
            tasks.append(self._cleanup_qdrant(source_id, user_id))
            if source.source_type == SourceType.UPLOAD and source.source_data.get('imagekit_file_id'):
                tasks.append(self._cleanup_storage(source.source_data['imagekit_file_id']))
        
        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        
        imagekit_failed = False
        if tasks and source.source_type == SourceType.UPLOAD and isinstance(results[-1], Exception):
            imagekit_failed = True        

        await self.source_repo.delete(source)
        
        message = "Source unlinked successfully" if usage_count > 1 else "Source deleted successfully"
        if imagekit_failed:
             message += " (storage deletion failed, manual cleanup may be required)"

        return SourceDeleteResponse(message=message, source_id=source_id)