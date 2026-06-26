"""Source service — supports upload, website, YouTube, topic, note."""

from sqlalchemy.exc import IntegrityError
import uuid
import hashlib
import tempfile
import aiofiles  
import os
import re
import asyncio
from typing import List, Optional, Tuple

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.models import Filter, FieldCondition, MatchValue
from googleapiclient.discovery import build
from urllib.parse import urlparse, urlunparse

from app.core.config import settings
from app.core.ai_clients import get_qdrant_client
from app.core.logger import logger
from app.core.exceptions import NotFoundException, ValidationError, BadRequestException, ConflictException
from app.core.providers.storage import get_storage_provider
from app.core.validator import StringValidator

from app.features.sources.loader import load_yt_bulk, get_topic_urls, load_web, is_supported
from app.features.sources.tasks import index_source_background
from app.features.sources.helpers import collection_exists
from app.features.notebooks.service import NotebookService
from app.features.sources.repository import SourceRepository
from app.features.sources.model import Source, SourceType, SourceStatus
from app.features.sources.schema import (
    SourceUploadResponse, SourceListResponse, SourceResponse,
    SourceDeleteResponse, SourceStatusResponse, NoteCreateRequest,
)

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
        try:
            t.result()
        except Exception as e:
            logger.error(f"Background indexing crashed for {s_id}: {e}", exc_info=True)

    @staticmethod
    def _extract_video_id(url: str) -> str:
        # Improved Regex: Strictly handles the 11-char ID ignoring query params like &t=10s
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", url)
        return match.group(1) if match else url

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Standardizes a URL for consistent ID generation.
        Handles trailing slashes, empty queries, fragments, and case-sensitivity.
        """
        # 1. Clean basic whitespace
        url = url.strip()
        
        # 2. Parse the URL into its core components
        parsed = urlparse(url)
        
        # 3. Clean the path (remove trailing slash)
        path = parsed.path.rstrip('/')
        
        # 4. Clean trailing query separators if there are no actual parameters
        query = parsed.query.rstrip('&?')
        
        # 5. Rebuild the URL
        # We lowercase the scheme and domain (netloc) because HTTP://GEMINI.COM is the same as http://gemini.com
        # We also pass an empty string for the fragment to drop #anchors
        clean_url = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            query,
            ""  # Explicitly drop fragments (e.g., #header-1)
        ))
        
        # Fallback: if stripping the slash left us with an empty URL (unlikely but safe), return original
        return clean_url or url 

    @staticmethod
    def _generate_source_id(identifier: str) -> str:
        return hashlib.md5(identifier.encode("utf-8", errors="ignore")).hexdigest()

    def _create_upload_response(self, source: Source) -> SourceUploadResponse:
        return SourceUploadResponse(
            id=source.id,
            source_id=source.source_id,
            title=source.title,
            source_type=source.source_type,
            status=source.status,
        )

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
                        thumbnails.get('standard', {}).get('url') or 
                        thumbnails.get('maxres', {}).get('url') or 
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

        existing_global = await self.source_repo.get_first_by_source_id(source_id, user_id)
        if existing_global:
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
    
    async def _handle_duplication_bulk(
        self, source_ids: list[str], user_id: uuid.UUID, notebook_id: uuid.UUID
    ) -> dict:
        """
        Takes a list of source_ids and returns a routing dictionary:
        - 'skip': Set of IDs that already exist in this notebook.
        - 'link': List of duplicated Source objects we just linked to this notebook.
        - 'fetch': Set of truly new IDs that need to be scraped/downloaded.
        """
        # 1. Get everything that already exists in this specific notebook (Query 1)
        existing_in_nb = await self.source_repo.get_in_notebook_bulk(source_ids, user_id, notebook_id)
        nb_existing_ids = {s.source_id for s in existing_in_nb}

        # 2. Filter out what we already have in the notebook
        remaining_ids = [sid for sid in source_ids if sid not in nb_existing_ids]
        
        linked_sources = []
        fetch_ids = set(remaining_ids)

        # 3. Check if the remaining IDs exist globally in other notebooks (Query 2)
        if remaining_ids:
            global_matches = await self.source_repo.get_global_bulk(remaining_ids, user_id)
            
            # Map them by ID so we only grab the first match if there are duplicates
            global_map = {s.source_id: s for s in global_matches}

            # 4. Create new links for existing global sources
            links_to_create = []
            for sid, existing_global in global_map.items():
                new_link = Source(
                    notebook_id=notebook_id,
                    user_id=user_id,
                    source_id=sid,
                    source_type=existing_global.source_type,
                    title=existing_global.title,
                    status=existing_global.status,
                    source_data=existing_global.source_data,
                    total_chunks=existing_global.total_chunks,
                    error_message=existing_global.error_message
                )
                links_to_create.append(new_link)
                fetch_ids.remove(sid) # Remove from the fetch queue!

            # Bulk insert the new links
            if links_to_create:
                linked_sources = await self.source_repo.create_many(links_to_create)

        return {
            "skip": nb_existing_ids,
            "link": linked_sources,
            "fetch": fetch_ids
        }

    async def _process_duplication_routing(
        self, url_map: dict[str, str], user_id: uuid.UUID, notebook_id: uuid.UUID
    ) -> tuple[list[SourceUploadResponse], list[str]]:
        """
        Universal routing middleware. 
        Takes a dictionary of {source_id: clean_url}, runs the bulk DB check, 
        auto-links existing global files, and returns the fetch queue.
        """
        responses = []
        routing = await self._handle_duplication_bulk(list(url_map.keys()), user_id, notebook_id)

        # Append responses for sources we just instantly linked
        for linked_source in routing.get("link", []):
            responses.append(self._create_upload_response(linked_source))

        # Filter out the URLs we actually need to hit the network for
        urls_to_fetch = [url_map[sid] for sid in routing.get("fetch", [])]

        return responses, urls_to_fetch

    async def _process_and_save_documents(
        self, doc_definitions: list[dict], user_id: uuid.UUID, notebook_id: uuid.UUID
    ) -> list[SourceUploadResponse]:
        """
        Unified loop to handle database insertion and background task dispatching.
        Relies on the parent methods to have already completed the duplication pre-checks.
        """
        responses = []
        new_sources = []
        
        for doc in doc_definitions:
            source_id = doc.get("source_id") or self._generate_source_id(doc["url"])
                
            source_data = {
                "url": doc["url"],
                "title": doc["title"],
                "content": doc["content"],
                **doc.get("extra_data", {})
            }
            
            source = Source(
                notebook_id=notebook_id,
                user_id=user_id,
                source_id=source_id,
                source_type=doc["source_type"],
                title=doc["title"],
                status=SourceStatus.PROCESSING,
                source_data=source_data,
            )
            new_sources.append(source)
            
        # 2. Bulk Insert with a concurrency safety net
        if new_sources:
            try:
                inserted = await self.source_repo.create_many(new_sources)
                
                for source in inserted:
                    responses.append(self._create_upload_response(source))
                    
                    task = asyncio.create_task(
                        index_source_background(
                            source_id=source.source_id,
                            source_type=source.source_type,
                            user_id=user_id,
                            notebook_id=notebook_id,
                            url=source.source_data["url"],
                            content=source.source_data["content"],
                            title=source.title,
                        )
                    )
                    task.add_done_callback(
                        lambda t, s_id=source.source_id: self._handle_bg_task_result(t, s_id)
                    )
            
            except IntegrityError:
                # If a race condition bypasses the pre-check, the DB throws an IntegrityError.
                # Rollback the session and raise a clear 409 Conflict.
                await self.db.rollback()
                raise ConflictException("A duplicate source insertion was detected due to concurrent requests.")
                
        return responses

    async def _process_web_urls(
        self, 
        urls: list[str], 
        notebook_id: uuid.UUID, 
        user_id: uuid.UUID,
        extra_data: dict | None = None
    ) -> list[SourceUploadResponse]:
        
        valid_urls = [u.strip() for u in urls if u and u.strip()]
        if not valid_urls:
            raise BadRequestException("No valid URLs provided for ingestion.")
        if len(valid_urls) > 10:
            raise BadRequestException("You can only process up to 10 URLs at a time.")

        await self.notebook_service.get_notebook(notebook_id, user_id)
        
        url_map = {} 
        for raw_url in valid_urls:
            clean_url = self._normalize_url(raw_url)
            source_id = self._generate_source_id(clean_url)
            url_map[source_id] = clean_url

        # The Magic 1-Liner
        responses, urls_to_fetch = await self._process_duplication_routing(url_map, user_id, notebook_id)

        if urls_to_fetch:
            web_docs = await load_web(urls_to_fetch)
            if web_docs:
                doc_defs = []
                for doc in web_docs:
                    doc_url = doc.metadata.get("source", "unknown_url").strip()
                    clean_doc_url = self._normalize_url(doc_url)
                    
                    doc_defs.append({
                        "source_id": self._generate_source_id(clean_doc_url),
                        "url": clean_doc_url,
                        "title": doc.metadata.get("title") or clean_doc_url,
                        "content": doc.page_content,
                        "source_type": SourceType.WEBSITE,
                        "extra_data": extra_data or {}
                    })
                    
                new_responses = await self._process_and_save_documents(doc_defs, user_id, notebook_id)
                responses.extend(new_responses)

        return responses

    async def create_upload_source(
        self, file: UploadFile, notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> SourceUploadResponse:
        await self.notebook_service.get_notebook(notebook_id, user_id)

        file_name = file.filename or "unknown"
        if not is_supported(file_name):
            raise ValidationError(f"Unsupported file type: {file_name}")

        ext = os.path.splitext(file_name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_path = temp_file.name

        total_size = 0

        try:
            async with aiofiles.open(temp_path, 'wb') as out_file:
                while chunk := await file.read(1024 * 1024):
                    total_size += len(chunk)
                    # Enforce File Size Limit during stream
                    if total_size > settings.MAX_UPLOAD_SIZE_BYTES:
                        raise BadRequestException("File size exceeds the 25MB maximum limit.")
                    await out_file.write(chunk) 
        except Exception as e:
            os.unlink(temp_path)
            if isinstance(e, BadRequestException):
                raise e
            raise ValidationError(f"Failed to save uploaded file: {e}")

        hasher = hashlib.md5()
        async with aiofiles.open(temp_path, 'rb') as f:
            while chunk := await f.read(8192):
                hasher.update(chunk)
        source_id = hasher.hexdigest()

        try:
            linked_source = await self._handle_duplication(source_id, user_id, notebook_id)
            if linked_source:
                os.unlink(temp_path)
                return self._create_upload_response(linked_source)
        except ConflictException:
            os.unlink(temp_path)
            raise

        try:
            upload_result = self.storage.upload_file(temp_path, file_name, folder=f"sources/{user_id}")
        except Exception as e:
            os.unlink(temp_path)
            logger.error(f"ImageKit upload failed: {e}")
            raise ValidationError(f"Failed to store file: {e}")

        source = Source(
            notebook_id=notebook_id,
            user_id=user_id,
            source_id=source_id,
            source_type=SourceType.UPLOAD,
            title=file_name,
            status=SourceStatus.PROCESSING,
            source_data={
                "file_name": file_name,
                "file_type": ext.lstrip('.'),
                "file_size_bytes": os.path.getsize(temp_path),
                "imagekit_file_id": upload_result.get("file_id"),
                "imagekit_url": upload_result.get("url"),
                "thumbnail_url": upload_result.get("thumbnail_url")
            },
        )
        
        # Guard clause: Delete remote file if local DB crashes to prevent orphaned files
        try:
            source = await self.source_repo.create(source)
        except Exception as e:
            if upload_result.get("file_id"):
                await self._cleanup_storage(upload_result["file_id"])
            os.unlink(temp_path)
            raise ValidationError(f"Database error saving source: {e}")

        asyncio.create_task(
            index_source_background(
                source_id=source_id,
                source_type=SourceType.UPLOAD,
                user_id=user_id,
                notebook_id=notebook_id,
                file_path=temp_path,
                file_name=file_name,
            )
        )

        return self._create_upload_response(source)

    async def create_website_sources(
        self, urls: list[str], notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> list[SourceUploadResponse]:
        """Public method for the web ingestion API route."""
        return await self._process_web_urls(urls, notebook_id, user_id)

    async def create_youtube_sources(
        self, urls: list[str], notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> list[SourceUploadResponse]:
        valid_urls = [u.strip() for u in urls if u and u.strip()]
        if not valid_urls:
            raise BadRequestException("No valid YouTube URLs provided.")
        if len(valid_urls) > 10:
            raise BadRequestException("You can only process up to 10 YouTube URLs at a time.")

        await self.notebook_service.get_notebook(notebook_id, user_id)
        
        url_map = {}
        for raw_url in valid_urls:
            video_id = self._extract_video_id(raw_url)
            standard_url = f"https://www.youtube.com/watch?v={video_id}"
            source_id = self._generate_source_id(standard_url)
            url_map[source_id] = standard_url

        # The Magic 1-Liner
        responses, urls_to_fetch = await self._process_duplication_routing(url_map, user_id, notebook_id)

        if urls_to_fetch:
            metadata_task = self._fetch_youtube_metadata(urls_to_fetch)
            transcripts_task = load_yt_bulk(urls_to_fetch)
            metadata_list, raw_docs = await asyncio.gather(metadata_task, transcripts_task)

            if raw_docs:
                metadata_by_vid = {m["video_id"]: m for m in metadata_list}
                docs_by_url = {}
                for doc in raw_docs:
                    source_url = doc.metadata.get("source")
                    if source_url:
                        docs_by_url.setdefault(source_url, []).append(doc.page_content)

                doc_defs = []
                for url, chunks in docs_by_url.items():
                    video_id = self._extract_video_id(url)
                    meta = metadata_by_vid.get(video_id, {})
                    content = "\n".join(chunks)
                    
                    standard_doc_url = f"https://www.youtube.com/watch?v={video_id}"

                    doc_defs.append({
                        "source_id": self._generate_source_id(standard_doc_url),
                        "url": meta.get("url", standard_doc_url),
                        "title": meta.get("title", f"YouTube Video: {standard_doc_url}"),
                        "content": content,
                        "source_type": SourceType.YOUTUBE,
                        "extra_data": {
                            "video_id": video_id,
                            "embed_url": meta.get("embed_url"),
                            "embed_html": meta.get("embed_html"),
                            "thumbnail_url": meta.get("thumbnail_url"),
                        }
                    })

                new_responses = await self._process_and_save_documents(doc_defs, user_id, notebook_id)
                responses.extend(new_responses)

        return responses

    async def create_topic_source(
        self, topic: str, notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> list[SourceUploadResponse]:
        """Public method for the topic ingestion API route."""
        from app.features.sources.loader import get_topic_urls 

        clean_topic = topic.strip()
        if not clean_topic:
            raise BadRequestException("Topic query cannot be empty.")

        target_urls = await get_topic_urls(clean_topic)
        if not target_urls:
            raise BadRequestException("No web search results found for this topic.")
        
        # Pass the retrieved URLs to the shared engine, injecting the metadata safely
        return await self._process_web_urls(
            urls=target_urls,
            notebook_id=notebook_id,
            user_id=user_id,
            extra_data={"origin_topic_query": clean_topic}
        )

    async def create_note_source(
        self, note_data: NoteCreateRequest, user_id: uuid.UUID
    ) -> SourceUploadResponse:
        notebook = await self.notebook_service.get_notebook(note_data.notebook_id, user_id)
        if not notebook:
            raise NotFoundException("Notebook not found or Access Denied")

        source_id = self._generate_source_id(note_data.content)
        linked_source = await self._handle_duplication(source_id, user_id, note_data.notebook_id)
        if linked_source:
            return self._create_upload_response(linked_source)

        source = await self.source_repo.create(Source(
            notebook_id=note_data.notebook_id,
            user_id=user_id,
            source_id=source_id,
            source_type=SourceType.NOTE,
            title=note_data.title,
            status=SourceStatus.PROCESSING,
            source_data={"content": note_data.content},
        ))

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

        return self._create_upload_response(source)

    async def retry_indexing(self, source_id: str, user_id: uuid.UUID, notebook_id: uuid.UUID) -> SourceStatusResponse:
        source = await self.source_repo.get_in_notebook(source_id, user_id, notebook_id)
        if not source:
            raise NotFoundException(f"Source {source_id} not found")
            
        if source.status != SourceStatus.ERROR:
            raise BadRequestException("Only failed sources can be retried.")

        # 1. Clean up any partial vectors in Qdrant just in case it failed halfway
        await self._cleanup_qdrant(source_id, user_id)

        # 2. Reset status in DB
        await self.source_repo.update_status(source_id, notebook_id, SourceStatus.PROCESSING, error_message=None)

        # 3. Handle File Downloads vs Stored Content
        if source.source_type == SourceType.UPLOAD:
            # For uploaded files, the temp file was deleted. We must fetch it from ImageKit.
            raise BadRequestException(
                "Retrying direct file uploads is not yet supported. Please delete and re-upload the file."
            )
        else:
            # For Web, YT, and Notes, the content is already safely in the DB!
            asyncio.create_task(
                index_source_background(
                    source_id=source.source_id,
                    source_type=source.source_type,
                    user_id=user_id,
                    notebook_id=notebook_id,
                    url=source.source_data.get("url"),
                    content=source.source_data.get("content"),
                    title=source.title,
                )
            )

        return SourceStatusResponse(
            source_id=source_id,
            status=SourceStatus.PROCESSING,
            total_chunks=0
        )

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

    async def get_source(self, source_id: str, user_id: uuid.UUID,notebook_id: uuid.UUID) -> Source:
        source = await self.source_repo.get_source_by_source_id_notebook_id(source_id, user_id,notebook_id)
        if not source:
            raise NotFoundException(f"Source not found or access denied")
        return source

    async def get_status(self, source_id: str, user_id: uuid.UUID,notebook_id: uuid.UUID) -> SourceStatusResponse:
        source = await self.get_source(source_id, user_id,notebook_id)
        if not source:
            raise NotFoundException(f"Source not found or access denied")
        return SourceStatusResponse(
            source_id=source_id,
            status=source.status,
            error_message=source.error_message,
            total_chunks=source.total_chunks,  
        )  

    async def _cleanup_qdrant(self, source_id: str, user_id: uuid.UUID) -> None:
        try:
            client = get_qdrant_client()
            collection = settings.QDRANT_COLLECTION
            if not collection_exists(client, collection):
                return
                
            client.delete(
                collection_name=collection, 
                points_selector=Filter(must=[
                    FieldCondition(key='source_id', match=MatchValue(value=source_id)),
                    FieldCondition(key='user_id', match=MatchValue(value=str(user_id))),
                ])
            )
        except Exception as e:
            logger.error(f"Qdrant cleanup error for source {source_id}: {e}")
    
    async def _cleanup_storage(self, file_id: str) -> bool:
        try:
            return self.storage.delete_file(file_id)
        except Exception as e:
            logger.error(f"Storage cleanup error for source {file_id}: {e}")
            return False

    async def delete_source(self, source_id: str, user_id: uuid.UUID, notebook_id: uuid.UUID) -> SourceDeleteResponse:
        source = await self.source_repo.get_in_notebook(source_id, user_id, notebook_id)
        if not source:
            raise NotFoundException(f"Source {source_id} not found in this notebook")

        usage_count = await self.source_repo.count_by_source_id(source_id, user_id)

        tasks = []
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