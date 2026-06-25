"""Artifact service — orchestrates source resolution, context building, compression, generation, and persistence."""

import uuid
from typing import Any, Optional, List
from fastapi.background import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import logger
from app.core.exceptions import NotFoundException, BadRequestException

from app.features.notebooks.service import NotebookService
from app.features.sources.repository import SourceRepository
from app.features.artifacts.repository import ArtifactRepository
from app.features.artifacts.model import Artifact
from app.features.artifacts.tasks import run_generation_task
from app.features.artifacts.schema import (
    ArtifactType,
    ArtifactStatus,
    BaseArtifactRequest,
    QuizCreateRequest,
    FlashcardCreateRequest,
    FAQCreateRequest,
    StudyGuideCreateRequest,
    SummaryCreateRequest,
    MindMapCreateRequest,
    SlideDeckCreateRequest,
    SourceFilterInfo,
    ArtifactResponse,
    ArtifactListResponse,
)


class ArtifactService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.artifact_repo = ArtifactRepository(db)
        self.notebook_service = NotebookService(db)
        self.source_repo = SourceRepository(db)

    # ── Source Resolution ─────────────────────────────────────────────────────────

    async def _resolve_source_ids(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, excluded_source_ids: List[str]
    ) -> tuple[List[str], SourceFilterInfo]:
        """Resolve active source IDs for artifact generation."""
        all_source_ids = await self.source_repo.get_source_ids_for_notebook(notebook_id, user_id)

        if not all_source_ids:
            raise NotFoundException("No sources available for this notebook")

        excluded_set = set(excluded_source_ids)
        resolved = [sid for sid in all_source_ids if sid not in excluded_set]

        if not resolved:
            raise BadRequestException("No sources selected for artifact generation")

        filter_info = SourceFilterInfo(
            excluded_source_ids=list(excluded_set),
            resolved_source_ids=resolved,
        )

        return resolved, filter_info

    # ── Core Generation Pipeline ──────────────────────────────────────────────────

    async def _create_processing_artifact(
        self,
        notebook_id: uuid.UUID,
        user_id: uuid.UUID,
        artifact_type: ArtifactType,
        request: BaseArtifactRequest,
        options: dict[str, Any],
        background_tasks: BackgroundTasks,
    ) -> Artifact:
        """Phase 1: Validate, resolve sources, save PROCESSING state, and enqueue task."""
        
        # 1. Validate notebook ownership synchronously (fail-fast)
        await self.notebook_service.get_notebook(notebook_id, user_id)

        # 2. Resolve source scope synchronously (fail-fast if invalid sources)
        resolved_ids, filter_info = await self._resolve_source_ids(
            notebook_id, user_id, request.excluded_source_ids
        )

        # 3. Create placeholder artifact
        artifact = Artifact(
            notebook_id=notebook_id,
            user_id=user_id,
            artifact_type=artifact_type,
            status=ArtifactStatus.PROCESSING,  
            title=f"Generating {artifact_type.value.title()}...",
            options_json={
                "prompt": request.prompt,
                **options,
            },
            included_sources=resolved_ids,
            content_json={},
        )

        saved_artifact = await self.artifact_repo.create(artifact)
        logger.info(f"Artifact saved with PROCESSING status: {saved_artifact}")
        
        # 4. Enqueue the heavy lifting to the background worker
        background_tasks.add_task(
            run_generation_task,
            artifact_id=saved_artifact.id,
            notebook_id=notebook_id,
            user_id=user_id,
            artifact_type=artifact_type,
            request_prompt=request.prompt,
            options=options,
            resolved_ids=resolved_ids
        )

        return saved_artifact

    # ── Public API Methods ────────────────────────────────────────────────────────

    async def create_quiz(
        self, 
        notebook_id: uuid.UUID, 
        user_id: uuid.UUID, 
        request: QuizCreateRequest,
        background_tasks: BackgroundTasks,
    ) -> Artifact:
        options = {
            "question_count": request.number_of_questions,
            "difficulty": request.difficulty.value,
        }
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.QUIZ, request, options, background_tasks
        )

    async def create_flashcards(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: FlashcardCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        options = {"card_count": request.number_of_cards}
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.FLASHCARDS, request, options, background_tasks
        )

    async def create_faqs(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: FAQCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        options = {"faq_count": request.number_of_faqs}
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.FAQ, request, options, background_tasks
        )

    async def create_study_guide(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: StudyGuideCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        options = {"size": request.size.value}
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.STUDY_GUIDE, request, options, background_tasks
        )

    async def create_summary(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: SummaryCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.SUMMARY, request, {}, background_tasks
        )

    async def create_mindmap(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: MindMapCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.MINDMAP, request, {}, background_tasks
        )

    async def create_slide_deck(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: SlideDeckCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        options = {"slide_count": request.number_of_slides}
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.SLIDE_DECK, request, options, background_tasks
        )
    
    async def retry_artifact_generation(
        self,
        notebook_id: uuid.UUID,
        artifact_id: uuid.UUID,
        user_id: uuid.UUID,
        background_tasks: BackgroundTasks,
    ) -> Artifact:
        """Retry a failed artifact generation."""
        # 1. Fetch the artifact
        artifact = await self.get_artifact(notebook_id, artifact_id, user_id)
        
        if artifact.status != ArtifactStatus.ERROR:
            raise BadRequestException("Only failed artifacts can be retried.")

        # 2. Reset status
        artifact.status = ArtifactStatus.PROCESSING
        artifact.error_message = None
        await self.artifact_repo.update(artifact)
        
        # 3. Re-enqueue the same generation task
        background_tasks.add_task(
            run_generation_task,
            artifact_id=artifact.id,
            notebook_id=notebook_id,
            user_id=user_id,
            artifact_type=artifact.artifact_type,
            request_prompt=artifact.options_json.get("prompt"),
            options=artifact.options_json,
            resolved_ids=artifact.included_sources
        )

        logger.info(f"Retrying generation for artifact {artifact_id}")
        return artifact

    # ── Read / Delete ─────────────────────────────────────────────────────────────

    async def list_artifacts(
        self,
        notebook_id: uuid.UUID,
        user_id: uuid.UUID,
        artifact_type: Optional[ArtifactType] = None,
        status_filter: Optional[ArtifactStatus] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ArtifactListResponse:
        """List artifacts for a notebook with optional filters."""
        await self.notebook_service.get_notebook(notebook_id, user_id)
        
        artifacts = await self.artifact_repo.list_by_notebook(
            notebook_id, user_id, artifact_type=artifact_type, status=status_filter, limit=limit, offset=offset
        )
        total = await self.artifact_repo.count_by_notebook(
            notebook_id, user_id, artifact_type=artifact_type, status=status_filter
        )
        
        return ArtifactListResponse(
            artifacts=[ArtifactResponse.model_validate(a) for a in artifacts],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get_artifact(
        self, notebook_id: uuid.UUID, artifact_id: uuid.UUID, user_id: uuid.UUID
    ) -> Artifact:
        """Get a specific artifact by ID."""
        await self.notebook_service.get_notebook(notebook_id, user_id)
        artifact = await self.artifact_repo.get_by_id(artifact_id, notebook_id, user_id)
        if not artifact:
            raise NotFoundException(f"Artifact {artifact_id} not found")
        return artifact

    async def delete_artifact(
        self, notebook_id: uuid.UUID, artifact_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        """Delete a single artifact."""
        artifact = await self.get_artifact(notebook_id, artifact_id, user_id)
        await self.artifact_repo.delete(artifact)
        logger.info(f"Artifact deleted: id={artifact_id}, notebook={notebook_id}")

    async def delete_all_artifacts(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        """Delete all artifacts for a notebook."""
        await self.notebook_service.get_notebook(notebook_id, user_id)
        count = await self.artifact_repo.delete_by_notebook(notebook_id, user_id)
        logger.info(f"Deleted {count} artifacts for notebook={notebook_id}")