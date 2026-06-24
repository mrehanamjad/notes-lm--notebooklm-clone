# # """Artifact service — orchestrates source resolution, context building, generation, and persistence."""

# # import uuid
# # from typing import Any

# # from sqlalchemy.ext.asyncio import AsyncSession

# # from app.core.ai_clients import get_llm
# # from app.core.logger import logger
# # from app.core.exceptions import NotFoundException, BadRequestException

# # from app.features.notebooks.service import NotebookService
# # from app.features.sources.repository import SourceRepository
# # from app.features.artifacts.repository import ArtifactRepository
# # from app.features.artifacts.model import Artifact, ArtifactType, ArtifactStatus
# # from app.features.artifacts.schema import (
# #     BaseArtifactRequest,
# #     QuizCreateRequest,
# #     FlashcardCreateRequest,
# #     FAQCreateRequest,
# #     SourceFilterInfo,
# #     ArtifactResponse,
# #     ArtifactListResponse,
# # )
# # from app.features.artifacts.context_builder import ArtifactContextBuilder
# # from app.features.artifacts.generators import get_generator


# # class ArtifactService:
# #     def __init__(self, db: AsyncSession):
# #         self.db = db
# #         self.artifact_repo = ArtifactRepository(db)
# #         self.notebook_service = NotebookService(db)
# #         self.source_repo = SourceRepository(db)

# #     # ── Source Resolution (mirrors chat flow) ─────────────────────────────────

# #     async def _resolve_source_ids(
# #         self, notebook_id: uuid.UUID, user_id: uuid.UUID, excluded_source_ids: list[str]
# #     ) -> tuple[list[str], SourceFilterInfo]:
# #         """Resolve active source IDs for artifact generation.

# #         Follows the same contract as chat:
# #           1. Get all READY source IDs for the notebook/user
# #           2. Subtract excluded source IDs
# #           3. Error if no sources remain
# #         """
# #         all_source_ids = await self.source_repo.get_source_ids_for_notebook(notebook_id, user_id)

# #         if not all_source_ids:
# #             raise NotFoundException("No sources available for this notebook")

# #         excluded_set = set(excluded_source_ids)
# #         resolved = [sid for sid in all_source_ids if sid not in excluded_set]

# #         if not resolved:
# #             raise BadRequestException("No sources selected for artifact generation")

# #         filter_info = SourceFilterInfo(
# #             excluded_source_ids=list(excluded_set),
# #             resolved_source_ids=resolved,
# #         )

# #         return resolved, filter_info

# #     # ── Core Generation Pipeline ──────────────────────────────────────────────

# #     async def _generate_artifact(
# #         self,
# #         notebook_id: uuid.UUID,
# #         user_id: uuid.UUID,
# #         artifact_type: ArtifactType,
# #         request: BaseArtifactRequest,
# #         options: dict[str, Any],
# #     ) -> Artifact:
# #         """Full artifact generation pipeline:
# #         1. Validate notebook ownership
# #         2. Resolve source scope
# #         3. Build artifact-specific context
# #         4. Generate structured output via LLM
# #         5. Save artifact to DB
# #         """
# #         # 1. Validate notebook ownership
# #         await self.notebook_service.get_notebook(notebook_id, user_id)

# #         # 2. Resolve source scope (mirrors chat)
# #         resolved_ids, filter_info = await self._resolve_source_ids(
# #             notebook_id, user_id, request.excluded_source_ids
# #         )

# #         # 3. Build artifact-specific context
# #         context_result = ArtifactContextBuilder.build_context(
# #             user_id=user_id,
# #             resolved_source_ids=resolved_ids,
# #             topic=request.topic,
# #             prompt=request.prompt,
# #             artifact_type=artifact_type.value,
# #         )

# #         logger.info(
# #             f"Artifact context built: mode={context_result.mode_used}, "
# #             f"chunks={context_result.total_chunks}, type={artifact_type.value}"
# #         )

# #         # 4. Generate structured output
# #         generator = get_generator(artifact_type)
# #         llm = get_llm()

# #         try:
# #             content = generator.generate(llm, context_result.context_text, options)
# #             status = ArtifactStatus.READY
# #             error_message = None
# #             title = content.get("title", f"{artifact_type.value.title()} Artifact")
# #         except Exception as e:
# #             logger.error(f"Artifact generation failed: {e}", exc_info=True)
# #             content = {}
# #             status = ArtifactStatus.FAILED
# #             error_message = str(e)
# #             title = f"Failed {artifact_type.value.title()} Artifact"

# #         # 5. Save to DB
# #         artifact = Artifact(
# #             notebook_id=notebook_id,
# #             user_id=user_id,
# #             artifact_type=artifact_type,
# #             status=status,
# #             title=title,
# #             generation_prompt=request.prompt,
# #             options_json=options,
# #             source_filter_json=filter_info.model_dump(),
# #             content_json=content,
# #             error_message=error_message,
# #         )

# #         saved = await self.artifact_repo.create(artifact)
# #         logger.info(f"Artifact saved: id={saved.id}, type={artifact_type.value}, status={status.value}")

# #         # Re-raise if generation failed so the API returns an error
# #         if status == ArtifactStatus.FAILED:
# #             from app.core.exceptions import InternalServerError
# #             raise InternalServerError(
# #                 message="Failed to generate artifact. Please try again.",
# #                 details={"artifact_id": str(saved.id), "reason": error_message},
# #             )

# #         return saved

# #     # ── Public API Methods ────────────────────────────────────────────────────

# #     async def create_quiz(
# #         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: QuizCreateRequest
# #     ) -> Artifact:
# #         options = {
# #             "number_of_questions": request.number_of_questions,
# #             "difficulty": request.difficulty.value,
# #             "topic": request.topic,
# #             "prompt": request.prompt,
# #         }
# #         return await self._generate_artifact(
# #             notebook_id, user_id, ArtifactType.QUIZ, request, options
# #         )

# #     async def create_flashcards(
# #         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: FlashcardCreateRequest
# #     ) -> Artifact:
# #         options = {
# #             "number_of_cards": request.number_of_cards,
# #             "topic": request.topic,
# #             "prompt": request.prompt,
# #         }
# #         return await self._generate_artifact(
# #             notebook_id, user_id, ArtifactType.FLASHCARDS, request, options
# #         )

# #     async def create_faqs(
# #         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: FAQCreateRequest
# #     ) -> Artifact:
# #         options = {
# #             "number_of_faqs": request.number_of_faqs,
# #             "topic": request.topic,
# #             "prompt": request.prompt,
# #         }
# #         return await self._generate_artifact(
# #             notebook_id, user_id, ArtifactType.FAQ, request, options
# #         )

# #     # ── Read / Delete ─────────────────────────────────────────────────────────

# #     async def list_artifacts(
# #         self, notebook_id: uuid.UUID, user_id: uuid.UUID
# #     ) -> ArtifactListResponse:
# #         await self.notebook_service.get_notebook(notebook_id, user_id)
# #         artifacts = await self.artifact_repo.list_by_notebook(notebook_id, user_id)
# #         total = await self.artifact_repo.count_by_notebook(notebook_id, user_id)
# #         return ArtifactListResponse(
# #             artifacts=[ArtifactResponse.model_validate(a) for a in artifacts],
# #             total=total,
# #         )

# #     async def get_artifact(
# #         self, notebook_id: uuid.UUID, artifact_id: uuid.UUID, user_id: uuid.UUID
# #     ) -> Artifact:
# #         await self.notebook_service.get_notebook(notebook_id, user_id)
# #         artifact = await self.artifact_repo.get_by_id(artifact_id, notebook_id, user_id)
# #         if not artifact:
# #             raise NotFoundException(f"Artifact {artifact_id} not found")
# #         return artifact

# #     async def delete_artifact(
# #         self, notebook_id: uuid.UUID, artifact_id: uuid.UUID, user_id: uuid.UUID
# #     ) -> None:
# #         artifact = await self.get_artifact(notebook_id, artifact_id, user_id)
# #         await self.artifact_repo.delete(artifact)
# #         logger.info(f"Artifact deleted: id={artifact_id}, notebook={notebook_id}")










# # service.py
# """Artifact service — orchestrates source resolution, context building, compression, generation, and persistence."""

# import uuid
# import json
# from typing import Any, Optional

# from sqlalchemy.ext.asyncio import AsyncSession

# from app.core.ai_clients import get_llm, get_embeddings
# from app.core.logger import logger
# from app.core.exceptions import NotFoundException, BadRequestException, InternalServerError

# from app.features.notebooks.service import NotebookService
# from app.features.sources.repository import SourceRepository
# from app.features.artifacts.repository import ArtifactRepository
# from app.features.artifacts.model import Artifact, ArtifactType, ArtifactStatus
# from app.features.artifacts.schema import (
#     BaseArtifactRequest,
#     QuizCreateRequest,
#     FlashcardCreateRequest,
#     FAQCreateRequest,
#     StudyGuideCreateRequest,
#     SummaryCreateRequest,
#     MindMapCreateRequest,
#     SlideDeckCreateRequest,
#     SourceFilterInfo,
#     ArtifactResponse,
#     ArtifactListResponse,
# )
# from app.features.artifacts.artifact_context_builder import (
#     ArtifactContextBuilder,
#     ContextResult,
# )
# from app.features.artifacts.artifact_evidence_compressor import ArtifactEvidenceCompressor
# from app.features.artifacts.artifact_prompt_builder import ArtifactPromptBuilder
# from app.features.artifacts.artifact_models import EvidencePack


# class ArtifactService:
#     def __init__(self, db: AsyncSession):
#         self.db = db
#         self.artifact_repo = ArtifactRepository(db)
#         self.notebook_service = NotebookService(db)
#         self.source_repo = SourceRepository(db)

#     # ── Source Resolution ─────────────────────────────────────────────────────────

#     async def _resolve_source_ids(
#         self, notebook_id: uuid.UUID, user_id: uuid.UUID, excluded_source_ids: list[str]
#     ) -> tuple[list[str], SourceFilterInfo]:
#         """Resolve active source IDs for artifact generation."""
#         all_source_ids = await self.source_repo.get_source_ids_for_notebook(notebook_id, user_id)

#         if not all_source_ids:
#             raise NotFoundException("No sources available for this notebook")

#         excluded_set = set(excluded_source_ids)
#         resolved = [sid for sid in all_source_ids if sid not in excluded_set]

#         if not resolved:
#             raise BadRequestException("No sources selected for artifact generation")

#         filter_info = SourceFilterInfo(
#             excluded_source_ids=list(excluded_set),
#             resolved_source_ids=resolved,
#         )

#         return resolved, filter_info

#     # ── Core Generation Pipeline ──────────────────────────────────────────────────

#     async def _generate_artifact(
#         self,
#         notebook_id: uuid.UUID,
#         user_id: uuid.UUID,
#         artifact_type: ArtifactType,
#         request: BaseArtifactRequest,
#         options: dict[str, Any],
#     ) -> Artifact:
#         """
#         Full artifact generation pipeline:

#         1. Validate notebook ownership
#         2. Resolve source scope
#         3. Build artifact-specific context (V4)
#         4. Compress context to evidence pack (token savings)
#         5. Generate structured output via LLM
#         6. Save artifact to DB
#         """
#         # 1. Validate notebook ownership
#         await self.notebook_service.get_notebook(notebook_id, user_id)

#         # 2. Resolve source scope
#         resolved_ids, filter_info = await self._resolve_source_ids(
#             notebook_id, user_id, request.excluded_source_ids
#         )

#         # 3. Build artifact-specific context using V4
#         context_result = ArtifactContextBuilder.build_context(
#             user_id=user_id,
#             resolved_source_ids=resolved_ids,
#             artifact_type=artifact_type.value,
#             topic=request.topic,
#             prompt=request.prompt,
#         )

#         logger.info(
#             f"Artifact context built: mode={context_result.mode_used}, "
#             f"chunks={context_result.total_chunks}, "
#             f"tokens={context_result.total_estimated_tokens}, "
#             f"type={artifact_type.value}"
#         )

#         # 4. Compress context to evidence pack (optional but recommended for Groq)
#         llm = get_llm()
#         compressor = ArtifactEvidenceCompressor(llm)
        
#         try:
#             evidence_pack = await compressor.compress(
#                 artifact_type=artifact_type.value,
#                 context_text=context_result.context_text,
#                 topic=request.topic,
#                 prompt=request.prompt,
#             )
            
#             logger.info(
#                 f"Evidence compression complete: "
#                 f"facts={len(evidence_pack.facts)}, "
#                 f"concepts={len(evidence_pack.concepts)}, "
#                 f"definitions={len(evidence_pack.definitions)}"
#             )
            
#             evidence_pack_json = json.dumps(evidence_pack.model_dump(), indent=2)
#         except Exception as e:
#             logger.warning(f"Evidence compression failed, using raw context: {e}")
#             evidence_pack_json = context_result.context_text

#         # 5. Generate structured output
#         generation_prompt = ArtifactPromptBuilder.build_generation_prompt(
#             artifact_type=artifact_type.value,
#             evidence_pack_json=evidence_pack_json,
#             topic=request.topic,
#             prompt=request.prompt,
#             options=options,
#         )

#         try:
#             # Use structured output from LLM
#             content = await self._generate_structured_content(
#                 llm=llm,
#                 artifact_type=artifact_type,
#                 prompt=generation_prompt,
#             )
#             status = ArtifactStatus.READY
#             error_message = None
#             title = content.get("title", f"{artifact_type.value.title()} Artifact")
#         except Exception as e:
#             logger.error(f"Artifact generation failed: {e}", exc_info=True)
#             content = {}
#             status = ArtifactStatus.FAILED
#             error_message = str(e)
#             title = f"Failed {artifact_type.value.title()} Artifact"

#         # 6. Save to DB
#         artifact = Artifact(
#             notebook_id=notebook_id,
#             user_id=user_id,
#             artifact_type=artifact_type,
#             status=status,
#             title=title,
#             generation_prompt=request.prompt,
#             options_json=options,
#             source_filter_json=filter_info.model_dump(),
#             content_json=content,
#             error_message=error_message,
#             context_metadata={
#                 "mode_used": context_result.mode_used,
#                 "total_chunks": context_result.total_chunks,
#                 "total_estimated_tokens": context_result.total_estimated_tokens,
#                 "source_refs": [
#                     {
#                         "source_id": ref.source_id,
#                         "file_name": ref.file_name,
#                         "page_number": ref.page_number,
#                         "chunk_index": ref.chunk_index,
#                         "similarity_score": ref.similarity_score,
#                     }
#                     for ref in context_result.source_refs
#                 ],
#             },
#         )

#         saved = await self.artifact_repo.create(artifact)
#         logger.info(f"Artifact saved: id={saved.id}, type={artifact_type.value}, status={status.value}")

#         if status == ArtifactStatus.FAILED:
#             raise InternalServerError(
#                 message="Failed to generate artifact. Please try again.",
#                 details={"artifact_id": str(saved.id), "reason": error_message},
#             )

#         return saved

#     async def _generate_structured_content(
#         self,
#         llm: Any,
#         artifact_type: ArtifactType,
#         prompt: str,
#     ) -> dict[str, Any]:
#         """Generate structured content using LLM with schema."""
#         from app.features.artifacts.artifact_models import (
#             QuizArtifact,
#             FlashcardsArtifact,
#             FAQArtifact,
#             StudyGuideArtifact,
#             SummaryArtifact,
#             MindMapArtifact,
#             SlideDeckArtifact,
#         )

#         schema_map = {
#             ArtifactType.QUIZ: QuizArtifact,
#             ArtifactType.FLASHCARDS: FlashcardsArtifact,
#             ArtifactType.FAQ: FAQArtifact,
#             ArtifactType.STUDY_GUIDE: StudyGuideArtifact,
#             ArtifactType.SUMMARY: SummaryArtifact,
#             ArtifactType.MINDMAP: MindMapArtifact,
#             ArtifactType.SLIDE_DECK: SlideDeckArtifact,
#         }

#         schema = schema_map.get(artifact_type, SummaryArtifact)

#         try:
#             # Using with_structured_output if available
#             if hasattr(llm, "with_structured_output"):
#                 structured_llm = llm.with_structured_output(schema)
#                 result = structured_llm.invoke(prompt)
#                 return result.model_dump() if hasattr(result, "model_dump") else result
#             else:
#                 # Fallback: text generation + JSON parsing
#                 raw = await llm.ainvoke(prompt)
#                 # Parse JSON from response
#                 import re
#                 json_match = re.search(r"```json\s*(.*?)\s*```", raw.content, re.DOTALL)
#                 if json_match:
#                     import json
#                     return json.loads(json_match.group(1))
#                 return {"raw": raw.content}
#         except Exception as e:
#             logger.error(f"Structured generation failed: {e}")
#             raise

#     # ── Public API Methods ────────────────────────────────────────────────────────

#     async def create_quiz(
#         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: QuizCreateRequest
#     ) -> Artifact:
#         options = {
#             "question_count": request.number_of_questions,
#             "difficulty": request.difficulty.value,
#             "topic": request.topic,
#             "prompt": request.prompt,
#         }
#         return await self._generate_artifact(
#             notebook_id, user_id, ArtifactType.QUIZ, request, options
#         )

#     async def create_flashcards(
#         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: FlashcardCreateRequest
#     ) -> Artifact:
#         options = {
#             "card_count": request.number_of_cards,
#             "topic": request.topic,
#             "prompt": request.prompt,
#         }
#         return await self._generate_artifact(
#             notebook_id, user_id, ArtifactType.FLASHCARDS, request, options
#         )

#     async def create_faqs(
#         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: FAQCreateRequest
#     ) -> Artifact:
#         options = {
#             "faq_count": request.number_of_faqs,
#             "topic": request.topic,
#             "prompt": request.prompt,
#         }
#         return await self._generate_artifact(
#             notebook_id, user_id, ArtifactType.FAQ, request, options
#         )

#     async def create_study_guide(
#         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: StudyGuideCreateRequest
#     ) -> Artifact:
#         options = {
#             "topic": request.topic,
#             "prompt": request.prompt,
#         }
#         return await self._generate_artifact(
#             notebook_id, user_id, ArtifactType.STUDY_GUIDE, request, options
#         )

#     async def create_summary(
#         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: SummaryCreateRequest
#     ) -> Artifact:
#         options = {
#             "topic": request.topic,
#             "prompt": request.prompt,
#         }
#         return await self._generate_artifact(
#             notebook_id, user_id, ArtifactType.SUMMARY, request, options
#         )

#     async def create_mindmap(
#         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: MindMapCreateRequest
#     ) -> Artifact:
#         options = {
#             "topic": request.topic,
#             "prompt": request.prompt,
#         }
#         return await self._generate_artifact(
#             notebook_id, user_id, ArtifactType.MINDMAP, request, options
#         )

#     async def create_slide_deck(
#         self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: SlideDeckCreateRequest
#     ) -> Artifact:
#         options = {
#             "slide_count": request.number_of_slides,
#             "topic": request.topic,
#             "prompt": request.prompt,
#         }
#         return await self._generate_artifact(
#             notebook_id, user_id, ArtifactType.SLIDE_DECK, request, options
#         )

#     # ── Read / Delete ─────────────────────────────────────────────────────────────

#     async def list_artifacts(
#         self, notebook_id: uuid.UUID, user_id: uuid.UUID
#     ) -> ArtifactListResponse:
#         await self.notebook_service.get_notebook(notebook_id, user_id)
#         artifacts = await self.artifact_repo.list_by_notebook(notebook_id, user_id)
#         total = await self.artifact_repo.count_by_notebook(notebook_id, user_id)
#         return ArtifactListResponse(
#             artifacts=[ArtifactResponse.model_validate(a) for a in artifacts],
#             total=total,
#         )

#     async def get_artifact(
#         self, notebook_id: uuid.UUID, artifact_id: uuid.UUID, user_id: uuid.UUID
#     ) -> Artifact:
#         await self.notebook_service.get_notebook(notebook_id, user_id)
#         artifact = await self.artifact_repo.get_by_id(artifact_id, notebook_id, user_id)
#         if not artifact:
#             raise NotFoundException(f"Artifact {artifact_id} not found")
#         return artifact

#     async def delete_artifact(
#         self, notebook_id: uuid.UUID, artifact_id: uuid.UUID, user_id: uuid.UUID
#     ) -> None:
#         artifact = await self.get_artifact(notebook_id, artifact_id, user_id)
#         await self.artifact_repo.delete(artifact)
#         logger.info(f"Artifact deleted: id={artifact_id}, notebook={notebook_id}")










# service.py
"""Artifact service — orchestrates source resolution, context building, compression, generation, and persistence."""

from app.features.artifacts.tasks import run_generation_task
from fastapi.background import BackgroundTasks
import uuid
import json
from typing import Any, Optional, List

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ai_clients import get_llm
from app.core.logger import logger
from app.core.exceptions import NotFoundException, BadRequestException, InternalServerError

from app.features.notebooks.service import NotebookService
from app.features.sources.repository import SourceRepository
from app.features.artifacts.repository import ArtifactRepository
from app.features.artifacts.model import Artifact, ArtifactType, ArtifactStatus
from app.features.artifacts.schema import (
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
    QuizOutput,
    FlashcardOutput,
    FAQOutput,
    SummaryOutput,
    StudyGuideOutput,
    MindMapOutput,
    SlideDeckOutput,
)
from app.features.artifacts.artifact_context_builder import (
    ArtifactContextBuilder,
    ContextResult,
)
from app.features.artifacts.artifact_evidence_compressor import ArtifactEvidenceCompressor
from app.features.artifacts.artifact_prompt_builder import ArtifactPromptBuilder


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
        """Resolve active source IDs for artifact generation.
        
        Follows the same contract as chat:
          1. Get all READY source IDs for the notebook/user
          2. Subtract excluded source IDs
          3. Error if no sources remain
        """
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
                "topic": request.topic,
                **options,
            },
            included_sources=resolved_ids,
            content_json={},
        )

        saved_artifact = await self.artifact_repo.create(artifact)
        logger.info(f"Artifact saved with PROCESSING status: {saved_artifact}")
        # 4. Enqueue the heavy lifting to the new tasks.py worker
        background_tasks.add_task(
            run_generation_task,
            artifact_id=saved_artifact.id,
            notebook_id=notebook_id,
            user_id=user_id,
            artifact_type=artifact_type,
            request_topic=request.topic,    # Passed explicitly to avoid Pydantic serialization issues
            request_prompt=request.prompt,  # Passed explicitly
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
        options = {
            "card_count": request.number_of_cards,
        }
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.FLASHCARDS, request, options, background_tasks
        )

    async def create_faqs(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: FAQCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        options = {
            "faq_count": request.number_of_faqs,
        }
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.FAQ, request, options, background_tasks
        )

    async def create_study_guide(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: StudyGuideCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        options = {
            "size": request.size.value,
        }
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.STUDY_GUIDE, request, options, background_tasks
        )

    async def create_summary(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: SummaryCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        options = {}
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.SUMMARY, request, options, background_tasks
        )

    async def create_mindmap(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: MindMapCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        options = {}
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.MINDMAP, request, options, background_tasks
        )

    async def create_slide_deck(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, request: SlideDeckCreateRequest, background_tasks: BackgroundTasks
    ) -> Artifact:
        options = {
            "slide_count": request.number_of_slides,
        }
        return await self._create_processing_artifact(
            notebook_id, user_id, ArtifactType.SLIDE_DECK, request, options, background_tasks
        )

    # ── Read / Delete ─────────────────────────────────────────────────────────────

    async def list_artifacts(
        self,
        notebook_id: uuid.UUID,
        user_id: uuid.UUID,
        artifact_type: Optional[ArtifactType] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ArtifactListResponse:
        """List artifacts for a notebook with optional filters."""
        await self.notebook_service.get_notebook(notebook_id, user_id)
        
        artifacts = await self.artifact_repo.list_by_notebook(
            notebook_id, user_id, artifact_type=artifact_type, limit=limit, offset=offset
        )
        total = await self.artifact_repo.count_by_notebook(
            notebook_id, user_id, artifact_type=artifact_type
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