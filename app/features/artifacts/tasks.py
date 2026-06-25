import uuid
import json
import re
from typing import Any

from app.core.logger import logger
from app.database.session import AsyncSessionLocal
from app.core.ai_clients import get_llm

from app.features.artifacts.repository import ArtifactRepository
from app.features.artifacts.artifact_context_builder import ArtifactContextBuilder
from app.features.artifacts.artifact_evidence_compressor import ArtifactEvidenceCompressor
from app.features.artifacts.artifact_prompt_builder import ArtifactPromptBuilder

from app.features.artifacts.schema import (
    ArtifactType,
    ArtifactStatus,
    QuizArtifact,
    FlashcardsArtifact,
    FAQArtifact,
    SummaryArtifact,
    StudyGuideArtifact,
    MindMapArtifact,
    SlideDeckArtifact,
)


async def generate_structured_content(
    llm: Any,
    artifact_type: ArtifactType,
    prompt: str,
) -> dict[str, Any]:
    """Generate structured content using LLM with schema."""
    schema_map = {
        ArtifactType.QUIZ: QuizArtifact,
        ArtifactType.FLASHCARDS: FlashcardsArtifact,
        ArtifactType.FAQ: FAQArtifact,
        ArtifactType.STUDY_GUIDE: StudyGuideArtifact,
        ArtifactType.SUMMARY: SummaryArtifact,
        ArtifactType.MINDMAP: MindMapArtifact,
        ArtifactType.SLIDE_DECK: SlideDeckArtifact,
    }

    schema = schema_map.get(artifact_type, SummaryArtifact)

    try:
        # Using with_structured_output if available
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(schema)
            result = await structured_llm.ainvoke(prompt)
            return result.model_dump() if hasattr(result, "model_dump") else result
        else:
            # Fallback: text generation + JSON parsing
            raw = await llm.ainvoke(prompt)
            
            try:
                # 1. Try finding markdown JSON block
                json_match = re.search(r"```json\s*(.*?)\s*```", raw.content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # 2. Try parsing the raw content directly (if LLM forgot backticks)
                return json.loads(raw.content)
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from fallback generation for {artifact_type.value}.")
                return {"raw": raw.content}
                
    except Exception as e:
        logger.error(f"Structured generation failed for {artifact_type.value}: {e}")
        raise e


async def run_generation_task(
    artifact_id: uuid.UUID,
    notebook_id: uuid.UUID,
    user_id: uuid.UUID,
    artifact_type: ArtifactType,
    request_prompt: str | None,
    options: dict[str, Any],
    resolved_ids: list[str],
):
    """Background worker that handles LLM calls and DB updates in a fresh session."""

    # Open a fresh database session explicitly for this background task
    async with AsyncSessionLocal() as db:
        repo = ArtifactRepository(db)
        try:
            # 1. Build Context
            # Note: If build_context makes DB/Vector DB calls, ensure it doesn't need to be awaited.
            context_result = ArtifactContextBuilder.build_context(
                user_id=user_id,
                resolved_source_ids=resolved_ids,
                artifact_type=artifact_type.value,
                prompt=request_prompt,
            )

            # 2. Compress Evidence
            llm = get_llm()
            compressor = ArtifactEvidenceCompressor(llm)
            try:
                evidence_pack = await compressor.compress(
                    artifact_type=artifact_type.value,
                    context_text=context_result.context_text,
                    prompt=request_prompt,
                )
                evidence_pack_json = json.dumps(evidence_pack.model_dump(), indent=2)
            except Exception as e:
                logger.warning(f"Evidence compression failed: {e}")
                evidence_pack_json = context_result.context_text

            # 3. Build Prompt & Generate Content
            generation_prompt = ArtifactPromptBuilder.build_generation_prompt(
                artifact_type=artifact_type.value,
                evidence_pack_json=evidence_pack_json,
                prompt=request_prompt,
                options=options,
            )

            content = await generate_structured_content(
                llm=llm,
                artifact_type=artifact_type,
                prompt=generation_prompt,
            )
            
            title = content.get("title", f"{artifact_type.value.title()} Artifact")

            # 4. Save Success Status (READY)
            artifact = await repo.get_by_id(artifact_id, notebook_id, user_id)
            if artifact:
                artifact.title = title
                artifact.content_json = content
                artifact.status = ArtifactStatus.READY
                artifact.context_metadata = {
                    "mode_used": context_result.mode_used,
                    "total_chunks": context_result.total_chunks,
                    "total_estimated_tokens": context_result.total_estimated_tokens,
                }
                await repo.update(artifact)
                logger.info(f"Background task complete for artifact {artifact_id}")

        except Exception as e:
            # 5. Save Failure Status (ERROR)
            logger.error(f"Background generation failed for {artifact_id}: {e}", exc_info=True)
            artifact = await repo.get_by_id(artifact_id, notebook_id, user_id)
            if artifact:
                await repo.update_status(
                    artifact=artifact, 
                    status=ArtifactStatus.ERROR, 
                    error_message=str(e)
                )

            logger.info(f"Background task failed for artifact {artifact_id}")