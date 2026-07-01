import uuid
import json
import os
import re
from typing import Any
import asyncio

from app.core.config import settings

from app.core.logger import logger
from app.database.session import AsyncSessionLocal
from app.core.ai_clients import get_llm
from app.core.providers.storage import get_storage_provider

from app.features.artifacts.repository import ArtifactRepository
from app.features.artifacts.artifact_context_builder import ArtifactContextBuilder
from app.features.artifacts.artifact_evidence_compressor import ArtifactEvidenceCompressor
from app.features.artifacts.artifact_prompt_builder import ArtifactPromptBuilder
from app.features.artifacts.artifact_audio_generator import (
    ArtifactAudioGenerator,
    ArtifactAudioGenerationError,
    get_audio_generator,
)

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
    AudioOverviewArtifact,
    AudioOverviewStoredContent,
    AudioOverviewMetadata,
    ReportArtifact,
    DataTableArtifact,
)
from app.features.artifacts.utils import IconifyService, PexelsService


# async def generate_structured_content(
#     llm: Any,
#     artifact_type: ArtifactType,
#     prompt: str,
# ) -> dict[str, Any]:
#     """Generate structured content using LLM with schema."""
#     schema_map = {
#         ArtifactType.QUIZ: QuizArtifact,
#         ArtifactType.FLASHCARDS: FlashcardsArtifact,
#         ArtifactType.FAQ: FAQArtifact,
#         ArtifactType.STUDY_GUIDE: StudyGuideArtifact,
#         ArtifactType.SUMMARY: SummaryArtifact,
#         ArtifactType.MINDMAP: MindMapArtifact,
#         ArtifactType.SLIDE_DECK: SlideDeckArtifact,
#     }

#     schema = schema_map.get(artifact_type, SummaryArtifact)

#     try:
#         # Using with_structured_output if available
#         if hasattr(llm, "with_structured_output"):
#             structured_llm = llm.with_structured_output(schema)
#             result = await structured_llm.ainvoke(prompt)
#             return result.model_dump() if hasattr(result, "model_dump") else result
#         else:
#             # Fallback: text generation + JSON parsing
#             raw = await llm.ainvoke(prompt)
            
#             try:
#                 # 1. Try finding markdown JSON block
#                 json_match = re.search(r"```json\s*(.*?)\s*```", raw.content, re.DOTALL)
#                 if json_match:
#                     return json.loads(json_match.group(1))
                
#                 # 2. Try parsing the raw content directly (if LLM forgot backticks)
#                 return json.loads(raw.content)
                
#             except json.JSONDecodeError:
#                 logger.error(f"Failed to parse JSON from fallback generation for {artifact_type.value}.")
#                 return {"raw": raw.content}
                
#     except Exception as e:
#         logger.error(f"Structured generation failed for {artifact_type.value}: {e}")
#         raise e

async def _repair_and_parse_json(text: str) -> dict[str, Any]:
    """Attempts to repair incomplete JSON strings by adding missing braces."""
    text = text.strip()
    
    # Simple count of opening vs closing brackets
    open_brackets = text.count('{') + text.count('[')
    close_brackets = text.count('}') + text.count(']')
    
    if open_brackets > close_brackets:
        text += '}' * (open_brackets - close_brackets)
        
    return json.loads(text)

async def generate_structured_content(
    llm: Any,
    artifact_type: ArtifactType,
    prompt: str,
) -> dict[str, Any]:
    """Generate structured content with robust repair for truncated JSON."""
    schema_map = {
        ArtifactType.QUIZ: QuizArtifact,
        ArtifactType.FLASHCARDS: FlashcardsArtifact,
        ArtifactType.FAQ: FAQArtifact,
        ArtifactType.STUDY_GUIDE: StudyGuideArtifact,
        ArtifactType.SUMMARY: SummaryArtifact,
        ArtifactType.MINDMAP: MindMapArtifact,
        ArtifactType.SLIDE_DECK: SlideDeckArtifact,
        ArtifactType.VOICE_OVERVIEW: AudioOverviewArtifact,
        ArtifactType.REPORT: ReportArtifact,
        ArtifactType.DATATABLE: DataTableArtifact,
    }

    schema = schema_map.get(artifact_type, SummaryArtifact)

    # 1. Primary Attempt: Structured Output
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(schema)
            result = await structured_llm.ainvoke(prompt)
            return result.model_dump() if hasattr(result, "model_dump") else result
        except Exception as e:
            logger.warning(f"Structured output failed, falling back to raw: {e}")

    # 2. Fallback: Text Generation + Repair
    raw = await llm.ainvoke(prompt)
    content = raw.content if hasattr(raw, "content") else str(raw)
    
    # Extract the JSON block
    json_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
    if json_match:
        try:
            return await _repair_and_parse_json(json_match.group(1))
        except json.JSONDecodeError:
            logger.error("JSON repair failed.")
    
    return {"raw": content}


async def resolve_slide_visuals(
    deck: dict,
    pexels_service: PexelsService,
    iconify_service: IconifyService,
):
    """
    Enrich every slide with a resolved image URL or icon name.
    """

    slides = deck.get("slides", [])

    async def resolve_slide(slide: dict):
        visual = slide.get("visual")

        if not visual:
            return

        visual_type = visual.get("type")
        query = visual.get("query") or visual.get("concept")

        if visual_type == "image" and query:
            url = await asyncio.to_thread(
                pexels_service.search_image,
                query,
            )
            visual["resolved"] = url

        elif visual_type == "icon" and query:
            icon = await asyncio.to_thread(
                iconify_service.search_icon,
                query,
            )
            visual["resolved"] = icon

    await asyncio.gather(
        *(resolve_slide(slide) for slide in slides)
    )

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

    # Voice overviews have a materially different pipeline (script -> TTS -> upload),
    # so they're routed to a dedicated worker rather than overloading this one.
    if artifact_type == ArtifactType.VOICE_OVERVIEW:
        await run_voice_overview_generation_task(
            artifact_id=artifact_id,
            notebook_id=notebook_id,
            user_id=user_id,
            request_prompt=request_prompt,
            options=options,
            resolved_ids=resolved_ids,
        )
        return

    # Open a fresh database session explicitly for this background task
    async with AsyncSessionLocal() as db:
        repo = ArtifactRepository(db)
        try:
            # 0. Fetch the artifact up front — needed either way, and lets us
            # check for a previously-cached evidence pack (e.g. from a prior
            # failed attempt) before doing any retrieval/compression work.
            artifact = await repo.get_by_id(artifact_id, notebook_id, user_id)
            if not artifact:
                logger.error(f"Artifact {artifact_id} not found when starting generation task")
                return

            cached_pack = artifact.evidence_pack_json
            context_metadata: dict[str, Any] = dict(artifact.context_metadata or {})

            if cached_pack:
                # Resuming after a prior failure that happened *after* compression
                # succeeded. Skip vector search + LLM compression entirely.
                logger.info(
                    f"Resuming artifact {artifact_id} from cached evidence pack "
                    f"(skipping context build + compression)"
                )
                evidence_pack_json = json.dumps(cached_pack, indent=2)
                llm = get_llm()
            else:
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
                    evidence_pack_dict = evidence_pack.model_dump()
                    evidence_pack_json = json.dumps(evidence_pack_dict, indent=2)
                except Exception as e:
                    logger.warning(f"Evidence compression failed: {e}")
                    evidence_pack_dict = None
                    evidence_pack_json = context_result.context_text

                context_metadata = {
                    "mode_used": context_result.mode_used,
                    "total_chunks": context_result.total_chunks,
                    "total_estimated_tokens": context_result.total_estimated_tokens,
                }

                # Persist the evidence pack (and context metadata) right away, before
                # attempting generation. If generation fails below, a retry can jump
                # straight back in here without re-running retrieval or compression.
                if evidence_pack_dict:
                    artifact.evidence_pack_json = evidence_pack_dict
                artifact.context_metadata = context_metadata
                artifact = await repo.update(artifact)

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

            pexels = PexelsService(settings.PEXELS_API_KEY)
            iconify = IconifyService(preferred_prefix="lucide")

            await resolve_slide_visuals(
                content,
                pexels,
                iconify,
            )
            
            title = content.get("title", f"{artifact_type.value.title()} Artifact")

            # 4. Save Success Status (READY).
            # update_content also clears evidence_pack_json now that it's no longer needed.
            artifact = await repo.get_by_id(artifact_id, notebook_id, user_id)
            if artifact:
                artifact.title = title
                artifact.context_metadata = context_metadata
                await repo.update_content(artifact, content=content, status=ArtifactStatus.READY)
                logger.info(f"Background task complete for artifact {artifact_id}")

        except Exception as e:
            # 5. Save Failure Status (ERROR).
            # Note: this does NOT touch evidence_pack_json, so if compression had
            # already succeeded and been persisted above, it survives for a retry.
            logger.error(f"Background generation failed for {artifact_id}: {e}", exc_info=True)
            artifact = await repo.get_by_id(artifact_id, notebook_id, user_id)
            if artifact:
                await repo.update_status(
                    artifact=artifact, 
                    status=ArtifactStatus.ERROR, 
                    error_message=str(e)
                )

            logger.info(f"Background task failed for artifact {artifact_id}")


# =============================================================================
# Voice Overview (two-host podcast) generation pipeline
# =============================================================================

async def run_voice_overview_generation_task(
    artifact_id: uuid.UUID,
    notebook_id: uuid.UUID,
    user_id: uuid.UUID,
    request_prompt: str | None,
    options: dict[str, Any],
    resolved_ids: list[str],
):
    """
    Background worker for Voice Overview artifacts.

    Pipeline:
    1. Build context from resolved sources (same retrieval system as other artifacts).
    2. Compress evidence (same compressor, voice_overview-tuned retrieval config).
    3. Generate a two-host dialogue script with the LLM (structured output).
    4. Synthesize the script into a single stitched MP3 with edge-tts + pydub.
    5. Upload the MP3 to the configured storage provider (ImageKit by default).
    6. Persist the script (content_json), audio URL, and duration; mark READY.
    7. On any failure, mark ERROR with a useful message; always clean up local temp files.
    """
    local_audio_path: str | None = None

    async with AsyncSessionLocal() as db:
        repo = ArtifactRepository(db)
        try:
            # 0. Fetch the artifact up front so we can check for a cached
            # evidence pack from a previously-failed attempt.
            artifact = await repo.get_by_id(artifact_id, notebook_id, user_id)
            if not artifact:
                logger.error(f"Artifact {artifact_id} not found when starting voice overview task")
                return

            cached_pack = artifact.evidence_pack_json
            context_metadata: dict[str, Any] = dict(artifact.context_metadata or {})

            if cached_pack:
                logger.info(
                    f"Resuming voice overview artifact {artifact_id} from cached evidence pack "
                    f"(skipping context build + compression)"
                )
                evidence_pack_json = json.dumps(cached_pack, indent=2)
                llm = get_llm()
            else:
                # 1. Build Context (reuses the same artifact-aware retrieval pipeline)
                context_result = ArtifactContextBuilder.build_context(
                    user_id=user_id,
                    resolved_source_ids=resolved_ids,
                    artifact_type=ArtifactType.VOICE_OVERVIEW.value,
                    prompt=request_prompt,
                )

                # 2. Compress Evidence
                llm = get_llm()
                compressor = ArtifactEvidenceCompressor(llm)
                try:
                    evidence_pack = await compressor.compress(
                        artifact_type=ArtifactType.VOICE_OVERVIEW.value,
                        context_text=context_result.context_text,
                        prompt=request_prompt,
                    )
                    evidence_pack_dict = evidence_pack.model_dump()
                    evidence_pack_json = json.dumps(evidence_pack_dict, indent=2)
                except Exception as e:
                    logger.warning(f"Evidence compression failed for voice overview: {e}")
                    evidence_pack_dict = None
                    evidence_pack_json = context_result.context_text

                context_metadata = {
                    "mode_used": context_result.mode_used,
                    "total_chunks": context_result.total_chunks,
                    "total_estimated_tokens": context_result.total_estimated_tokens,
                }

                # Persist the evidence pack before attempting script generation/TTS,
                # so a failure further down the pipeline (script gen, TTS, upload)
                # can be retried without re-running retrieval or compression.
                if evidence_pack_dict:
                    artifact.evidence_pack_json = evidence_pack_dict
                artifact.context_metadata = context_metadata
                artifact = await repo.update(artifact)

            # 3. Build Prompt & Generate the Dialogue Script
            generation_prompt = ArtifactPromptBuilder.build_generation_prompt(
                artifact_type=ArtifactType.VOICE_OVERVIEW.value,
                evidence_pack_json=evidence_pack_json,
                prompt=request_prompt,
                options=options,
            )

            script_content = await generate_structured_content(
                llm=llm,
                artifact_type=ArtifactType.VOICE_OVERVIEW,
                prompt=generation_prompt,
            )

            script_content = _normalize_dialogue_content(script_content)
            script = AudioOverviewArtifact.model_validate(script_content)

            if not script.dialogue:
                raise ValueError("Generated voice overview script had no dialogue lines.")

            title = script.title or "Voice Overview"

            # 4. Synthesize audio (edge-tts per line, stitched with pydub)
            voice_style = options.get("voice_style", "default")
            generator: ArtifactAudioGenerator = get_audio_generator(voice_style=voice_style)

            try:
                audio_result = await generator.generate(artifact=script)
                local_audio_path = audio_result.file_path
            except ArtifactAudioGenerationError as e:
                raise RuntimeError(f"Audio synthesis failed: {e}") from e

            # 5. Upload to storage (ImageKit by default, per app config)
            storage = get_storage_provider()
            file_name = f"{artifact_id}.mp3"
            upload_result = storage.upload_file(
                file_path=local_audio_path,
                file_name=file_name,
                folder=f"sources/{user_id}/voice_overviews",
            )

            # 6. Persist success — audio metadata is nested inside content_json
            # (no dedicated audio_url/audio_file_id/audio_duration_seconds columns)
            stored_content = AudioOverviewStoredContent(
                title=script.title,
                description=script.description,
                dialogue=script.dialogue,
                audio=AudioOverviewMetadata(
                    audio_url=upload_result.get("url"),
                    audio_file_id=upload_result.get("file_id"),
                    audio_duration_seconds=audio_result.duration_seconds,
                ),
            )

            artifact = await repo.get_by_id(artifact_id, notebook_id, user_id)
            if artifact:
                artifact.title = title
                context_metadata = {
                    **context_metadata,
                    "dialogue_line_count": audio_result.line_count,
                }
                artifact.context_metadata = context_metadata
                # update_content also clears evidence_pack_json now that it's no longer needed.
                await repo.update_content(
                    artifact,
                    content=stored_content.model_dump(),
                    status=ArtifactStatus.READY,
                )
                logger.info(
                    f"Voice overview generation complete for artifact {artifact_id} "
                    f"({audio_result.duration_seconds}s, {audio_result.line_count} lines)"
                )

        except Exception as e:
            logger.error(f"Voice overview generation failed for {artifact_id}: {e}", exc_info=True)
            artifact = await repo.get_by_id(artifact_id, notebook_id, user_id)
            if artifact:
                await repo.update_status(
                    artifact=artifact,
                    status=ArtifactStatus.ERROR,
                    error_message=str(e),
                )
            logger.info(f"Background task failed for voice overview artifact {artifact_id}")

        finally:
            # Always clean up the local temp audio file once it's uploaded (or on failure)
            if local_audio_path and os.path.exists(local_audio_path):
                try:
                    os.remove(local_audio_path)
                except OSError as e:
                    logger.warning(f"Failed to remove local temp audio file {local_audio_path}: {e}")


def _normalize_dialogue_content(content: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize LLM output into the shape AudioOverviewArtifact expects.

    Handles minor LLM inconsistencies gracefully:
    - speaker values like "Host 1", "HOST_1", "speaker_1" -> "host_1" / "host_2"
    - missing title -> falls back to a generic title
    """
    if not isinstance(content, dict):
        return {"title": "Voice Overview", "dialogue": []}

    dialogue_raw = content.get("dialogue") or []
    normalized_lines = []

    for line in dialogue_raw:
        if not isinstance(line, dict):
            continue
        speaker_raw = str(line.get("speaker", "")).strip().lower()
        if "1" in speaker_raw:
            speaker = "host_1"
        elif "2" in speaker_raw:
            speaker = "host_2"
        else:
            # Alternate as a last resort so we never drop a line outright
            speaker = "host_1" if len(normalized_lines) % 2 == 0 else "host_2"

        text = str(line.get("text", "")).strip()
        if text:
            normalized_lines.append({"speaker": speaker, "text": text})

    return {
        "title": content.get("title") or "Voice Overview",
        "description": content.get("description"),
        "dialogue": normalized_lines,
    }