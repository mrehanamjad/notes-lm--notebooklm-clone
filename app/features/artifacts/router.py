import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.core.deps import get_db, get_current_user
from app.features.users.model import User
from app.features.artifacts.service import ArtifactService

# ── Unified Schema Imports ───────────────────────────────────────────────────
from app.features.artifacts.schema import (
    ArtifactType,
    ArtifactStatus,
    ArtifactResponse,
    ArtifactListResponse,
    QuizCreateRequest,
    FlashcardCreateRequest,
    FAQCreateRequest,
    StudyGuideCreateRequest,
    SummaryCreateRequest,
    MindMapCreateRequest,
    SlideDeckCreateRequest,
    AudioOverviewCreateRequest,
    ReportCreateRequest,
    DataTableCreateRequest,
)

router = APIRouter(tags=["Artifacts"])


@router.post(
    "/{notebook_id}/artifacts/quiz", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_quiz(
    notebook_id: uuid.UUID,
    data: QuizCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a quiz artifact from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_quiz(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="Quiz artifact generation has been initiated successfully",
        data=artifact,
    )


@router.post(
    "/{notebook_id}/artifacts/flashcards", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_flashcards(
    notebook_id: uuid.UUID,
    data: FlashcardCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a flashcards artifact from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_flashcards(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="Flashcards artifact generation has been initiated successfully",
        data=artifact,
    )


@router.post(
    "/{notebook_id}/artifacts/faqs", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_faqs(
    notebook_id: uuid.UUID,
    data: FAQCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a FAQ artifact from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_faqs(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="FAQ artifact generation has been initiated successfully",
        data=artifact,
    )


@router.post(
    "/{notebook_id}/artifacts/study-guide", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_study_guide(
    notebook_id: uuid.UUID,
    data: StudyGuideCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a study guide artifact from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_study_guide(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="Study guide generation has been initiated successfully",
        data=artifact,
    )


@router.post(
    "/{notebook_id}/artifacts/summary", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_summary(
    notebook_id: uuid.UUID,
    data: SummaryCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a summary artifact from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_summary(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="Summary generation has been initiated successfully",
        data=artifact,
    )


@router.post(
    "/{notebook_id}/artifacts/mindmap", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_mindmap(
    notebook_id: uuid.UUID,
    data: MindMapCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a mind map artifact from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_mindmap(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="Mind map generation has been initiated successfully",
        data=artifact,
    )


@router.post(
    "/{notebook_id}/artifacts/slide-deck", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_slide_deck(
    notebook_id: uuid.UUID,
    data: SlideDeckCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a slide deck artifact from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_slide_deck(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="Slide deck generation has been initiated successfully",
        data=artifact,
    )


@router.post(
    "/{notebook_id}/artifacts/audio-overview", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_voice_overview(
    notebook_id: uuid.UUID,
    data: AudioOverviewCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a two-host voice overview (podcast-style audio) from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_voice_overview(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="Voice overview generation has been initiated successfully",
        data=artifact,
    )


@router.post(
    "/{notebook_id}/artifacts/report", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_report(
    notebook_id: uuid.UUID,
    data: ReportCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a structured report artifact from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_report(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="Report artifact generation has been initiated successfully",
        data=artifact,
    )


@router.post(
    "/{notebook_id}/artifacts/datatable", 
    response_model=APIResponse[ArtifactResponse], 
    status_code=status.HTTP_202_ACCEPTED
)
async def create_datatable(
    notebook_id: uuid.UUID,
    data: DataTableCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a structured data table artifact from notebook sources."""
    service = ArtifactService(db)
    artifact = await service.create_datatable(notebook_id, current_user.id, data, background_tasks)
    return APIResponse(
        message="Data table artifact generation has been initiated successfully",
        data=artifact,
    )

@router.post(
    "/{notebook_id}/artifacts/{artifact_id}/retry",
    response_model=APIResponse[ArtifactResponse],
    status_code=status.HTTP_202_ACCEPTED
)

async def retry_artifact(
    notebook_id: uuid.UUID,
    artifact_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retry a failed artifact generation task."""
    service = ArtifactService(db)
    artifact = await service.retry_artifact_generation(
        notebook_id, artifact_id, uuid.UUID(str(current_user.id)), background_tasks
    )
    return APIResponse(
        message="Artifact retry has been initiated successfully",
        data=artifact,
    )


# ── Generic Artifact Routes ────────────────────────────────────────────────────

@router.get(
    "/{notebook_id}/artifacts", 
    response_model=APIResponse[ArtifactListResponse]
)
async def list_artifacts(
    notebook_id: uuid.UUID,
    artifact_type: Optional[ArtifactType] = Query(None, description="Filter by artifact type"),
    status_filter: Optional[ArtifactStatus] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Number of artifacts to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all artifacts for a notebook with optional filters."""
    service = ArtifactService(db)
    data = await service.list_artifacts(
        notebook_id, 
        current_user.id,
        artifact_type=artifact_type,
        status_filter=status_filter,
        limit=limit,
        offset=offset,
    )
    return APIResponse(
        message="Artifacts fetched successfully",
        data=data,
    )


@router.get(
    "/{notebook_id}/artifacts/{artifact_id}", 
    response_model=APIResponse[ArtifactResponse]
)
async def get_artifact(
    notebook_id: uuid.UUID,
    artifact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific artifact by ID."""
    service = ArtifactService(db)
    artifact = await service.get_artifact(notebook_id, artifact_id, uuid.UUID(str(current_user.id)))
    return APIResponse(
        message="Artifact fetched successfully",
        data=artifact,
    )


@router.delete(
    "/{notebook_id}/artifacts/{artifact_id}", 
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_artifact(
    notebook_id: uuid.UUID,
    artifact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete an artifact."""
    service = ArtifactService(db)
    await service.delete_artifact(notebook_id, artifact_id, uuid.UUID(str(current_user.id)))
    return


@router.delete(
    "/{notebook_id}/artifacts", 
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_all_artifacts(
    notebook_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete all artifacts for a notebook."""
    service = ArtifactService(db)
    await service.delete_all_artifacts(notebook_id, uuid.UUID(str(current_user.id)))
    return