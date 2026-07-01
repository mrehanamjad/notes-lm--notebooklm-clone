# repository.py
"""Artifact repository with CRUD operations."""

from app.features.artifacts.schema import ArtifactShortResponse
import uuid
from typing import Optional, List, cast
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine import CursorResult 

from app.features.artifacts.model import Artifact, ArtifactStatus
from app.features.artifacts.schema import ArtifactType


class ArtifactRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Create ──────────────────────────────────────────────────────────────────
    
    async def create(self, artifact: Artifact) -> Artifact:
        """Create a new artifact."""
        self.db.add(artifact)
        await self.db.commit()
        await self.db.refresh(artifact)
        return artifact

    # ── Read ────────────────────────────────────────────────────────────────────
    
    async def get_by_id(
        self, 
        artifact_id: uuid.UUID, 
        notebook_id: uuid.UUID, 
        user_id: uuid.UUID
    ) -> Optional[Artifact]:
        """Get artifact by ID with ownership validation."""
        result = await self.db.execute(
            select(Artifact).where(
                Artifact.id == artifact_id,
                Artifact.notebook_id == notebook_id,
                Artifact.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_notebook(
        self, 
        notebook_id: uuid.UUID, 
        user_id: uuid.UUID,
        artifact_type: Optional[ArtifactType] = None,
        status: Optional[ArtifactStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ArtifactShortResponse]:
        """List artifacts for a notebook with optional filters."""
        # Select only the fields you need
        query = select(
            Artifact.id,
            Artifact.notebook_id,
            Artifact.user_id,
            Artifact.artifact_type,
            Artifact.status,
            Artifact.title,
            Artifact.created_at,
            Artifact.updated_at
        ).where(
            Artifact.notebook_id == notebook_id,
            Artifact.user_id == user_id,
        )
        
        if artifact_type:
            query = query.where(Artifact.artifact_type == artifact_type)
        
        if status:
            query = query.where(Artifact.status == status)
        
        query = query.order_by(Artifact.created_at.desc())
        
        if limit is not None:
            query = query.limit(limit)
        if offset is not None:
            query = query.offset(offset)
        
        result = await self.db.execute(query)
        rows = result.all()
        
        # Convert to response model
        return [
            ArtifactShortResponse(
                id=row.id,
                notebook_id=row.notebook_id,
                user_id=row.user_id,
                artifact_type=row.artifact_type,
                status=row.status,
                title=row.title,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            for row in rows
        ]

    async def count_by_notebook(
        self, 
        notebook_id: uuid.UUID, 
        user_id: uuid.UUID,
        artifact_type: Optional[ArtifactType] = None,
        status: Optional[ArtifactStatus] = None,
    ) -> int:
        """Count artifacts for a notebook with optional filters."""
        query = select(func.count()).select_from(Artifact).where(
            Artifact.notebook_id == notebook_id,
            Artifact.user_id == user_id,
        )
        
        if artifact_type:
            query = query.where(Artifact.artifact_type == artifact_type)
        
        if status:
            query = query.where(Artifact.status == status)
        
        result = await self.db.execute(query)
        return result.scalar_one()

    async def get_by_source_id(
        self, 
        source_id: str, 
        user_id: uuid.UUID,
        limit: Optional[int] = None,
    ) -> List[Artifact]:
        """Find all artifacts that include a specific source."""
        query = select(Artifact).where(
            Artifact.user_id == user_id,
            Artifact.included_sources.contains([source_id])
        ).order_by(Artifact.created_at.desc())
        
        if limit is not None:
            query = query.limit(limit)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_by_source_ids(
        self, 
        source_ids: List[str], 
        user_id: uuid.UUID,
    ) -> List[Artifact]:
        """Find all artifacts that include any of the given sources."""
        query = select(Artifact).where(
            Artifact.user_id == user_id,
            Artifact.included_sources.overlap(source_ids)
        ).order_by(Artifact.created_at.desc())
        
        result = await self.db.execute(query)
        return list(result.scalars().all())

    # ── Update ──────────────────────────────────────────────────────────────────
    async def update(self, artifact: Artifact) -> Artifact:
        """Save changes to an existing artifact."""
        self.db.add(artifact)
        await self.db.commit()
        await self.db.refresh(artifact)
        return artifact

    async def update_status(
        self, 
        artifact: Artifact, 
        status: ArtifactStatus, 
        error_message: Optional[str] = None
    ) -> Artifact:
        """Helper to quickly update just the status and optional error."""
        artifact.status = status # type: ignore
        if error_message:
            artifact.error_message = error_message # type: ignore
        
        self.db.add(artifact)
        await self.db.commit()
        await self.db.refresh(artifact)
        return artifact

    async def update_content(
        self, 
        artifact: Artifact, 
        content: dict,
        status: ArtifactStatus = ArtifactStatus.READY,
    ) -> Artifact:
        """
        Update artifact with final generated content and mark READY.

        Also clears evidence_pack_json: once final content has been generated
        successfully, the cached evidence pack is no longer needed and is
        dropped so it doesn't take up storage or get reused stale.
        """
        artifact.content_json = content # type: ignore
        artifact.status = status # type: ignore
        artifact.evidence_pack_json = None # type: ignore
        self.db.add(artifact)
        await self.db.commit()
        await self.db.refresh(artifact)
        return artifact

    async def update_evidence_pack(
        self,
        artifact: Artifact,
        evidence_pack: dict,
    ) -> Artifact:
        """
        Persist the compressed evidence pack for an artifact without touching
        its status or final content.

        Called right after evidence compression succeeds, so that a later
        retry (e.g. after a generation-step failure) can skip vector search +
        LLM compression entirely and resume directly from content generation.
        """
        artifact.evidence_pack_json = evidence_pack # type: ignore
        self.db.add(artifact)
        await self.db.commit()
        await self.db.refresh(artifact)
        return artifact

    # ── Delete ──────────────────────────────────────────────────────────────────
    
    async def delete(self, artifact: Artifact) -> None:
        """Delete an artifact."""
        await self.db.delete(artifact)
        await self.db.commit()

    async def delete_by_notebook(
        self, 
        notebook_id: uuid.UUID, 
        user_id: uuid.UUID
    ) -> int:
        """Delete all artifacts for a notebook. Returns count deleted."""
        # Highly optimized bulk delete statement
        query = delete(Artifact).where(
            Artifact.notebook_id == notebook_id,
            Artifact.user_id == user_id,
        )
        result = await self.db.execute(query)
        await self.db.commit()
        cursor_results = cast(CursorResult, result)
        return cursor_results.rowcount

    # ── Bulk Operations ────────────────────────────────────────────────────────
    
    async def bulk_create(self, artifacts: List[Artifact]) -> List[Artifact]:
        """Bulk create artifacts."""
        for artifact in artifacts:
            self.db.add(artifact)
        await self.db.commit()
        
        for artifact in artifacts:
            await self.db.refresh(artifact)
        
        return artifacts

    async def get_artifacts_by_type(
        self,
        notebook_id: uuid.UUID,
        user_id: uuid.UUID,
        artifact_type: ArtifactType,
        limit: int = 10,
    ) -> List[Artifact]:
        """Get recent artifacts of a specific type for a notebook."""
        result = await self.db.execute(
            select(Artifact)
            .where(
                Artifact.notebook_id == notebook_id,
                Artifact.user_id == user_id,
                Artifact.artifact_type == artifact_type,
                Artifact.status == ArtifactStatus.READY,
            )
            .order_by(Artifact.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())