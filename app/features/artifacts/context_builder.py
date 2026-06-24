"""Artifact-specific context assembly with hybrid retrieval strategy.

Two modes:
  - Semantic mode: when user provides topic/prompt — build retrieval queries,
    search Qdrant with high k, deduplicate, and cap context.
  - Coverage mode: when no topic/prompt and source set is small — sample chunks
    evenly across sources for broad material coverage.
"""

import hashlib
import uuid
from typing import List, Tuple
from dataclasses import dataclass, field

from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from app.core.config import settings
from app.core.ai_clients import get_qdrant_client, get_embeddings
from app.core.logger import logger


# ── Configuration ────────────────────────────────────────────────────────────

SEMANTIC_K = 20                    # chunks to retrieve in semantic mode
COVERAGE_K_PER_SOURCE = 4         # chunks per source in coverage mode
COVERAGE_SOURCE_THRESHOLD = 5     # max sources for coverage mode
MAX_CONTEXT_CHARS = 15_000        # hard cap on final context size


@dataclass
class SourceRef:
    """Metadata about a retrieved chunk for optional citation tracking."""
    source_id: str
    file_name: str
    page_number: int
    chunk_index: int
    similarity_score: float


@dataclass
class ContextResult:
    """Output of the context builder."""
    context_text: str
    source_refs: List[SourceRef] = field(default_factory=list)
    mode_used: str = "semantic"
    total_chunks: int = 0


class ArtifactContextBuilder:
    """Assembles grounded context for artifact generation."""

    @staticmethod
    def build_context(
        user_id: uuid.UUID,
        resolved_source_ids: List[str],
        topic: str | None = None,
        prompt: str | None = None,
        artifact_type: str = "artifact",
    ) -> ContextResult:
        """Entry point — picks the right retrieval mode and assembles context."""
        if not resolved_source_ids:
            return ContextResult(context_text="No documents available.", mode_used="none")

        has_focus = bool(topic or prompt)
        is_small_corpus = len(resolved_source_ids) <= COVERAGE_SOURCE_THRESHOLD

        if has_focus:
            return ArtifactContextBuilder._semantic_retrieval(
                user_id, resolved_source_ids, topic, prompt, artifact_type
            )
        elif is_small_corpus:
            return ArtifactContextBuilder._coverage_retrieval(
                user_id, resolved_source_ids, artifact_type
            )
        else:
            # Large corpus with no focus — use semantic with a generic query
            return ArtifactContextBuilder._semantic_retrieval(
                user_id, resolved_source_ids, topic=None, prompt=None,
                artifact_type=artifact_type
            )

    # ── Semantic Mode ─────────────────────────────────────────────────────────

    @staticmethod
    def _semantic_retrieval(
        user_id: uuid.UUID,
        source_ids: List[str],
        topic: str | None,
        prompt: str | None,
        artifact_type: str,
    ) -> ContextResult:
        """Retrieve chunks via semantic similarity against topic/prompt."""
        query = ArtifactContextBuilder._build_retrieval_query(topic, prompt, artifact_type)

        try:
            embeddings = get_embeddings()
            client = get_qdrant_client()
            collection = settings.QDRANT_COLLECTION

            query_vector = embeddings.embed_query(query)
            search_filter = Filter(must=[
                FieldCondition(key='user_id', match=MatchValue(value=str(user_id))),
                FieldCondition(key='source_id', match=MatchAny(any=source_ids)),
            ])

            results = client.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=search_filter,
                limit=SEMANTIC_K,
                with_payload=True,
                with_vectors=False,
            )

            chunks, refs = ArtifactContextBuilder._process_hits(results.points)
            context_text = ArtifactContextBuilder._assemble_context(chunks)

            return ContextResult(
                context_text=context_text,
                source_refs=refs,
                mode_used="semantic",
                total_chunks=len(chunks),
            )
        except Exception as e:
            logger.error(f"Artifact semantic retrieval error: {e}", exc_info=True)
            return ContextResult(context_text="No documents available.", mode_used="semantic_error")

    # ── Coverage Mode ─────────────────────────────────────────────────────────

    @staticmethod
    def _coverage_retrieval(
        user_id: uuid.UUID,
        source_ids: List[str],
        artifact_type: str,
    ) -> ContextResult:
        """Retrieve chunks spread evenly across sources for broad coverage."""
        all_chunks: List[Tuple[str, SourceRef]] = []

        try:
            embeddings = get_embeddings()
            client = get_qdrant_client()
            collection = settings.QDRANT_COLLECTION

            # Build a generic query for coverage relevance ranking
            coverage_query = f"Key concepts and important information for {artifact_type} generation"
            query_vector = embeddings.embed_query(coverage_query)

            for sid in source_ids:
                search_filter = Filter(must=[
                    FieldCondition(key='user_id', match=MatchValue(value=str(user_id))),
                    FieldCondition(key='source_id', match=MatchValue(value=sid)),
                ])

                results = client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=COVERAGE_K_PER_SOURCE,
                    with_payload=True,
                    with_vectors=False,
                )

                chunks, refs = ArtifactContextBuilder._process_hits(results.points)
                all_chunks.extend(zip(chunks, refs))

            # Deduplicate
            seen_hashes: set[str] = set()
            deduped_chunks: List[str] = []
            deduped_refs: List[SourceRef] = []

            for chunk_text, ref in all_chunks:
                h = hashlib.md5(chunk_text.encode()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    deduped_chunks.append(chunk_text)
                    deduped_refs.append(ref)

            context_text = ArtifactContextBuilder._assemble_context(deduped_chunks)

            return ContextResult(
                context_text=context_text,
                source_refs=deduped_refs,
                mode_used="coverage",
                total_chunks=len(deduped_chunks),
            )
        except Exception as e:
            logger.error(f"Artifact coverage retrieval error: {e}", exc_info=True)
            return ContextResult(context_text="No documents available.", mode_used="coverage_error")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_retrieval_query(topic: str | None, prompt: str | None, artifact_type: str) -> str:
        """Construct a retrieval query from available user inputs."""
        parts: List[str] = []
        if topic:
            parts.append(topic)
        if prompt:
            parts.append(prompt)
        if not parts:
            parts.append(f"Key concepts and important information for {artifact_type} generation")
        return " ".join(parts)

    @staticmethod
    def _process_hits(points: list) -> Tuple[List[str], List[SourceRef]]:
        """Extract chunk texts and source refs from Qdrant hits."""
        chunks: List[str] = []
        refs: List[SourceRef] = []

        for hit in points:
            p = hit.payload
            chunk_text = p.get('chunk_text', '')
            if not chunk_text:
                continue

            chunks.append(f"[{p['file_name']} | page {p['page_number']}]\n{chunk_text}")
            refs.append(SourceRef(
                source_id=p['source_id'],
                file_name=p['file_name'],
                page_number=p['page_number'],
                chunk_index=p['chunk_index'],
                similarity_score=round(hit.score, 4),
            ))

        return chunks, refs

    @staticmethod
    def _assemble_context(chunks: List[str]) -> str:
        """Join chunks and enforce max context size."""
        if not chunks:
            return "No documents available."

        context = "\n\n---\n\n".join(chunks)

        # Hard cap on context size
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]
            # Trim to last complete chunk separator to avoid mid-sentence cuts
            last_sep = context.rfind("\n\n---\n\n")
            if last_sep > 0:
                context = context[:last_sep]

        return context
