# artifact_context_builder.py
"""
Artifact Context Builder
============================

Optimized for:
- NotebookLM-style artifact generation
- Groq free-tier token constraints
- llama-3.3-70b-versatile
- Qdrant + all-MiniLM-L6-v2
- Current payload shape:
    {
        "source_id": "...",
        "source_type": "youtube|pdf|web|audio|note|doc",
        "title": "...",
        "chunk_index": 145,
        "chunk_text": "...",
        "page_number": 1,
        "is_table": false,
        "user_id": "...",
        "notebook_id": "...",
        "file_name": "..."
    }

Goals:
- Keep retrieval context compact and useful
- Avoid blowing Groq free-tier token limits
- Support artifact generation:
    * quiz
    * flashcards
    * faq
    * study_guide
    * summary
    * mindmap
    * slide_deck
- Work even without rich metadata like section_title / contains_definition

Core strategy:
1. Focused semantic retrieval if prompt exists
2. Broad coverage retrieval across sources
3. Merge with artifact-aware heuristics
4. Enforce very tight token budget
5. Assemble compact structured context

This builder is intentionally conservative with context size.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

from app.core.ai_clients import get_embeddings, get_qdrant_client
from app.core.config import settings
from app.core.logger import logger

# ── Unified Schema Imports ───────────────────────────────────────────────────
from app.features.artifacts.schema import ArtifactType, ArtifactSourceRef

@dataclass
class ContextResult:
    """Output of the context builder."""
    context_text: str
    source_refs: List[ArtifactSourceRef] = field(default_factory=list)
    mode_used: str = "compact_hybrid"
    total_chunks: int = 0
    total_estimated_tokens: int = 0


# =============================================================================
# Internal chunk model
# =============================================================================

@dataclass
class RetrievedChunk:
    """Internal representation of a retrieved chunk with scoring."""
    source_id: str
    source_type: str
    title: str
    file_name: str
    page_number: int | str | None
    chunk_index: int
    chunk_text: str
    similarity_score: float
    is_table: bool = False

    retrieval_bucket: str = "semantic"  # semantic | coverage
    artifact_score: float = 0.0
    estimated_tokens: int = 0

    @property
    def unique_id(self) -> str:
        """Unique identifier for deduplication."""
        return f"{self.source_id}:{self.chunk_index}"

    def to_source_ref(self) -> ArtifactSourceRef:
        """Convert to public SourceRef."""
        return ArtifactSourceRef(
            source_id=self.source_id,
            file_name=self.file_name,
            page_number=self.page_number,
            chunk_index=self.chunk_index,
            similarity_score=round(self.similarity_score, 4),
        )


@dataclass(frozen=True)
class ArtifactRetrievalConfig:
    """Configuration for artifact-specific retrieval."""
    artifact_type: str

    # retrieval sizes
    semantic_k: int
    coverage_k_per_source: int

    # final selected chunks
    final_chunk_target: int

    # ratio of semantic vs coverage when focus exists
    focused_semantic_ratio: float

    # ratio of semantic vs coverage when no focus
    broad_semantic_ratio: float

    # source caps
    max_chunks_per_source: int
    max_table_chunks_per_source: int

    # token budget for assembled context
    max_context_tokens: int
    max_context_chars: int

    # heuristics
    prefer_tables: bool = False
    prefer_fact_density: bool = False
    prefer_explanatory_chunks: bool = False
    prefer_summary_like_chunks: bool = False
    prefer_qa_like_chunks: bool = False


ARTIFACT_CONFIGS: Dict[str, ArtifactRetrievalConfig] = {
    # Quiz wants fact-heavy, definitional, formula/example-friendly content
    "quiz": ArtifactRetrievalConfig(
        artifact_type="quiz",
        semantic_k=12,
        coverage_k_per_source=2,
        final_chunk_target=8,
        focused_semantic_ratio=0.70,
        broad_semantic_ratio=0.45,
        max_chunks_per_source=2,
        max_table_chunks_per_source=1,
        max_context_tokens=2400,
        max_context_chars=14000,
        prefer_tables=True,
        prefer_fact_density=True,
    ),
    "flashcards": ArtifactRetrievalConfig(
        artifact_type="flashcards",
        semantic_k=12,
        coverage_k_per_source=2,
        final_chunk_target=8,
        focused_semantic_ratio=0.70,
        broad_semantic_ratio=0.45,
        max_chunks_per_source=2,
        max_table_chunks_per_source=1,
        max_context_tokens=2200,
        max_context_chars=13000,
        prefer_fact_density=True,
    ),
    "faq": ArtifactRetrievalConfig(
        artifact_type="faq",
        semantic_k=12,
        coverage_k_per_source=2,
        final_chunk_target=8,
        focused_semantic_ratio=0.75,
        broad_semantic_ratio=0.50,
        max_chunks_per_source=2,
        max_table_chunks_per_source=1,
        max_context_tokens=2500,
        max_context_chars=14000,
        prefer_explanatory_chunks=True,
        prefer_qa_like_chunks=True,
    ),
    "study_guide": ArtifactRetrievalConfig(
        artifact_type="study_guide",
        semantic_k=14,
        coverage_k_per_source=2,
        final_chunk_target=10,
        focused_semantic_ratio=0.65,
        broad_semantic_ratio=0.45,
        max_chunks_per_source=2,
        max_table_chunks_per_source=1,
        max_context_tokens=3000,
        max_context_chars=17000,
        prefer_tables=True,
        prefer_fact_density=True,
        prefer_explanatory_chunks=True,
        prefer_summary_like_chunks=True,
    ),
    "summary": ArtifactRetrievalConfig(
        artifact_type="summary",
        semantic_k=12,
        coverage_k_per_source=2,
        final_chunk_target=8,
        focused_semantic_ratio=0.65,
        broad_semantic_ratio=0.40,
        max_chunks_per_source=2,
        max_table_chunks_per_source=1,
        max_context_tokens=2400,
        max_context_chars=14000,
        prefer_explanatory_chunks=True,
        prefer_summary_like_chunks=True,
    ),
    "mindmap": ArtifactRetrievalConfig(
        artifact_type="mindmap",
        semantic_k=10,
        coverage_k_per_source=2,
        final_chunk_target=8,
        focused_semantic_ratio=0.60,
        broad_semantic_ratio=0.35,
        max_chunks_per_source=2,
        max_table_chunks_per_source=1,
        max_context_tokens=2200,
        max_context_chars=12000,
        prefer_summary_like_chunks=True,
        prefer_fact_density=True,
    ),
    "slide_deck": ArtifactRetrievalConfig(
        artifact_type="slide_deck",
        semantic_k=14,
        coverage_k_per_source=2,
        final_chunk_target=10,
        focused_semantic_ratio=0.65,
        broad_semantic_ratio=0.45,
        max_chunks_per_source=2,
        max_table_chunks_per_source=1,
        max_context_tokens=3000,
        max_context_chars=17000,
        prefer_tables=True,
        prefer_explanatory_chunks=True,
        prefer_summary_like_chunks=True,
    ),
    "artifact": ArtifactRetrievalConfig(
        artifact_type="artifact",
        semantic_k=12,
        coverage_k_per_source=2,
        final_chunk_target=8,
        focused_semantic_ratio=0.65,
        broad_semantic_ratio=0.40,
        max_chunks_per_source=2,
        max_table_chunks_per_source=1,
        max_context_tokens=2200,
        max_context_chars=13000,
    ),
}

ARTIFACT_HINTS: Dict[str, str] = {
    "quiz": (
        "Retrieve concepts, definitions, important facts, examples, comparisons, "
        "dates, formulas, and table information useful for creating quiz questions."
    ),
    "flashcards": (
        "Retrieve terms, definitions, concise explanations, named concepts, "
        "important facts, and memorization-worthy content useful for flashcards."
    ),
    "faq": (
        "Retrieve explanatory passages, how/why/what content, clarifications, "
        "steps, constraints, and question-answer style content useful for FAQ generation."
    ),
    "study_guide": (
        "Retrieve core concepts, explanations, important facts, examples, tables, "
        "and key takeaways useful for a study guide."
    ),
    "summary": (
        "Retrieve main ideas, important explanations, conclusions, findings, and "
        "key supporting facts useful for a summary."
    ),
    "mindmap": (
        "Retrieve main topics, subtopics, grouped concepts, relationships, "
        "and summary-like content useful for a mind map."
    ),
    "slide_deck": (
        "Retrieve main ideas, supporting evidence, key explanations, examples, "
        "comparisons, conclusions, and tables useful for presentation slides."
    ),
    "artifact": (
        "Retrieve the most important source content useful for artifact generation."
    ),
}

class ArtifactContextBuilder:
    """
    Compact artifact context builder optimized for Groq free-tier usage.
    """
    
    # ── Constants ────────────────────────────────────────────────────────────────
    
    MAX_CONTEXT_CHARS = 15000
    MAX_CONTEXT_TOKENS = 2600

    # ── Public entrypoint ──────────────────────────────────────────────────────

    @staticmethod
    def build_context(
        user_id: uuid.UUID,
        resolved_source_ids: List[str],
        artifact_type: ArtifactType = "artifact",
        prompt: str | None = None,
        max_context_tokens: int | None = None,
        max_context_chars: int | None = None,
    ) -> ContextResult:
        """
        Build compact artifact context.

        Notes:
        - This version is intentionally strict with token budget.
        - It assumes the downstream generator prompt will also consume tokens,
          so we do not try to fill the whole model context window.
        """
        if not resolved_source_ids:
            return ContextResult(
                context_text="No documents available.",
                mode_used="none",
                total_chunks=0,
                total_estimated_tokens=0,
            )

        cfg = ArtifactContextBuilder._get_config(artifact_type)

        # allow caller override
        if max_context_tokens is None:
            max_context_tokens = cfg.max_context_tokens
        if max_context_chars is None:
            max_context_chars = cfg.max_context_chars

        has_focus = bool((prompt or "").strip())

        try:
            semantic_chunks = ArtifactContextBuilder._semantic_retrieval(
                user_id=user_id,
                source_ids=resolved_source_ids,
                artifact_type=artifact_type,
                prompt=prompt,
                semantic_k=cfg.semantic_k,
            )

            coverage_chunks = ArtifactContextBuilder._coverage_retrieval(
                user_id=user_id,
                source_ids=resolved_source_ids,
                artifact_type=artifact_type,
                per_source_k=cfg.coverage_k_per_source,
            )

            selected = ArtifactContextBuilder._merge_and_select(
                semantic_chunks=semantic_chunks,
                coverage_chunks=coverage_chunks,
                artifact_type=artifact_type,
                config=cfg,
                has_focus=has_focus,
                max_context_tokens=max_context_tokens,
            )

            context_text = ArtifactContextBuilder._assemble_context(
                chunks=selected,
                artifact_type=artifact_type,
                prompt=prompt,
                max_context_chars=max_context_chars,
            )

            refs = [chunk.to_source_ref() for chunk in selected]
            total_estimated_tokens = sum(c.estimated_tokens for c in selected)

            return ContextResult(
                context_text=context_text,
                source_refs=refs,
                mode_used="compact_hybrid_focused" if has_focus else "compact_hybrid_broad",
                total_chunks=len(selected),
                total_estimated_tokens=total_estimated_tokens,
            )

        except Exception as e:
            logger.error(
                f"ArtifactContextBuilder failed for artifact_type={artifact_type}: {e}",
                exc_info=True,
            )
            return ContextResult(
                context_text="No documents available.",
                mode_used="error",
                total_chunks=0,
                total_estimated_tokens=0,
            )

    # ── Retrieval ──────────────────────────────────────────────────────────────

    @staticmethod
    def _semantic_retrieval(
        user_id: uuid.UUID,
        source_ids: List[str],
        artifact_type: str,
        prompt: str | None,
        semantic_k: int,
    ) -> List[RetrievedChunk]:
        """
        Global semantic retrieval across all selected sources.

        Build Semantic Query
        ├── If prompt exists:
        │   └── "User request: {prompt}\nArtifact type: {artifact_type}\n{hint}"
        ├── If no prompt:
        │   └── "Artifact type: {artifact_type}\n{hint}"
        └── Example for "quiz":
            └── "User request: Create quiz on ML basics\nArtifact type: quiz\nRetrieve concepts, definitions, important facts..."

        Embed & Search Qdrant
        ├── query_vector = embeddings.embed_query(semantic_query)
        ├── Filter: user_id + source_id IN source_ids
        ├── Search: query_points(limit=semantic_k)  # e.g., 12 chunks
        └── Returns: Top K semantically similar chunks

        Process Semantic Hits
        └── Convert to RetrievedChunk objects
            ├── retrieval_bucket = "semantic"
            ├── estimated_tokens = len(text)/4 + 6
            └── compute artifact_score (more on this later)
        """
        query = ArtifactContextBuilder._build_semantic_query(
            artifact_type=artifact_type,
            prompt=prompt,
        )

        embeddings = get_embeddings()
        client = get_qdrant_client()

        query_vector = embeddings.embed_query(query)
        search_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=str(user_id))),
                FieldCondition(key="source_id", match=MatchAny(any=source_ids)),
            ]
        )

        results = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            query_filter=search_filter,
            limit=semantic_k,
            with_payload=True,
            with_vectors=False,
        )

        return ArtifactContextBuilder._process_hits(
            results.points,
            retrieval_bucket="semantic",
        )

    @staticmethod
    def _coverage_retrieval(
        user_id: uuid.UUID,
        source_ids: List[str],
        artifact_type: str,
        per_source_k: int,
    ) -> List[RetrievedChunk]:
        """
        Retrieve a few chunks per source to ensure coverage and source diversity.

        Strategy:
        - run one artifact-specific generic query per source
        - fetch a few extra candidates
        - select compact diverse picks from that source
        """
        embeddings = get_embeddings()
        client = get_qdrant_client()

        query = ArtifactContextBuilder._build_coverage_query(artifact_type)
        query_vector = embeddings.embed_query(query)

        all_selected: List[RetrievedChunk] = []

        for sid in source_ids:
            source_filter = Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=str(user_id))),
                    FieldCondition(key="source_id", match=MatchValue(value=sid)),
                ]
            )

            # fetch extra candidates so we can choose a better diverse subset
            fetch_limit = max(per_source_k * 3, 6)

            results = client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                query_filter=source_filter,
                limit=fetch_limit,
                with_payload=True,
                with_vectors=False,
            )

            candidates = ArtifactContextBuilder._process_hits(
                results.points,
                retrieval_bucket="coverage",
            )
            if not candidates:
                continue

            selected = ArtifactContextBuilder._select_compact_source_coverage(
                candidates=candidates,
                limit=per_source_k,
            )
            all_selected.extend(selected)

        return all_selected

    # ── Query builders ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_semantic_query(artifact_type: str, prompt: str | None) -> str:
        """Build the semantic search query from available inputs."""
        hint = ARTIFACT_HINTS.get(artifact_type, ARTIFACT_HINTS["artifact"])

        if prompt and prompt.strip():
            return f"User request: {prompt.strip()}\nArtifact type: {artifact_type}\n{hint}"

        return f"Artifact type: {artifact_type}\n{hint}"

    @staticmethod
    def _build_coverage_query(artifact_type: str) -> str:
        """Build the coverage search query."""
        hint = ARTIFACT_HINTS.get(artifact_type, ARTIFACT_HINTS["artifact"])
        return f"Important information from this source for {artifact_type} generation. {hint}"

    @staticmethod
    def _process_hits(points: list, retrieval_bucket: str) -> List[RetrievedChunk]:
        """Extract chunk data from Qdrant hits."""
        chunks: List[RetrievedChunk] = []

        for hit in points:
            payload = hit.payload or {}
            chunk_text = (payload.get("chunk_text") or "").strip()
            if not chunk_text:
                continue

            chunks.append(
                RetrievedChunk(
                    source_id=str(payload.get("source_id", "")),
                    source_type=str(payload.get("source_type", "document")).upper(),
                    title=payload.get("title") or payload.get("file_name") or "Unknown Source",
                    file_name=payload.get("file_name") or payload.get("title") or "Unknown Source",
                    page_number=payload.get("page_number"),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    chunk_text=chunk_text,
                    similarity_score=float(getattr(hit, "score", 0.0) or 0.0),
                    is_table=bool(payload.get("is_table", False)),
                    retrieval_bucket=retrieval_bucket,
                    estimated_tokens=ArtifactContextBuilder._estimate_tokens(chunk_text),
                )
            )
        return chunks

    @staticmethod
    def _select_compact_source_coverage(
        candidates: List[RetrievedChunk],
        limit: int,
    ) -> List[RetrievedChunk]:
        """
        Select a compact, diverse subset from a single source.

        Heuristic:
        1. take best chunk
        2. if available, prefer a table chunk as second pick
        3. otherwise pick a chunk far away in chunk_index from the first one
        """
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda c: c.similarity_score, reverse=True)

        selected: List[RetrievedChunk] = []
        used_ids: Set[str] = set()

        # 1) best chunk
        first = candidates[0]
        selected.append(first)
        used_ids.add(first.unique_id)

        if limit == 1:
            return selected

        # 2) prefer a table chunk if available and not same chunk
        table_candidate = None
        for c in candidates[1:]:
            if c.unique_id in used_ids:
                continue
            if c.is_table:
                table_candidate = c
                break

        if table_candidate:
            selected.append(table_candidate)
            used_ids.add(table_candidate.unique_id)

        # 3) fill remaining slots with chunk-index diversity
        if len(selected) < limit:
            remaining = [c for c in candidates if c.unique_id not in used_ids]
            diverse = ArtifactContextBuilder._pick_far_apart_chunks(
                anchor_chunks=selected,
                candidates=remaining,
                needed=limit - len(selected),
            )
            for c in diverse:
                if c.unique_id not in used_ids and len(selected) < limit:
                    selected.append(c)
                    used_ids.add(c.unique_id)

        return selected[:limit]

    @staticmethod
    def _pick_far_apart_chunks(
        anchor_chunks: List[RetrievedChunk],
        candidates: List[RetrievedChunk],
        needed: int,
    ) -> List[RetrievedChunk]:
        """
        Pick chunks that are far in chunk_index from already selected chunks.
        This is a cheap way to avoid grabbing only one tiny region of a source.
        """
        if needed <= 0 or not candidates:
            return []

        chosen: List[RetrievedChunk] = []
        used_anchor_positions = [c.chunk_index for c in anchor_chunks]

        remaining = candidates[:]
        while remaining and len(chosen) < needed:
            best = None
            best_distance = -1

            for c in remaining:
                min_dist = min(abs(c.chunk_index - pos) for pos in used_anchor_positions)
                # combine distance and semantic relevance lightly
                score = (min_dist * 0.01) + c.similarity_score
                if score > best_distance:
                    best_distance = score
                    best = c

            if best is None:
                break

            chosen.append(best)
            used_anchor_positions.append(best.chunk_index)
            remaining = [c for c in remaining if c.unique_id != best.unique_id]

        return chosen

    # ── Merge + rank + select ─────────────────────────────────────────────────

    @staticmethod
    def _merge_and_select(
        semantic_chunks: List[RetrievedChunk],
        coverage_chunks: List[RetrievedChunk],
        artifact_type: str,
        config: ArtifactRetrievalConfig,
        has_focus: bool,
        max_context_tokens: int,
    ) -> List[RetrievedChunk]:
        """
        Merge semantic and coverage pools under a small token budget.
        """
        # Dedup pools first
        semantic_chunks = ArtifactContextBuilder._dedup_chunks(semantic_chunks)
        coverage_chunks = ArtifactContextBuilder._dedup_chunks(coverage_chunks)

        # Score each chunk
        for chunk in semantic_chunks:
            chunk.artifact_score = ArtifactContextBuilder._compute_artifact_score(
                chunk=chunk,
                artifact_type=artifact_type,
                config=config,
            )

        for chunk in coverage_chunks:
            chunk.artifact_score = ArtifactContextBuilder._compute_artifact_score(
                chunk=chunk,
                artifact_type=artifact_type,
                config=config,
            )

        semantic_ranked = sorted(
            semantic_chunks,
            key=lambda c: (c.artifact_score, c.similarity_score),
            reverse=True,
        )
        coverage_ranked = sorted(
            coverage_chunks,
            key=lambda c: (c.artifact_score, c.similarity_score),
            reverse=True,
        )

        semantic_ratio = (
            config.focused_semantic_ratio if has_focus else config.broad_semantic_ratio
        )
        semantic_target = max(1, round(config.final_chunk_target * semantic_ratio))
        coverage_target = max(1, config.final_chunk_target - semantic_target)

        initial = semantic_ranked[:semantic_target] + coverage_ranked[:coverage_target]
        merged = ArtifactContextBuilder._dedup_chunks(initial)

        # Backfill if dedup removed too many
        if len(merged) < config.final_chunk_target:
            leftovers = ArtifactContextBuilder._dedup_chunks(
                semantic_ranked[semantic_target:] + coverage_ranked[coverage_target:]
            )
            seen = {m.unique_id for m in merged}
            for chunk in leftovers:
                if len(merged) >= config.final_chunk_target:
                    break
                if chunk.unique_id not in seen:
                    merged.append(chunk)
                    seen.add(chunk.unique_id)

        # Global rerank
        merged = sorted(
            merged,
            key=lambda c: (
                c.artifact_score,
                c.similarity_score,
                1 if c.retrieval_bucket == "semantic" else 0,
            ),
            reverse=True,
        )

        # Final compact selection under:
        # - source caps
        # - table caps
        # - token budget
        return ArtifactContextBuilder._apply_caps_and_budget(
            ranked_chunks=merged,
            config=config,
            max_context_tokens=max_context_tokens,
        )

    @staticmethod
    def _dedup_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Dedup by source_id + chunk_index.
        """
        best_by_id: Dict[str, RetrievedChunk] = {}

        for chunk in chunks:
            existing = best_by_id.get(chunk.unique_id)
            if existing is None:
                best_by_id[chunk.unique_id] = chunk
                continue

            # Prefer higher similarity/artifact score
            existing_score = max(existing.artifact_score, existing.similarity_score)
            new_score = max(chunk.artifact_score, chunk.similarity_score)

            if new_score > existing_score:
                best_by_id[chunk.unique_id] = chunk
            elif math.isclose(new_score, existing_score, rel_tol=1e-9):
                # prefer semantic over coverage if tied
                if existing.retrieval_bucket != "semantic" and chunk.retrieval_bucket == "semantic":
                    best_by_id[chunk.unique_id] = chunk

        return list(best_by_id.values())

    @staticmethod
    def _compute_artifact_score(
        chunk: RetrievedChunk,
        artifact_type: str,
        config: ArtifactRetrievalConfig,
    ) -> float:
        """
        Compact heuristic reranker.
        Works with your current payload (no rich metadata required).
        """
        score = 0.0

        # base semantic relevance
        score += chunk.similarity_score * 0.65

        # coverage chunks get small bonus so they survive merging
        if chunk.retrieval_bucket == "coverage":
            score += 0.05

        # table bonus
        if config.prefer_tables and chunk.is_table:
            score += 0.10

        text = chunk.chunk_text.lower()

        # factual density cues (good for quiz/flashcards)
        if config.prefer_fact_density:
            score += ArtifactContextBuilder._fact_density_bonus(text)

        # explanatory cues (good for faq/summary/study guide)
        if config.prefer_explanatory_chunks:
            score += ArtifactContextBuilder._explanatory_bonus(text)

        # summary-like cues
        if config.prefer_summary_like_chunks:
            score += ArtifactContextBuilder._summary_bonus(text)

        # q/a-like cues
        if config.prefer_qa_like_chunks:
            score += ArtifactContextBuilder._qa_bonus(text)

        # mild length preference:
        # very tiny chunks are often poor; huge chunks are expensive
        tok = chunk.estimated_tokens
        if 60 <= tok <= 220:
            score += 0.05
        elif 221 <= tok <= 320:
            score += 0.03
        elif tok < 25:
            score -= 0.04
        elif tok > 420:
            score -= 0.08

        # slight preference for earlier chunks in transcript-like material only if close score
        # this is intentionally tiny
        if chunk.chunk_index < 20:
            score += 0.01

        return round(score, 6)

    @staticmethod
    def _apply_caps_and_budget(
        ranked_chunks: List[RetrievedChunk],
        config: ArtifactRetrievalConfig,
        max_context_tokens: int,
    ) -> List[RetrievedChunk]:
        """
        Final compact selection with source caps and token budget.
        """
        selected: List[RetrievedChunk] = []
        source_counts: Dict[str, int] = {}
        source_table_counts: Dict[str, int] = {}
        used_tokens = 0

        for chunk in ranked_chunks:
            if len(selected) >= config.final_chunk_target:
                break

            src_count = source_counts.get(chunk.source_id, 0)
            if src_count >= config.max_chunks_per_source:
                continue

            if chunk.is_table:
                table_count = source_table_counts.get(chunk.source_id, 0)
                if table_count >= config.max_table_chunks_per_source:
                    continue

            # reserve overhead for formatting / separators
            projected = used_tokens + chunk.estimated_tokens + 18
            if projected > max_context_tokens:
                continue

            selected.append(chunk)
            used_tokens = projected
            source_counts[chunk.source_id] = src_count + 1

            if chunk.is_table:
                source_table_counts[chunk.source_id] = (
                    source_table_counts.get(chunk.source_id, 0) + 1
                )

        return selected

    # ── Context assembly ──────────────────────────────────────────────────────

    @staticmethod
    def _assemble_context(
        chunks: List[RetrievedChunk],
        artifact_type: str,
        prompt: str | None,
        max_context_chars: int,
    ) -> str:
        """
        Build compact structured context for the downstream generator.
        """
        if not chunks:
            return "No documents available."

        semantic_chunks = [c for c in chunks if c.retrieval_bucket == "semantic"]
        coverage_chunks = [c for c in chunks if c.retrieval_bucket == "coverage"]

        parts: List[str] = []
        parts.append(f"# Artifact Type: {artifact_type}")


        if prompt and prompt.strip():
            parts.append(f"# User Prompt: {prompt.strip()}")

        parts.append(
            "# Use only the grounded source information below. Prefer factual, specific, "
            "and source-supported content. Avoid inventing details."
        )

        if semantic_chunks:
            parts.append("## Focused evidence")
            for chunk in semantic_chunks:
                parts.append(ArtifactContextBuilder._format_chunk(chunk))

        if coverage_chunks:
            parts.append("## Supporting evidence")
            for chunk in coverage_chunks:
                parts.append(ArtifactContextBuilder._format_chunk(chunk))

        context = "\n\n---\n\n".join(parts)

        if len(context) > max_context_chars:
            context = context[:max_context_chars]
            last_sep = context.rfind("\n\n---\n\n")
            if last_sep > 0:
                context = context[:last_sep]
            else:
                last_space = context.rfind(" ")
                if last_space > 0:
                    context = context[:last_space]

        return context

    @staticmethod
    def _format_chunk(chunk: RetrievedChunk) -> str:
        """Format a single chunk for inclusion in context."""
        location = chunk.page_number if chunk.page_number is not None else "N/A"
        flags = " | TABLE" if chunk.is_table else ""

        header = (
            f"[{chunk.source_type}: {chunk.title} | Location: {location} | "
            f"Chunk: {chunk.chunk_index}{flags}]"
        )

        return f"{header}\n{chunk.chunk_text}"

    # ── Utility helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_config(artifact_type: str) -> ArtifactRetrievalConfig:
        """Get configuration for the given artifact type."""
        return ARTIFACT_CONFIGS.get(artifact_type, ARTIFACT_CONFIGS["artifact"])

    @staticmethod
    def _get_artifact_hint(artifact_type: str) -> str:
        """Get the retrieval hint for the given artifact type."""
        return ARTIFACT_HINTS.get(artifact_type, ARTIFACT_HINTS["artifact"])

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough token estimate.
        all-MiniLM chunk text tends to be plain text, so chars/4 is fine enough here.
        """
        return max(1, int(len(text) / 4) + 6)

    # ── Text heuristics for reranking ────────────────────────────────────────

    @staticmethod
    def _fact_density_bonus(text: str) -> float:
        """
        Useful for quiz/flashcards.
        Looks for signs of dense factual content without requiring metadata.
        """
        bonus = 0.0

        # numbers / enumerations / named patterns often indicate concrete facts
        if re.search(r"\b\d+\b", text):
            bonus += 0.02

        if ":" in text:
            bonus += 0.01

        if any(token in text for token in [" is ", " refers to ", " defined as ", " means "]):
            bonus += 0.03

        if any(token in text for token in ["types of", "steps", "advantages", "disadvantages"]):
            bonus += 0.02

        if len(re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", text)) >= 3:
            bonus += 0.01

        return bonus

    @staticmethod
    def _explanatory_bonus(text: str) -> float:
        """
        Useful for FAQ / study guide / summary.
        """
        bonus = 0.0
        cues = [
            "because",
            "therefore",
            "this means",
            "in order to",
            "for example",
            "for instance",
            "so that",
            "the reason",
            "how to",
            "why",
            "what happens",
        ]
        for cue in cues:
            if cue in text:
                bonus += 0.01

        return min(bonus, 0.06)

    @staticmethod
    def _summary_bonus(text: str) -> float:
        """Bonus for summary-like content."""
        bonus = 0.0
        cues = [
            "in summary",
            "to summarize",
            "overall",
            "in conclusion",
            "main idea",
            "key takeaway",
            "summary",
            "conclusion",
        ]
        for cue in cues:
            if cue in text:
                bonus += 0.015
        return min(bonus, 0.06)

    @staticmethod
    def _qa_bonus(text: str) -> float:
        """Bonus for Q&A-like content."""
        bonus = 0.0
        if "?" in text:
            bonus += 0.03

        cues = [
            "what is",
            "how does",
            "how do",
            "why does",
            "why do",
            "question",
            "answer",
        ]
        for cue in cues:
            if cue in text:
                bonus += 0.01

        return min(bonus, 0.06)