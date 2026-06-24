# artifact_models.py
"""Pydantic models for all artifact types and evidence pack."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Type Definitions ──────────────────────────────────────────────────────────

ArtifactType = Literal[
    "quiz",
    "flashcards",
    "faq",
    "study_guide",
    "summary",
    "mindmap",
    "slide_deck",
    "artifact",
]


# ── Shared Models ─────────────────────────────────────────────────────────────

class ArtifactSourceRef(BaseModel):
    """Reference to a source used in artifact generation."""
    source_id: str = Field(..., description="Unique source identifier")
    file_name: str = Field(..., description="Source file name")
    page_number: int | str | None = Field(None, description="Page number or location")
    chunk_index: int = Field(..., description="Chunk index within the source")
    similarity_score: float = Field(..., description="Qdrant similarity score")


class ArtifactMetadata(BaseModel):
    """Metadata about artifact generation."""
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    title: Optional[str] = Field(None, description="Artifact title")
    topic: Optional[str] = Field(None, description="User-provided topic")
    prompt: Optional[str] = Field(None, description="User-provided prompt")
    source_ids: List[str] = Field(default_factory=list, description="Source IDs used")
    total_chunks_used: int = Field(0, description="Total chunks used in context")
    estimated_context_tokens: int = Field(0, description="Estimated token count")
    compression_used: bool = Field(True, description="Whether compression was used")


class ArtifactEnvelope(BaseModel):
    """Wrapper for storing any artifact with metadata."""
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    data: Dict[str, Any] = Field(..., description="The actual artifact data")
    metadata: ArtifactMetadata = Field(..., description="Generation metadata")
    source_refs: List[ArtifactSourceRef] = Field(default_factory=list, description="Source references")


# ── Quiz Models ───────────────────────────────────────────────────────────────

class QuizOption(BaseModel):
    """A single quiz option."""
    id: str = Field(..., description="Option identifier (A, B, C, D)")
    text: str = Field(..., description="Option text")


class QuizQuestion(BaseModel):
    """A single quiz question."""
    question: str = Field(..., description="The quiz question text")
    type: Literal["mcq", "true_false"] = Field("mcq", description="Question type")
    options: List[QuizOption] = Field(default_factory=list, description="Answer options")
    answer: str = Field(..., description="Correct answer (A, B, C, D, or True/False)")
    explanation: str = Field(..., description="Explanation of the correct answer")


class QuizArtifact(BaseModel):
    """Complete quiz artifact."""
    title: str = Field(..., description="Quiz title")
    description: Optional[str] = Field(None, description="Quiz description")
    questions: List[QuizQuestion] = Field(default_factory=list, description="Quiz questions")


# ── Flashcard Models ─────────────────────────────────────────────────────────

class FlashcardItem(BaseModel):
    """A single flashcard."""
    front: str = Field(..., description="Front of the card (question/concept)")
    back: str = Field(..., description="Back of the card (answer/explanation)")
    hint: Optional[str] = Field(None, description="Optional hint")


class FlashcardsArtifact(BaseModel):
    """Complete flashcards artifact."""
    title: str = Field(..., description="Flashcards title")
    description: Optional[str] = Field(None, description="Flashcards description")
    cards: List[FlashcardItem] = Field(default_factory=list, description="Flashcards")


# ── FAQ Models ───────────────────────────────────────────────────────────────

class FAQItem(BaseModel):
    """A single FAQ entry."""
    question: str = Field(..., description="The question")
    answer: str = Field(..., description="The answer")


class FAQArtifact(BaseModel):
    """Complete FAQ artifact."""
    title: str = Field(..., description="FAQ title")
    description: Optional[str] = Field(None, description="FAQ description")
    items: List[FAQItem] = Field(default_factory=list, description="FAQ items")


# ── Summary Models ───────────────────────────────────────────────────────────

class SummarySection(BaseModel):
    """A section in a summary."""
    heading: str = Field(..., description="Section heading")
    bullets: List[str] = Field(default_factory=list, description="Bullet points")


class SummaryArtifact(BaseModel):
    """Complete summary artifact."""
    title: str = Field(..., description="Summary title")
    overview: str = Field(..., description="Overview paragraph")
    key_points: List[str] = Field(default_factory=list, description="Key takeaways")
    sections: List[SummarySection] = Field(default_factory=list, description="Detailed sections")


# ── Study Guide Models ──────────────────────────────────────────────────────

class StudyGuideSection(BaseModel):
    """A section in a study guide."""
    heading: str = Field(..., description="Section heading")
    explanation: str = Field(..., description="Detailed explanation")
    key_points: List[str] = Field(default_factory=list, description="Key points")
    important_terms: List[str] = Field(default_factory=list, description="Important terms with definitions")


class StudyGuideArtifact(BaseModel):
    """Complete study guide artifact."""
    title: str = Field(..., description="Study guide title")
    overview: str = Field(..., description="Overview paragraph")
    sections: List[StudyGuideSection] = Field(default_factory=list, description="Study guide sections")
    review_questions: List[str] = Field(default_factory=list, description="Review questions")


# ── Mind Map Models ─────────────────────────────────────────────────────────

class MindMapNode(BaseModel):
    """A node in a mind map."""
    label: str = Field(..., description="Node label")
    children: List[MindMapNode] = Field(default_factory=list, description="Child nodes")


# Enable recursive model reference
MindMapNode.model_rebuild()


class MindMapArtifact(BaseModel):
    """Complete mind map artifact."""
    title: str = Field(..., description="Mind map title")
    root: MindMapNode = Field(..., description="Root node")


# ── Slide Deck Models ───────────────────────────────────────────────────────

class SlideItem(BaseModel):
    """A single slide."""
    title: str = Field(..., description="Slide title")
    bullets: List[str] = Field(default_factory=list, description="Bullet points")
    speaker_notes: Optional[str] = Field(None, description="Speaker notes")


class SlideDeckArtifact(BaseModel):
    """Complete slide deck artifact."""
    title: str = Field(..., description="Slide deck title")
    slides: List[SlideItem] = Field(default_factory=list, description="Slides")


# ── Evidence Pack Models ────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """A single piece of evidence."""
    fact: str = Field(..., description="The fact or statement")
    source_label: Optional[str] = Field(None, description="Source reference")
    importance: Literal["high", "medium", "low"] = Field("medium", description="Importance level")


class EvidencePack(BaseModel):
    """Compressed evidence pack for artifact generation."""
    title: str = Field("Evidence Pack", description="Pack title")
    topic: Optional[str] = Field(None, description="Main topic")
    facts: List[EvidenceItem] = Field(default_factory=list, description="Key facts")
    concepts: List[str] = Field(default_factory=list, description="Important concepts")
    formulas: List[str] = Field(default_factory=list, description="Relevant formulas")
    definitions: List[str] = Field(default_factory=list, description="Key definitions")
    procedures: List[str] = Field(default_factory=list, description="Procedures or steps")
    examples: List[str] = Field(default_factory=list, description="Concrete examples")

    def get_high_importance_facts(self) -> List[EvidenceItem]:
        """Get only high importance facts."""
        return [f for f in self.facts if f.importance == "high"]

    def get_medium_importance_facts(self) -> List[EvidenceItem]:
        """Get only medium importance facts."""
        return [f for f in self.facts if f.importance == "medium"]

    def get_low_importance_facts(self) -> List[EvidenceItem]:
        """Get only low importance facts."""
        return [f for f in self.facts if f.importance == "low"]

    def get_all_facts_text(self) -> List[str]:
        """Get all facts as plain text."""
        return [f.fact for f in self.facts]

    def get_all_content(self) -> List[str]:
        """Get all content as a flat list of strings."""
        content = []
        content.extend(self.get_all_facts_text())
        content.extend(self.concepts)
        content.extend(self.formulas)
        content.extend(self.definitions)
        content.extend(self.procedures)
        content.extend(self.examples)
        return content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidencePack:
        """Create from dictionary."""
        return cls.model_validate(data)

    def get_summary_stats(self) -> Dict[str, int]:
        """Get summary statistics."""
        return {
            "total_facts": len(self.facts),
            "total_concepts": len(self.concepts),
            "total_formulas": len(self.formulas),
            "total_definitions": len(self.definitions),
            "total_procedures": len(self.procedures),
            "total_examples": len(self.examples),
            "high_importance": len(self.get_high_importance_facts()),
            "medium_importance": len(self.get_medium_importance_facts()),
            "low_importance": len(self.get_low_importance_facts()),
            "total_items": len(self.get_all_content()),
        }

    def is_empty(self) -> bool:
        """Check if the evidence pack is empty."""
        return (
            len(self.facts) == 0 and
            len(self.concepts) == 0 and
            len(self.formulas) == 0 and
            len(self.definitions) == 0 and
            len(self.procedures) == 0 and
            len(self.examples) == 0
        )

    def to_markdown(self) -> str:
        """Convert evidence pack to markdown format for debugging."""
        lines = []
        lines.append(f"# {self.title}")
        if self.topic:
            lines.append(f"**Topic:** {self.topic}")
        lines.append("")

        if self.facts:
            lines.append("## Facts")
            for fact in self.facts:
                importance_map = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                icon = importance_map.get(fact.importance, "•")
                source = f" ({fact.source_label})" if fact.source_label else ""
                lines.append(f"- {icon} {fact.fact}{source}")
            lines.append("")

        if self.concepts:
            lines.append("## Concepts")
            for concept in self.concepts:
                lines.append(f"- {concept}")
            lines.append("")

        if self.definitions:
            lines.append("## Definitions")
            for definition in self.definitions:
                lines.append(f"- {definition}")
            lines.append("")

        if self.formulas:
            lines.append("## Formulas")
            for formula in self.formulas:
                lines.append(f"- {formula}")
            lines.append("")

        if self.procedures:
            lines.append("## Procedures")
            for procedure in self.procedures:
                lines.append(f"- {procedure}")
            lines.append("")

        if self.examples:
            lines.append("## Examples")
            for example in self.examples:
                lines.append(f"- {example}")

        return "\n".join(lines)


# ── Factory Functions ────────────────────────────────────────────────────────

def create_empty_evidence_pack(topic: Optional[str] = None) -> EvidencePack:
    """Create an empty evidence pack."""
    return EvidencePack(
        title="Evidence Pack",
        topic=topic,
        facts=[],
        concepts=[],
        formulas=[],
        definitions=[],
        procedures=[],
        examples=[],
    )


def create_evidence_pack_from_facts(
    facts: List[str],
    topic: Optional[str] = None,
    importance: Literal["high", "medium", "low"] = "medium",
) -> EvidencePack:
    """Create an evidence pack from a list of facts."""
    return EvidencePack(
        title="Evidence Pack",
        topic=topic,
        facts=[EvidenceItem(fact=f, importance=importance) for f in facts],
        concepts=[],
        formulas=[],
        definitions=[],
        procedures=[],
        examples=[],
    )


def create_evidence_pack_from_text(
    text: str,
    topic: Optional[str] = None,
    max_items: int = 20,
) -> EvidencePack:
    """
    Create a simple evidence pack from plain text.
    Splits text by sentences or lines.
    """
    # Split by sentences or lines
    import re
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    # Take up to max_items
    sentences = sentences[:max_items]
    
    facts = [
        EvidenceItem(fact=s, importance="medium") 
        for s in sentences
    ]
    
    return EvidencePack(
        title="Evidence Pack",
        topic=topic,
        facts=facts,
        concepts=[],
        formulas=[],
        definitions=[],
        procedures=[],
        examples=[],
    )


# ── Artifact Type Registry ──────────────────────────────────────────────────

ARTIFACT_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    "quiz": {
        "model": QuizArtifact,
        "display_name": "Quiz",
        "description": "Multiple-choice quiz with explanations",
        "icon": "📝",
        "default_options": {"question_count": 10, "difficulty": "mixed"},
    },
    "flashcards": {
        "model": FlashcardsArtifact,
        "display_name": "Flashcards",
        "description": "Flashcards for memorization and review",
        "icon": "🃏",
        "default_options": {"card_count": 12},
    },
    "faq": {
        "model": FAQArtifact,
        "display_name": "FAQ",
        "description": "Frequently asked questions with answers",
        "icon": "❓",
        "default_options": {"faq_count": 10},
    },
    "study_guide": {
        "model": StudyGuideArtifact,
        "display_name": "Study Guide",
        "description": "Comprehensive study guide with sections and review questions",
        "icon": "📚",
        "default_options": {"size": "medium"},
    },
    "summary": {
        "model": SummaryArtifact,
        "display_name": "Summary",
        "description": "Concise summary with key points",
        "icon": "📄",
        "default_options": {},
    },
    "mindmap": {
        "model": MindMapArtifact,
        "display_name": "Mind Map",
        "description": "Hierarchical concept map",
        "icon": "🧠",
        "default_options": {},
    },
    "slide_deck": {
        "model": SlideDeckArtifact,
        "display_name": "Slide Deck",
        "description": "Presentation slide deck outline",
        "icon": "📊",
        "default_options": {"slide_count": 8},
    },
}


def get_artifact_model(artifact_type: str):
    """Get the Pydantic model class for an artifact type."""
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    if info:
        return info["model"]
    return SummaryArtifact  # Default fallback


def get_artifact_display_name(artifact_type: str) -> str:
    """Get the display name for an artifact type."""
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    return info["display_name"] if info else artifact_type.title()


def get_artifact_description(artifact_type: str) -> str:
    """Get the description for an artifact type."""
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    return info["description"] if info else "Learning artifact"


def get_artifact_icon(artifact_type: str) -> str:
    """Get the icon for an artifact type."""
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    return info["icon"] if info else "📄"


def get_artifact_default_options(artifact_type: str) -> Dict[str, Any]:
    """Get default options for an artifact type."""
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    return info["default_options"] if info else {}