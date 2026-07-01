"""Unified Pydantic schemas for API requests/responses, RAG metadata, and LLM structured output."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional,Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class ArtifactType(str, Enum):
    QUIZ = "quiz"
    FLASHCARDS = "flashcards"
    FAQ = "faq"
    STUDY_GUIDE = "study_guide"
    SUMMARY = "summary"
    MINDMAP = "mindmap"
    SLIDE_DECK = "slide_deck"
    VOICE_OVERVIEW = "voice_overview"
    REPORT = "report"
    DATATABLE = "datatable"
    GENERIC = "artifact"


class ArtifactStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class QuizDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    MIX = "mix"


class StudyGuideSize(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LARGE = "large"


class ReportLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LARGE = "large"


class AudioOverviewLength(str, Enum):
    SHORT = "short"      # ~3-5 min, ~700-1000 words
    MEDIUM = "medium"    # ~6-9 min, ~1200-1800 words
    LONG = "long"         # ~10-15 min, ~2000-2800 words


class AudioOverviewVoiceStyle(str, Enum):
    """Voice pairing style for the two podcast hosts."""
    DEFAULT = "default"          # warm male + expressive female (Andrew & Ava)
    ENERGETIC = "energetic"      # both upbeat/energetic (Guy & Jenny)
    CALM = "calm"                # both calm/measured (Brian & Aria)


# ── API Request Schemas ───────────────────────────────────────────────────────

class BaseArtifactRequest(BaseModel):
    """Common fields shared by all artifact creation requests."""
    prompt: Optional[str] = Field(None, max_length=2000, description="Custom scope/instructions for generation")
    excluded_source_ids: List[str] = Field(
        default_factory=list,
        description="Source IDs to exclude from artifact generation context",
    )

class QuizCreateRequest(BaseArtifactRequest):
    number_of_questions: int = Field(5, ge=3, le=15, description="Number of quiz questions")
    difficulty: QuizDifficulty = QuizDifficulty.MIX

class FlashcardCreateRequest(BaseArtifactRequest):
    number_of_cards: int = Field(5, ge=3, le=15, description="Number of flashcards")

class FAQCreateRequest(BaseArtifactRequest):
    number_of_faqs: int = Field(5, ge=3, le=15, description="Number of FAQ items")

class StudyGuideCreateRequest(BaseArtifactRequest):
    size: StudyGuideSize = StudyGuideSize.MEDIUM

class SummaryCreateRequest(BaseArtifactRequest):
    pass

class MindMapCreateRequest(BaseArtifactRequest):
    pass

class SlideDeckCreateRequest(BaseArtifactRequest):
    number_of_slides: int = Field(8, ge=4, le=16, description="Number of slides")

class ReportCreateRequest(BaseArtifactRequest):
    length: ReportLength = Field(ReportLength.MEDIUM, description="Target length/depth of the report")

class DataTableCreateRequest(BaseArtifactRequest):
    max_rows: int = Field(15, ge=3, le=50, description="Maximum number of rows to include in the table")

class AudioOverviewCreateRequest(BaseArtifactRequest):
    length: AudioOverviewLength = Field(
        AudioOverviewLength.MEDIUM, description="Target length of the voice overview"
    )
    voice_style: AudioOverviewVoiceStyle = Field(
        AudioOverviewVoiceStyle.DEFAULT, description="Voice pairing style for the two hosts"
    )
    host_names: Optional[List[str]] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Optional custom display names for [host_1, host_2]. Defaults to 'Alex' and 'Jordan'.",
    )


# ── Internal RAG / Metadata Schemas ──────────────────────────────────────────

class ArtifactSourceRef(BaseModel):
    """Reference to a source used in artifact generation."""
    source_id: str = Field(..., description="Unique source identifier")
    file_name: str = Field(..., description="Source file name")
    page_number: int | str | None = Field(None, description="Page number or location")
    chunk_index: int = Field(..., description="Chunk index within the source")
    similarity_score: float = Field(..., description="Vector DB similarity score")

class ArtifactMetadata(BaseModel):
    """Metadata about artifact generation."""
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    title: Optional[str] = Field(None, description="Artifact title")
    prompt: Optional[str] = Field(None, description="User-provided prompt")
    source_ids: List[str] = Field(default_factory=list, description="Source IDs used")
    total_chunks_used: int = Field(0, description="Total chunks used in context")
    estimated_context_tokens: int = Field(0, description="Estimated token count")
    compression_used: bool = Field(True, description="Whether context compression was used")

class ArtifactEnvelope(BaseModel):
    """Wrapper for storing any generated artifact along with its context metadata."""
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    data: Dict[str, Any] = Field(..., description="The actual LLM structured output data")
    metadata: ArtifactMetadata = Field(..., description="Generation metadata")
    source_refs: List[ArtifactSourceRef] = Field(default_factory=list, description="Source references")

class SourceFilterInfo(BaseModel):
    """Internal schema for tracking source resolution."""
    excluded_source_ids: List[str] = Field(default_factory=list)
    resolved_source_ids: List[str] = Field(default_factory=list)


# ── LLM Structured Output Models (The Artifacts) ─────────────────────────────

# Quiz
class QuizOption(BaseModel):
    id: str = Field(..., description="Option identifier (A, B, C, D)")
    text: str = Field(..., description="Option text")

class QuizQuestion(BaseModel):
    question: str = Field(..., description="The quiz question text")
    type: str = Field("mcq", description="Question type: mcq or true_false")
    options: List[QuizOption] = Field(..., description="List of answer options as objects")
    answer: str = Field(..., description="The correct answer (A, B, C, D, or True/False)")
    explanation: str = Field(..., description="Brief explanation of why the answer is correct")

class QuizArtifact(BaseModel):
    title: str = Field(..., description="A concise title for the quiz")
    description: Optional[str] = Field(None, description="Short description of the quiz")
    questions: List[QuizQuestion] = Field(..., description="List of quiz questions")

# Flashcards
class FlashcardItem(BaseModel):
    front: str = Field(..., description="The question or concept on the front of the card")
    back: str = Field(..., description="The answer or explanation on the back of the card")

class FlashcardsArtifact(BaseModel):
    title: str = Field(..., description="A concise title for the flashcard set")
    description: Optional[str] = Field(None, description="Short description of the flashcard set")
    cards: List[FlashcardItem] = Field(..., description="List of flashcards")

# FAQ
class FAQItem(BaseModel):
    question: str = Field(..., description="The frequently asked question")
    answer: str = Field(..., description="Comprehensive answer to the question")

class FAQArtifact(BaseModel):
    title: str = Field(..., description="A concise title for the FAQ set")
    description: Optional[str] = Field(None, description="Short description of the FAQ set")
    items: List[FAQItem] = Field(..., description="List of FAQ entries")

# Summary
class SummarySection(BaseModel):
    heading: str = Field(..., description="Section heading")
    bullets: List[str] = Field(default_factory=list, description="Key points as bullets")

class SummaryArtifact(BaseModel):
    title: str = Field(..., description="A concise title for the summary")
    overview: str = Field(..., description="Short overview paragraph")
    key_points: List[str] = Field(default_factory=list, description="Key takeaways")
    sections: List[SummarySection] = Field(default_factory=list, description="Detailed sections")

# Study Guide
class StudyGuideSection(BaseModel):
    heading: str = Field(..., description="Section heading")
    explanation: str = Field(..., description="Detailed explanation of the section")
    key_points: List[str] = Field(default_factory=list, description="Key points from this section")
    important_terms: List[str] = Field(default_factory=list, description="Important terms and definitions")

class StudyGuideArtifact(BaseModel):
    title: str = Field(..., description="A concise title for the study guide")
    overview: str = Field(..., description="Short overview of the study guide")
    sections: List[StudyGuideSection] = Field(default_factory=list, description="Study guide sections")
    review_questions: List[str] = Field(default_factory=list, description="Review questions")

# Mind Map
class MindMapNode(BaseModel):
    label: str = Field(..., description="Node label")
    children: List["MindMapNode"] = Field(default_factory=list, description="Child nodes")

# Enable recursive model reference
MindMapNode.model_rebuild()

class MindMapArtifact(BaseModel):
    title: str = Field(..., description="A concise title for the mind map")
    root: MindMapNode = Field(..., description="Root node of the mind map")

# Slide Deck
class SlideVisual(BaseModel):
    type: Literal["none", "icon", "image"] = Field(
        ..., description="Visual type for the slide"
    )

    query: Optional[str] = Field(
        None,
        description=(
            "Search query for the visual. "
            "For images this should be descriptive (e.g. "
            "'modern cloud computing datacenter'). "
            "For icons use a concise concept (e.g. 'cloud', 'database', 'brain')."
        ),
    )

    placement: Literal[
        "left",
        "right",
        "top",
        "background"
    ] = Field(
        "right",
        description="Preferred placement of the visual."
    )


class SlideItem(BaseModel):
    title: str = Field(..., description="Slide title")

    type: Literal[
        "paragraph",
        "bullets_points",
        "2_paragraph_cards",
        "2_bullets_point_cards",
        "3_paragraph_cards",
        "3_bullets_point_cards",
        "table"
    ]

    visual: SlideVisual = Field(
        ...,
        description="Visual recommendation for the slide."
    )

    content: List[Any]


class SlideDeckArtifact(BaseModel):
    title: str = Field(..., description="Title for the slide deck")
    description: str = Field(..., description="Short description of the slide deck")
    slides: List[SlideItem] = Field(default_factory=list, description="List of slides")


# Report
class ReportSection(BaseModel):
    heading: str = Field(..., description="Section heading")
    body: str = Field(..., description="Section body text, written in clear prose")
    bullets: List[str] = Field(default_factory=list, description="Optional supporting bullet points")

class ReportArtifact(BaseModel):
    title: str = Field(..., description="A concise, professional title for the report")
    executive_summary: str = Field(..., description="Short executive summary of the report's findings")
    sections: List[ReportSection] = Field(default_factory=list, description="Main body sections of the report")
    key_findings: List[str] = Field(default_factory=list, description="Most important findings or takeaways")
    conclusion: str = Field(..., description="Closing conclusion or recommendation paragraph")


# Data Table
class DataTableColumn(BaseModel):
    name: str = Field(..., description="Column header/name")
    type: Literal["string", "number", "date", "boolean"] = Field(
        "string", description="Data type of the column's values"
    )

class DataTableArtifact(BaseModel):
    title: str = Field(..., description="A concise title describing what the table represents")
    description: Optional[str] = Field(None, description="Short description of the table's contents")
    columns: List[DataTableColumn] = Field(..., description="Ordered list of column definitions")
    rows: List[List[Any]] = Field(
        default_factory=list,
        description="Table rows; each row is a list of cell values matching the order of `columns`",
    )
    notes: List[str] = Field(default_factory=list, description="Optional notes or insights about the data")


# Audio Overview (two-host podcast script)
class DialogueLine(BaseModel):
    speaker: str = Field(..., description="Which host speaks this line: 'host_1' or 'host_2'")
    text: str = Field(..., description="The spoken line text, written for natural speech")

class AudioOverviewArtifact(BaseModel):
    title: str = Field(..., description="A concise, podcast-style title for the episode")
    description: Optional[str] = Field(None, description="One-sentence episode description")
    dialogue: List[DialogueLine] = Field(
        ..., description="Ordered back-and-forth dialogue lines between host_1 and host_2"
    )


class AudioOverviewMetadata(BaseModel):
    """Audio-specific metadata, stored nested inside content_json (no dedicated DB columns)."""
    audio_url: Optional[str] = Field(None, description="Playable URL of the generated audio")
    audio_file_id: Optional[str] = Field(None, description="Storage provider's file ID, needed to delete the file")
    audio_duration_seconds: Optional[float] = Field(None, description="Duration of the generated audio in seconds")


class AudioOverviewStoredContent(AudioOverviewArtifact):
    """
    What actually gets persisted into Artifact.content_json for VOICE_OVERVIEW artifacts.

    Extends the script (title/description/dialogue) with an "audio" key holding
    playback metadata, so no separate DB columns are needed for this artifact type.
    """
    audio: AudioOverviewMetadata = Field(default_factory=AudioOverviewMetadata)


# ── API Response Schemas ─────────────────────────────────────────────────────

class ArtifactShortResponse(BaseModel):
    id: uuid.UUID
    notebook_id: uuid.UUID
    user_id: uuid.UUID
    artifact_type: ArtifactType
    status: ArtifactStatus
    title: str
    created_at: datetime
    updated_at: datetime

class ArtifactResponse(BaseModel):
    """Response schema for artifact data (exposed to frontend)."""
    id: uuid.UUID
    notebook_id: uuid.UUID
    user_id: uuid.UUID
    artifact_type: ArtifactType
    status: ArtifactStatus
    title: str
    options_json: dict[str, Any]
    included_sources: List[str]
    content_json: dict[str, Any]
    # audio_url: Optional[str] = Field(None, description="Playable audio URL, only set for voice_overview artifacts")
    # audio_duration_seconds: Optional[float] = Field(None, description="Audio duration in seconds, only set for voice_overview artifacts")
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def _surface_audio_metadata(self) -> "ArtifactResponse":
        """
        For voice_overview artifacts, content_json stores an "audio" sub-object
        (no dedicated DB columns). Surface those two fields at the top level
        for frontend convenience without duplicating storage.
        """
        if self.artifact_type == ArtifactType.VOICE_OVERVIEW:
            audio_meta = (self.content_json or {}).get("audio") or {}
            self.audio_url = audio_meta.get("audio_url")
            self.audio_duration_seconds = audio_meta.get("audio_duration_seconds")
        return self

class ArtifactListResponse(BaseModel):
    """Response schema for listing artifacts with pagination."""
    artifacts: List[ArtifactShortResponse]
    total: int
    limit: Optional[int] = None
    offset: Optional[int] = None


# ── Evidence Pack Models & Logic ────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """A single piece of evidence."""
    fact: str = Field(..., description="The fact or statement")
    source_label: Optional[str] = Field(None, description="Source reference")
    importance: str = Field("medium", description="Importance level: high, medium, low")

class EvidencePack(BaseModel):
    """Compressed evidence pack for context injection prior to artifact generation."""
    title: str = Field("Evidence Pack", description="Pack title")
    facts: List[EvidenceItem] = Field(default_factory=list, description="Key facts")
    concepts: List[str] = Field(default_factory=list, description="Important concepts")
    formulas: List[str] = Field(default_factory=list, description="Relevant formulas")
    definitions: List[str] = Field(default_factory=list, description="Key definitions")
    procedures: List[str] = Field(default_factory=list, description="Procedures or steps")
    examples: List[str] = Field(default_factory=list, description="Concrete examples")

    def get_high_importance_facts(self) -> List[EvidenceItem]:
        return [f for f in self.facts if f.importance == "high"]

    def get_medium_importance_facts(self) -> List[EvidenceItem]:
        return [f for f in self.facts if f.importance == "medium"]

    def get_low_importance_facts(self) -> List[EvidenceItem]:
        return [f for f in self.facts if f.importance == "low"]

    def get_all_facts_text(self) -> List[str]:
        return [f.fact for f in self.facts]

    def get_all_content(self) -> List[str]:
        content = []
        content.extend(self.get_all_facts_text())
        content.extend(self.concepts)
        content.extend(self.formulas)
        content.extend(self.definitions)
        content.extend(self.procedures)
        content.extend(self.examples)
        return content

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidencePack:
        return cls.model_validate(data)

    def get_summary_stats(self) -> Dict[str, int]:
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
        return (
            len(self.facts) == 0 and
            len(self.concepts) == 0 and
            len(self.formulas) == 0 and
            len(self.definitions) == 0 and
            len(self.procedures) == 0 and
            len(self.examples) == 0
        )

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.facts:
            lines.append("## Facts")
            importance_map = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            for fact in self.facts:
                icon = importance_map.get(fact.importance, "•")
                source = f" ({fact.source_label})" if fact.source_label else ""
                lines.append(f"- {icon} {fact.fact}{source}")
            lines.append("")
        for attr in ["concepts", "definitions", "formulas", "procedures", "examples"]:
            items = getattr(self, attr)
            if items:
                lines.append(f"## {attr.capitalize()}")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")
        return "\n".join(lines).strip()

def create_empty_evidence_pack() -> EvidencePack:
    return EvidencePack()

def create_evidence_pack_from_facts(facts: List[str], importance: str = "medium") -> EvidencePack:
    return EvidencePack(facts=[EvidenceItem(fact=f, importance=importance) for f in facts])

def create_evidence_pack_from_text(text: str, max_items: int = 20) -> EvidencePack:
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:max_items]
    return EvidencePack(facts=[EvidenceItem(fact=s, importance="medium") for s in sentences])


# ── Artifact Type Registry ──────────────────────────────────────────────────

ARTIFACT_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    ArtifactType.QUIZ.value: {
        "model": QuizArtifact,
        "display_name": "Quiz",
        "description": "Multiple-choice quiz with explanations",
        "icon": "📝",
        "default_options": {"question_count": 10, "difficulty": "mixed"},
    },
    ArtifactType.FLASHCARDS.value: {
        "model": FlashcardsArtifact,
        "display_name": "Flashcards",
        "description": "Flashcards for memorization and review",
        "icon": "🃏",
        "default_options": {"card_count": 12},
    },
    ArtifactType.FAQ.value: {
        "model": FAQArtifact,
        "display_name": "FAQ",
        "description": "Frequently asked questions with answers",
        "icon": "❓",
        "default_options": {"faq_count": 10},
    },
    ArtifactType.STUDY_GUIDE.value: {
        "model": StudyGuideArtifact,
        "display_name": "Study Guide",
        "description": "Comprehensive study guide with sections and review questions",
        "icon": "📚",
        "default_options": {"size": "medium"},
    },
    ArtifactType.SUMMARY.value: {
        "model": SummaryArtifact,
        "display_name": "Summary",
        "description": "Concise summary with key points",
        "icon": "📄",
        "default_options": {},
    },
    ArtifactType.MINDMAP.value: {
        "model": MindMapArtifact,
        "display_name": "Mind Map",
        "description": "Hierarchical concept map",
        "icon": "🧠",
        "default_options": {},
    },
    ArtifactType.SLIDE_DECK.value: {
        "model": SlideDeckArtifact,
        "display_name": "Slide Deck",
        "description": "Presentation slide deck outline",
        "icon": "📊",
        "default_options": {"slide_count": 8},
    },
    ArtifactType.VOICE_OVERVIEW.value: {
        "model": AudioOverviewArtifact,
        "display_name": "Voice Overview",
        "description": "Two-host podcast-style audio discussion of your sources",
        "icon": "🎧",
        "default_options": {"length": "medium", "voice_style": "default"},
    },
    ArtifactType.REPORT.value: {
        "model": ReportArtifact,
        "display_name": "Report",
        "description": "Structured analytical report with findings and conclusion",
        "icon": "📑",
        "default_options": {"length": "medium"},
    },
    ArtifactType.DATATABLE.value: {
        "model": DataTableArtifact,
        "display_name": "Data Table",
        "description": "Structured data table extracted from your sources",
        "icon": "📋",
        "default_options": {"max_rows": 15},
    },
}

def get_artifact_model(artifact_type: str | ArtifactType):
    if isinstance(artifact_type, ArtifactType):
        artifact_type = artifact_type.value
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    return info["model"] if info else SummaryArtifact

def get_artifact_display_name(artifact_type: str | ArtifactType) -> str:
    if isinstance(artifact_type, ArtifactType):
        artifact_type = artifact_type.value
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    return info["display_name"] if info else str(artifact_type).title()

def get_artifact_description(artifact_type: str | ArtifactType) -> str:
    if isinstance(artifact_type, ArtifactType):
        artifact_type = artifact_type.value
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    return info["description"] if info else "Learning artifact"

def get_artifact_icon(artifact_type: str | ArtifactType) -> str:
    if isinstance(artifact_type, ArtifactType):
        artifact_type = artifact_type.value
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    return info["icon"] if info else "📄"

def get_artifact_default_options(artifact_type: str | ArtifactType) -> Dict[str, Any]:
    if isinstance(artifact_type, ArtifactType):
        artifact_type = artifact_type.value
    info = ARTIFACT_TYPE_INFO.get(artifact_type)
    return info["default_options"] if info else {}


# Aliases for generator compatibility
QuizOutput = QuizArtifact
FlashcardOutput = FlashcardsArtifact
FAQOutput = FAQArtifact
AudioOverviewOutput = AudioOverviewArtifact
ReportOutput = ReportArtifact
DataTableOutput = DataTableArtifact