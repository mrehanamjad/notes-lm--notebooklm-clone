# schema.py
"""Pydantic schemas for artifact requests, responses, and LLM structured output."""

from datetime import datetime
import uuid
from typing import Optional, Any, List
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


# ── Enums ─────────────────────────────────────────────────────────────────────

class ArtifactType(str, Enum):
    QUIZ = "quiz"
    FLASHCARDS = "flashcards"
    FAQ = "faq"
    STUDY_GUIDE = "study_guide"
    SUMMARY = "summary"
    MINDMAP = "mindmap"
    SLIDE_DECK = "slide_deck"


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


# ── Shared Base Request ───────────────────────────────────────────────────────

class BaseArtifactRequest(BaseModel):
    """Common fields shared by all artifact creation requests."""
    topic: Optional[str] = Field(None, max_length=500, description="Optional topic to focus on")
    prompt: Optional[str] = Field(None, max_length=2000, description="Custom scope/instructions for generation")
    excluded_source_ids: List[str] = Field(
        default_factory=list,
        description="Source IDs to exclude from artifact generation context",
    )


# ── Type-Specific Requests ───────────────────────────────────────────────────

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


# ── LLM Structured Output Schemas ────────────────────────────────────────────

# Quiz
class QuizOption(BaseModel):
    id: str = Field(..., description="Option identifier: A, B, C, D")
    text: str = Field(..., description="Option text")


class QuizQuestion(BaseModel):
    question: str = Field(..., description="The quiz question text")
    type: str = Field("mcq", description="Question type: mcq or true_false")
    options: List[QuizOption] = Field(..., description="List of answer options")
    answer: str = Field(..., description="The correct answer (A, B, C, or D)")
    explanation: str = Field(..., description="Brief explanation of why the answer is correct")


class QuizOutput(BaseModel):
    title: str = Field(..., description="A concise title for the quiz")
    description: Optional[str] = Field(None, description="Short description of the quiz")
    questions: List[QuizQuestion] = Field(..., description="List of quiz questions")


# Flashcards
class FlashcardItem(BaseModel):
    front: str = Field(..., description="The question or concept on the front of the card")
    back: str = Field(..., description="The answer or explanation on the back of the card")
    hint: Optional[str] = Field(None, description="Optional hint to help recall the answer")


class FlashcardOutput(BaseModel):
    title: str = Field(..., description="A concise title for the flashcard set")
    description: Optional[str] = Field(None, description="Short description of the flashcard set")
    cards: List[FlashcardItem] = Field(..., description="List of flashcards")


# FAQ
class FAQItem(BaseModel):
    question: str = Field(..., description="The frequently asked question")
    answer: str = Field(..., description="Comprehensive answer to the question")


class FAQOutput(BaseModel):
    title: str = Field(..., description="A concise title for the FAQ set")
    description: Optional[str] = Field(None, description="Short description of the FAQ set")
    items: List[FAQItem] = Field(..., description="List of FAQ entries")


# Summary
class SummarySection(BaseModel):
    heading: str = Field(..., description="Section heading")
    bullets: List[str] = Field(default_factory=list, description="Key points as bullets")


class SummaryOutput(BaseModel):
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


class StudyGuideOutput(BaseModel):
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


class MindMapOutput(BaseModel):
    title: str = Field(..., description="A concise title for the mind map")
    root: MindMapNode = Field(..., description="Root node of the mind map")


# Slide Deck
class SlideItem(BaseModel):
    title: str = Field(..., description="Slide title")
    bullets: List[str] = Field(default_factory=list, description="Bullet points for the slide")
    speaker_notes: Optional[str] = Field(None, description="Optional speaker notes")


class SlideDeckOutput(BaseModel):
    title: str = Field(..., description="A concise title for the slide deck")
    slides: List[SlideItem] = Field(default_factory=list, description="List of slides")


# ── API Response Schemas ─────────────────────────────────────────────────────

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
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ArtifactListResponse(BaseModel):
    """Response schema for listing artifacts with pagination."""
    artifacts: List[ArtifactResponse]
    total: int
    limit: Optional[int] = None
    offset: Optional[int] = None


# ── Internal Schemas (not exposed to frontend) ─────────────────────────────

class SourceFilterInfo(BaseModel):
    """Internal schema for tracking source resolution."""
    excluded_source_ids: List[str] = Field(default_factory=list)
    resolved_source_ids: List[str] = Field(default_factory=list)