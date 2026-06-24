# artifact_prompt_builder.py
"""Prompt builder for evidence compression and artifact generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


class ArtifactPromptBuilder:
    """
    Builds prompts for:
    1) evidence compression
    2) final artifact generation
    """

    # =========================================================================
    # Compression prompt
    # =========================================================================

    @staticmethod
    def build_evidence_compression_prompt(
        artifact_type: str,
        context_text: str,
        topic: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """
        Build a prompt to compress retrieved context into a compact evidence pack.
        
        Args:
            artifact_type: Type of artifact being generated
            context_text: The retrieved context text
            topic: Optional user-provided topic
            prompt: Optional user-provided prompt/instructions
            
        Returns:
            A formatted prompt string for the LLM
        """
        user_focus = []
        if topic:
            user_focus.append(f"Topic: {topic}")
        if prompt:
            user_focus.append(f"User request: {prompt}")

        focus_text = "\n".join(user_focus).strip()
        if not focus_text:
            focus_text = "No extra topic/prompt provided."

        return f"""
You are compressing grounded study material for later artifact generation.

Your task:
- Read the source context.
- Extract only the most important grounded information.
- Keep it compact and factual.
- Do not invent anything not supported by the source context.
- Prefer information useful for generating a high-quality {artifact_type} artifact.

User focus:
{focus_text}

Return STRICT JSON only with this shape:
{{
  "title": "Evidence Pack",
  "topic": "string or null",
  "facts": [
    {{
      "fact": "grounded fact",
      "source_label": "optional source label or null",
      "importance": "high|medium|low"
    }}
  ],
  "concepts": ["list of important concepts"],
  "formulas": ["list of formulas if present"],
  "definitions": ["list of grounded definitions"],
  "procedures": ["list of important procedures or steps"],
  "examples": ["list of concrete examples"]
}}

Compression rules:
- facts: 8 to 20 items max
- concepts: short list of important concepts
- formulas: only if present in source
- definitions: only grounded definitions from source
- procedures: important steps/workflows if present
- examples: short concrete examples if present
- remove filler, repetition, and conversational noise
- if source is transcript-like, keep the technical content and discard chatter

Source context:
{context_text}
""".strip()

    # =========================================================================
    # Final generation prompt
    # =========================================================================

    @staticmethod
    def build_generation_prompt(
        artifact_type: str,
        evidence_pack_json: str,
        topic: str | None = None,
        prompt: str | None = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the final generation prompt for the specific artifact type.
        
        Args:
            artifact_type: Type of artifact to generate
            evidence_pack_json: JSON string of the compressed evidence pack
            topic: Optional user-provided topic
            prompt: Optional user-provided prompt/instructions
            options: Additional options specific to the artifact type
            
        Returns:
            A formatted prompt string for the LLM
        """
        options = options or {}

        # Route to the appropriate artifact-specific prompt builder
        prompt_builders = {
            "quiz": ArtifactPromptBuilder._build_quiz_prompt,
            "flashcards": ArtifactPromptBuilder._build_flashcards_prompt,
            "faq": ArtifactPromptBuilder._build_faq_prompt,
            "study_guide": ArtifactPromptBuilder._build_study_guide_prompt,
            "summary": ArtifactPromptBuilder._build_summary_prompt,
            "mindmap": ArtifactPromptBuilder._build_mindmap_prompt,
            "slide_deck": ArtifactPromptBuilder._build_slide_deck_prompt,
        }

        builder = prompt_builders.get(artifact_type)
        if builder:
            return builder(
                evidence_pack_json=evidence_pack_json,
                topic=topic,
                prompt=prompt,
                options=options,
            )

        # Default to summary if artifact type not found
        return ArtifactPromptBuilder._build_summary_prompt(
            evidence_pack_json=evidence_pack_json,
            topic=topic,
            prompt=prompt,
            options=options,
        )

    # =========================================================================
    # Artifact-specific prompt builders
    # =========================================================================

    @staticmethod
    def _build_quiz_prompt(
        evidence_pack_json: str,
        topic: str | None,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for quiz generation."""
        question_count = int(options.get("question_count", 10))
        difficulty = options.get("difficulty", "mixed")

        return f"""
You are generating a grounded quiz from an evidence pack.

Goal:
- Create a useful, non-trivial quiz.
- Use ONLY information from the evidence pack.
- Avoid duplicate questions.
- Cover multiple concepts if possible.
- Prefer concept understanding over superficial trivia.

User topic: {topic or "N/A"}
User request: {prompt or "N/A"}
Requested question count: {question_count}
Difficulty: {difficulty}

Return STRICT JSON only in this shape:
{{
  "title": "Quiz title",
  "description": "Short description or null",
  "questions": [
    {{
      "question": "Question text",
      "type": "mcq",
      "options": [
        {{"id": "A", "text": "option 1"}},
        {{"id": "B", "text": "option 2"}},
        {{"id": "C", "text": "option 3"}},
        {{"id": "D", "text": "option 4"}}
      ],
      "answer": "A",
      "explanation": "Grounded explanation"
    }}
  ]
}}

Rules:
- Generate exactly {question_count} questions
- Prefer MCQs
- 4 options per MCQ
- Only one correct answer
- Explanation must be grounded in the evidence pack
- No trick questions
- Avoid asking the same concept twice unless necessary

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_flashcards_prompt(
        evidence_pack_json: str,
        topic: str | None,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for flashcard generation."""
        card_count = int(options.get("card_count", 12))

        return f"""
You are generating grounded flashcards from an evidence pack.

Goal:
- Create clear, useful flashcards for learning/revision.
- Use ONLY the evidence pack.
- Prefer important terms, concepts, definitions, distinctions, procedures, and facts.
- Keep each card focused on one concept.

User topic: {topic or "N/A"}
User request: {prompt or "N/A"}
Requested card count: {card_count}

Return STRICT JSON only in this shape:
{{
  "title": "Flashcards title",
  "description": "Short description or null",
  "cards": [
    {{
      "front": "Question / term / prompt",
      "back": "Answer / definition / explanation",
      "hint": "optional hint or null"
    }}
  ]
}}

Rules:
- Generate exactly {card_count} cards
- Avoid vague fronts like "Explain this"
- Back should be concise but useful
- Prefer one concept per card
- Avoid duplicate cards

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_faq_prompt(
        evidence_pack_json: str,
        topic: str | None,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for FAQ generation."""
        faq_count = int(options.get("faq_count", 10))

        return f"""
You are generating a grounded FAQ from an evidence pack.

Goal:
- Create realistic and useful FAQ questions and answers.
- Use ONLY the evidence pack.
- Prefer questions a learner or reader would naturally ask.

User topic: {topic or "N/A"}
User request: {prompt or "N/A"}
Requested FAQ count: {faq_count}

Return STRICT JSON only in this shape:
{{
  "title": "FAQ title",
  "description": "Short description or null",
  "items": [
    {{
      "question": "FAQ question",
      "answer": "Grounded answer"
    }}
  ]
}}

Rules:
- Generate exactly {faq_count} FAQ items
- Questions should be natural and useful
- Answers should be concise but complete
- Avoid near-duplicate questions

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_study_guide_prompt(
        evidence_pack_json: str,
        topic: str | None,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for study guide generation."""
        size = options.get("size", "medium")
        
        # Adjust section count based on size
        section_guidance = {
            "short": "Generate 3-4 sections",
            "medium": "Generate 5-7 sections",
            "large": "Generate 8-10 sections",
        }.get(size, "Generate 5-7 sections")

        return f"""
You are generating a grounded study guide from an evidence pack.

Goal:
- Build a clean study guide that helps a learner revise the material.
- Use ONLY the evidence pack.
- Organize content into meaningful sections.

User topic: {topic or "N/A"}
User request: {prompt or "N/A"}
Study guide size: {size}
{section_guidance}

Return STRICT JSON only in this shape:
{{
  "title": "Study guide title",
  "overview": "Short overview paragraph",
  "sections": [
    {{
      "heading": "Section title",
      "explanation": "Section explanation",
      "key_points": ["key point 1", "key point 2"],
      "important_terms": ["term 1", "term 2"]
    }}
  ],
  "review_questions": ["question 1", "question 2"]
}}

Rules:
- Organize by concepts, not random fragments
- Keep explanations grounded
- Include review questions at the end
- Avoid filler
- {section_guidance}

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_summary_prompt(
        evidence_pack_json: str,
        topic: str | None,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for summary generation."""
        return f"""
You are generating a grounded summary from an evidence pack.

Goal:
- Create a useful structured summary.
- Use ONLY the evidence pack.
- Focus on main ideas, key facts, and major takeaways.

User topic: {topic or "N/A"}
User request: {prompt or "N/A"}

Return STRICT JSON only in this shape:
{{
  "title": "Summary title",
  "overview": "Short overview paragraph",
  "key_points": ["key takeaway 1", "key takeaway 2"],
  "sections": [
    {{
      "heading": "Section heading",
      "bullets": ["point 1", "point 2"]
    }}
  ]
}}

Rules:
- Avoid filler
- Keep the summary grounded and specific
- Prefer clarity over verbosity
- Organize sections logically

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_mindmap_prompt(
        evidence_pack_json: str,
        topic: str | None,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for mind map generation."""
        return f"""
You are generating a grounded mind map structure from an evidence pack.

Goal:
- Build a hierarchical concept tree.
- Use ONLY the evidence pack.
- Organize concepts into a clean parent-child structure.

User topic: {topic or "N/A"}
User request: {prompt or "N/A"}

Return STRICT JSON only in this shape:
{{
  "title": "Mind map title",
  "root": {{
    "label": "Root topic",
    "children": [
      {{
        "label": "Child topic",
        "children": [
          {{"label": "Subtopic", "children": []}}
        ]
      }}
    ]
  }}
}}

Rules:
- Keep the structure concept-oriented
- Avoid too many shallow duplicate nodes
- Prefer 2-4 levels max
- Root should be the main topic or concept

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_slide_deck_prompt(
        evidence_pack_json: str,
        topic: str | None,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for slide deck generation."""
        slide_count = int(options.get("slide_count", 8))

        return f"""
You are generating a grounded slide deck outline from an evidence pack.

Goal:
- Build a clean presentation outline.
- Use ONLY the evidence pack.
- Each slide should have a clear purpose.

User topic: {topic or "N/A"}
User request: {prompt or "N/A"}
Requested slide count: {slide_count}

Return STRICT JSON only in this shape:
{{
  "title": "Slide deck title",
  "slides": [
    {{
      "title": "Slide title",
      "bullets": ["bullet 1", "bullet 2"],
      "speaker_notes": "optional short note or null"
    }}
  ]
}}

Rules:
- Generate around {slide_count} slides
- Avoid dumping everything onto one slide
- Prefer logical narrative flow
- First slide should be introduction/overview
- Last slide should be conclusion/summary

Evidence pack:
{evidence_pack_json}
""".strip()

    # =========================================================================
    # Utility methods
    # =========================================================================

    @staticmethod
    def get_artifact_type_description(artifact_type: str) -> str:
        """Get a human-readable description of an artifact type."""
        descriptions = {
            "quiz": "Multiple-choice quiz with explanations",
            "flashcards": "Flashcards for memorization and review",
            "faq": "Frequently asked questions with answers",
            "study_guide": "Comprehensive study guide with sections and review questions",
            "summary": "Concise summary with key points",
            "mindmap": "Hierarchical concept map",
            "slide_deck": "Presentation slide deck outline",
        }
        return descriptions.get(artifact_type, "Learning artifact")

    @staticmethod
    def get_artifact_type_instructions(artifact_type: str) -> str:
        """Get specific instructions for a given artifact type."""
        instructions = {
            "quiz": "Focus on testing understanding, not just memorization. Include a mix of question types where appropriate.",
            "flashcards": "Focus on key terms, definitions, and important concepts that are worth memorizing.",
            "faq": "Focus on questions that address common confusions or important clarifications.",
            "study_guide": "Organize content logically with clear headings and comprehensive coverage.",
            "summary": "Focus on the most important ideas and takeaways. Be concise but complete.",
            "mindmap": "Create a hierarchical structure that shows relationships between concepts.",
            "slide_deck": "Create a logical flow that tells a story. Each slide should have a clear purpose.",
        }
        return instructions.get(artifact_type, "Generate a high-quality learning artifact.")

    @staticmethod
    def build_simple_prompt(
        artifact_type: str,
        context: str,
        topic: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """
        Build a simple prompt when evidence compression is not used.
        
        This is a fallback for when compression fails or is skipped.
        """
        artifact_description = ArtifactPromptBuilder.get_artifact_type_description(artifact_type)
        instructions = ArtifactPromptBuilder.get_artifact_type_instructions(artifact_type)

        user_focus = []
        if topic:
            user_focus.append(f"Topic: {topic}")
        if prompt:
            user_focus.append(f"User request: {prompt}")

        focus_text = "\n".join(user_focus).strip()
        if not focus_text:
            focus_text = "No specific focus provided."

        return f"""
You are generating a {artifact_description} from the source material below.

Goal:
- {instructions}
- Use ONLY the source material provided.
- Do not invent facts or details.
- Stay grounded in the source content.

User focus:
{focus_text}

Source material:
{context}

Generate the {artifact_type} now. Return as structured JSON.
""".strip()