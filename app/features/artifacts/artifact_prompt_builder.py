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
        prompt: str | None = None,
    ) -> str:
        """
        Build a prompt to compress retrieved context into a compact evidence pack.
        
        Args:
            artifact_type: Type of artifact being generated
            context_text: The retrieved context text
            prompt: Optional user-provided prompt/instructions
            
        Returns:
            A formatted prompt string for the LLM
        """
        focus_text = f"User request: {prompt.strip()}" if prompt and prompt.strip() else "No extra prompt provided."

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
        prompt: str | None = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the final generation prompt for the specific artifact type.
        
        Args:
            artifact_type: Type of artifact to generate
            evidence_pack_json: JSON string of the compressed evidence pack
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
                prompt=prompt,
                options=options,
            )

        # Default to summary if artifact type not found
        return ArtifactPromptBuilder._build_summary_prompt(
            evidence_pack_json=evidence_pack_json,
            prompt=prompt,
            options=options,
        )

    # =========================================================================
    # Artifact-specific prompt builders
    # =========================================================================

    @staticmethod
    def _build_quiz_prompt(
        evidence_pack_json: str,
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

User request: {prompt or "N/A"}
Requested question count: {question_count}
Difficulty: {difficulty}

Rules:
- Generate exactly {question_count} questions.
- Prefer multiple choice questions (MCQs) with 4 unique options (A, B, C, D).
- Only one correct answer per question.
- The "answer" field must map directly to the correct option identifier (e.g., "A", "B", "C", or "D").
- Explanations must be detailed and grounded entirely in the provided evidence pack.
- No trick questions.
- Avoid asking about the same concept twice unless necessary for different difficulty levels.

Evidence pack:
{evidence_pack_json}
""".strip()


    @staticmethod
    def _build_flashcards_prompt(
        evidence_pack_json: str,
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

User request: {prompt or "N/A"}
Requested card count: {card_count}

Rules:
- Generate exactly {card_count} cards.
- Avoid vague fronts like "Explain this"; the front should be a clear question, term, or prompt.
- The back should be concise but provide a useful answer, definition, or explanation.
- Prefer one concept per card.
- Avoid duplicate cards.

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_faq_prompt(
        evidence_pack_json: str,
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

User request: {prompt or "N/A"}
Requested FAQ count: {faq_count}

Rules:
- Generate exactly {faq_count} FAQ items.
- Questions should be natural, useful, and reflect common points of confusion or interest.
- Answers must be concise but complete, and grounded entirely in the evidence pack.
- Avoid near-duplicate questions.

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_study_guide_prompt(
        evidence_pack_json: str,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for study guide generation."""
        size = options.get("size", "medium")
        
        # Adjust section count based on size
        section_guidance = {
            "short": "Generate 3-4 distinct sections",
            "medium": "Generate 5-7 distinct sections",
            "large": "Generate 8-10 distinct sections",
        }.get(size, "Generate 5-7 distinct sections")

        return f"""
You are generating a grounded study guide from an evidence pack.

Goal:
- Build a clean study guide that helps a learner revise the material comprehensively.
- Use ONLY the evidence pack.
- Organize content into meaningful, concept-based sections.

User request: {prompt or "N/A"}
Study guide size: {size}

Rules:
- {section_guidance}.
- Organize the guide by core concepts, not just random textual fragments.
- Provide a brief, high-level overview of the entire study guide content.
- For each section, provide a clear heading, a detailed explanation, a list of key points, and a list of important terms/definitions.
- At the end of the guide, generate a list of review questions to test the learner's understanding of the material.
- Keep all explanations strictly grounded in the provided text.
- Avoid filler words and conversational noise.

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_summary_prompt(
        evidence_pack_json: str,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for summary generation."""
        return f"""
You are generating a grounded summary from an evidence pack.

Goal:
- Create a highly structured, useful summary of the provided text.
- Use ONLY the evidence pack.
- Focus on main ideas, key facts, and major takeaways.

User request: {prompt or "N/A"}

Rules:
- Provide a concise overview paragraph that captures the core essence of the text.
- Extract the absolute most important "key points" as top-level takeaways.
- Break the rest of the summary down into logical sections.
- Each section must have a descriptive heading and summarize its content using clear bullet points.
- Keep the summary strictly grounded, specific, and factual.
- Prefer clarity and density of information over verbosity; avoid filler.

Evidence pack:
{evidence_pack_json}
""".strip()


    @staticmethod
    def _build_mindmap_prompt(
        evidence_pack_json: str,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for mind map generation."""
        return f"""
You are generating a grounded mind map structure from an evidence pack.

Goal:
- Build a hierarchical concept tree.
- Use ONLY the evidence pack.
- Organize concepts into a clean, logical parent-child structure.

User request: {prompt or "N/A"}

Rules:
- The mind map must have exactly one central "root" node representing the primary topic.
- Branch out from the root node using nested child nodes.
- Keep all node labels concise and punchy (prefer 1-5 words).
- Limit the depth to 2 to 4 levels maximum (e.g., Root -> Child -> Subtopic -> Detail).
- Group related concepts logically under their appropriate parent nodes.
- Avoid creating too many shallow, redundant nodes that don't add structural value.
- Stay strictly grounded in the provided text.

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_slide_deck_prompt(
        evidence_pack_json: str,
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
        prompt: str | None = None,
    ) -> str:
        """
        Build a simple prompt when evidence compression is not used.
        
        This is a fallback for when compression fails or is skipped.
        """
        artifact_description = ArtifactPromptBuilder.get_artifact_type_description(artifact_type)
        instructions = ArtifactPromptBuilder.get_artifact_type_instructions(artifact_type)

        focus_text = f"User request: {prompt.strip()}" if prompt and prompt.strip() else "No specific focus provided."

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