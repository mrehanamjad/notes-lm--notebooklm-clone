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
            "voice_overview": ArtifactPromptBuilder._build_voice_overview_prompt,
            "report": ArtifactPromptBuilder._build_report_prompt,
            "datatable": ArtifactPromptBuilder._build_datatable_prompt,
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
    def _build_report_prompt(
        evidence_pack_json: str,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for structured analytical report generation."""
        length = options.get("length", "medium")

        section_guidance = {
            "short": "Generate 2-3 focused sections.",
            "medium": "Generate 4-6 sections.",
            "large": "Generate 7-9 in-depth sections.",
        }.get(length, "Generate 4-6 sections.")

        return f"""
You are generating a grounded analytical report from an evidence pack.

Goal:
- Write a clear, professional report that presents findings from the source material.
- Use ONLY the evidence pack. Do not invent facts, numbers, or claims.
- Structure the report so a reader can quickly grasp the key findings and conclusion.

User request: {prompt or "N/A"}
Report length: {length}

Rules:
- {section_guidance}
- Start with a concise executive summary (2-4 sentences) capturing the report's core findings.
- Each section must have a clear heading and a well-written body paragraph; add bullet points
  only where they genuinely clarify a list of items, steps, or comparisons.
- Extract the most important "key findings" as a separate top-level list of concise statements.
- End with a conclusion paragraph that synthesizes the findings into a clear takeaway.
- Keep all content strictly grounded in the provided evidence pack.
- Use a neutral, professional tone; avoid filler and conversational language.

Evidence pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_datatable_prompt(
        evidence_pack_json: str,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for structured data table generation."""
        max_rows = int(options.get("max_rows", 15))

        return f"""
You are extracting a grounded, structured data table from an evidence pack.

Goal:
- Identify the most relevant structured/comparable data in the evidence pack
  (numbers, named entities, categories, dates, metrics, comparisons, etc.).
- Organize that data into ONE clean table with consistent columns.
- Use ONLY information from the evidence pack. Do not invent values, numbers, or rows.

User request: {prompt or "N/A"}
Maximum rows: {max_rows}

Rules:
- Define between 2 and 6 columns. Each column needs a short, clear "name" and a "type"
  of exactly one of: "string", "number", "date", "boolean".
- Generate at most {max_rows} rows. Fewer rows are fine if the evidence pack doesn't support more.
- Every row must be a list of cell values in the SAME ORDER as the columns, with one
  value per column (use null if a specific cell value is genuinely not available).
- Keep numeric values as actual numbers (not strings) when the column type is "number".
- Give the table a concise, descriptive title and a one-sentence description of what it shows.
- Optionally include a short list of "notes" highlighting any interesting patterns, gaps,
  or insights visible in the data — grounded strictly in the evidence pack.
- If the evidence pack does not contain enough structured data for a meaningful table,
  build the best reasonable table from whatever concrete facts/figures are available.

Evidence pack:
{evidence_pack_json}
""".strip()

#     @staticmethod
#     def _build_slide_deck_prompt(
#         evidence_pack_json: str,
#         prompt: str | None,
#         options: Dict[str, Any],
#     ) -> str:
#         """Build prompt for structured slide deck generation."""

#         deck_length = "short" #options.get("length", "medium")

#         slide_count_guidance = {
#             "short": "Create 5-7 slides.",
#             "medium": "Create 8-10 slides.",
#             "long": "Create 12-16 slides.",
#         }.get(deck_length, "Create 8-12 slides.")

#         return f"""
# You are creating a professional presentation slide deck from the provided evidence pack.

# Your response MUST conform exactly to the provided Pydantic schema.

# Goal:
# - Build a logical, engaging presentation.
# - Use ONLY information contained in the evidence pack.
# - Do not invent facts, numbers, dates, or claims.
# - Focus on clarity rather than completeness.
# - Every slide should communicate ONE main idea.

# User focus:
# {prompt or "No specific focus provided. Create a presentation covering the most important concepts."}

# General Rules:
# - {slide_count_guidance}
# - Give the deck a concise, descriptive title.
# - Arrange slides in a logical storytelling order.
# - Start with an introductory slide.
# - End with a summary or key takeaways slide.
# - Each slide should have a short, meaningful title.
# - Avoid repeating information across slides.
# - Do not include speaker notes.
# - Do not include markdown.
# - Do not include slide numbers.
# - Do not reference the evidence pack or source documents.

# Choose the most appropriate slide layout for each slide.

# Available slide types:

# 1. paragraph
#    Use when explaining a concept.
#    Content:
#    [
#        "<paragraph>"
#    ]

# 2. bullets_points
#    Use for lists, features, advantages, steps, or takeaways.
#    Content:
#    [
#        "bullet 1",
#        "bullet 2",
#        "bullet 3"
#    ]

# 3. 2_paragraph_cards
#    Use when comparing or introducing two related concepts.
#    Content:
#    [
#        {{
#            "title": "...",
#            "text": "..."
#        }},
#        {{
#            "title": "...",
#            "text": "..."
#        }}
#    ]

# 4. 2_bullets_point_cards
#    Use when comparing two categories that each contain bullet points.
#    Content:
#    [
#        {{
#            "title": "...",
#            "bullets": ["...", "..."]
#        }},
#        {{
#            "title": "...",
#            "bullets": ["...", "..."]
#        }}
#    ]

# 5. 3_paragraph_cards
#    Use for three related ideas.
#    Content:
#    [
#        {{
#            "title": "...",
#            "text": "..."
#        }},
#        {{
#            "title": "...",
#            "text": "..."
#        }},
#        {{
#            "title": "...",
#            "text": "..."
#        }}
#    ]

# 6. 3_bullets_point_cards
#    Use when presenting three categories with bullet points.
#    Content:
#    [
#        {{
#            "title": "...",
#            "bullets": ["...", "..."]
#        }},
#        {{
#            "title": "...",
#            "bullets": ["...", "..."]
#        }},
#        {{
#            "title": "...",
#            "bullets": ["...", "..."]
#        }}
#    ]

# 7. table:
#     Use when presenting structured data in rows and columns.
#     Content:
#     [
#         {{
#             header: "...",
#             items: ["...","..."]
#         }}
#         ...
#     ]

# Layout Guidelines:
# - Prefer paragraph slides for explanations.
# - Prefer bullet slides for lists or sequential information or explaining workflow.
# - Prefer card layouts for comparisons, categories, dimensions examples, or grouped ideas.
# - Table can be used for structured data, comapritions, but avoid overloading with too many rows(max: 7) or columns(max: 5).
# - bullet or paragraph slides can be used for key takeaways, conclusions, or summary slides.
# - Keep paragraphs concise (roughly 65-150 words).
# - Keep bullet lists to 3-7 bullets.
# - Keep card titles short.
# - Avoid slides with too much text.

# Evidence Pack:
# {evidence_pack_json}
# """.strip()

    @staticmethod
    def _build_slide_deck_prompt(
        evidence_pack_json: str,
        prompt: str | None,
        options: Dict[str, Any],
) -> str:
        """Build prompt for structured slide deck generation."""

        deck_length = options.get("length", "medium")

        slide_count_guidance = {
            "short": "Create 5-7 slides.",
            "medium": "Create 8-10 slides.",
            "long": "Create 12-16 slides.",
        }.get(deck_length, "Create 8-10 slides.")

        return f"""
You are creating a professional presentation slide deck from the provided evidence pack.

Your response MUST conform exactly to the provided Pydantic schema.

Goal:
- Build a logical, engaging presentation.
- Use ONLY information from the evidence pack.
- Do not invent facts, numbers, or claims.
- Every slide should communicate ONE primary idea.
- Choose the most appropriate slide type for the content.

User focus:
{prompt or "No specific focus provided. Present the most important concepts."}

General Rules:
- {slide_count_guidance}
- Give the deck a concise title and description.
- Organize slides in a logical storytelling order.
- Start with an introduction and end with key takeaways when appropriate.
- Avoid repeating information across slides.
- Do not use markdown, speaker notes, or slide numbers.
- Do not mention the evidence pack or source documents.

Every slide MUST include a visual recommendation.

Visual Types:
- image → cover slides, concept explanations, people, places, historical events, nature, products.
- icon → features, benefits, workflows, categories, cards, summaries, technology concepts.
- none → tables or slides where visuals add little value.

Visual Query Rules:
- For image: generate a descriptive stock-photo search query.
  Example:
    "modern cloud computing datacenter"
    "artificial intelligence illustration"
    "solar panels on rooftop"

- For icon: generate ONE concise concept.
  Examples:
    cloud
    database
    brain
    shield
    users
    chart
    server
    robot

Visual Placement:
- paragraph → right
- bullets → right
- cards → top or right
- cover/introduction → background
- table → none

Available slide types:

1. paragraph
Content:
[
    "<paragraph>"
]

2. bullets_points
Content:
[
    "bullet 1",
    "bullet 2",
    "bullet 3"
]

3. 2_paragraph_cards
Content:
[
    {{
        "title": "...",
        "text": "..."
    }},
    {{
        "title": "...",
        "text": "..."
    }}
]

4. 2_bullets_point_cards
Content:
[
    {{
        "title": "...",
        "bullets": ["...", "..."]
    }},
    {{
        "title": "...",
        "bullets": ["...", "..."]
    }}
]

5. 3_paragraph_cards
Content:
[
    {{
        "title": "...",
        "text": "..."
    }},
    {{
        "title": "...",
        "text": "..."
    }},
    {{
        "title": "...",
        "text": "..."
    }}
]

6. 3_bullets_point_cards
Content:
[
    {{
        "title": "...",
        "bullets": ["...", "..."]
    }},
    {{
        "title": "...",
        "bullets": ["...", "..."]
    }},
    {{
        "title": "...",
        "bullets": ["...", "..."]
    }}
]

7. table
Content:
[
    {{
        "header": "...",
        "items": ["...", "..."]
    }}
]

Layout Guidelines:
- Every slide must have a title
- Paragraphs: 60-140 words.
- Bullet lists: 3-7 bullets.
- Cards: concise title and short text.
- Tables: 2-5 columns and at most 7 rows.
- Prefer cards for grouped ideas or comparisons.
- Prefer tables only for structured comparisons.
- Keep every slide visually balanced and concise.


Evidence Pack:
{evidence_pack_json}
""".strip()

    @staticmethod
    def _build_voice_overview_prompt(
        evidence_pack_json: str,
        prompt: str | None,
        options: Dict[str, Any],
    ) -> str:
        """Build prompt for two-host voice overview (podcast) script generation."""
        length = options.get("length", "medium")
        voice_style = options.get("voice_style", "default")
        host_names = options.get("host_names") or ["Alex", "Jordan"]
        host_1_name, host_2_name = host_names[0], host_names[1]

        length_guidance = {
            "short": "Write roughly 700-1000 words of dialogue in total (around 3-5 minutes spoken).",
            "medium": "Write roughly 1200-1800 words of dialogue in total (around 6-9 minutes spoken).",
            "long": "Write roughly 2000-2800 words of dialogue in total (around 10-15 minutes spoken).",
        }.get(length, "Write roughly 1200-1800 words of dialogue in total (around 6-9 minutes spoken).")

        tone_guidance = {
            "energetic": "Keep the energy upbeat and enthusiastic throughout, with quick back-and-forth exchanges.",
            "calm": "Keep the tone calm, measured, and thoughtful, with longer reflective exchanges.",
            "default": "Keep the tone warm, curious, and conversational, like two friends genuinely interested in the topic.",
        }.get(voice_style, "Keep the tone warm, curious, and conversational, like two friends genuinely interested in the topic.")

        return f"""
You are writing a two-host podcast-style "voice overview" script, in the style of
Google NotebookLM's Audio Overview feature. Two hosts discuss the source material
in a natural, engaging conversation — NOT a dry summary read aloud.

Hosts:
- host_1 ({host_1_name}): drives the conversation, introduces topics, asks questions, frames context.
- host_2 ({host_2_name}): responds, adds insight, builds on points, occasionally pushes back or asks for clarification.

Goal:
- Use ONLY information from the evidence pack. Do not invent facts not supported by it.
- Make it sound like real spoken conversation: contractions, brief reactions ("right",
  "exactly", "that's interesting"), natural transitions — not a script read word-for-word
  from a document.
- {tone_guidance}
- Vary turn length: mix short reactive lines with longer explanatory ones.
- Avoid robotic, list-like dialogue ("Point one is... Point two is..."). Weave the
  information into a flowing discussion instead.

User focus: {prompt or "No specific focus provided, cover the most important and interesting parts of the material."}

Structure:
- Open with a short, inviting intro where host_1 sets up what the episode covers.
- Develop the discussion through the most important concepts, facts, and takeaways
  in the evidence pack, in a logical order.
- Have host_2 occasionally ask clarifying or "so what does that mean" questions to
  keep things grounded and accessible for a listener with no prior context.
- Close with a brief, natural wrap-up/takeaway from one or both hosts. Do not say
  "in conclusion" — end it like a real podcast sign-off.

Rules:
- {length_guidance}
- Every line's "speaker" field must be exactly "host_1" or "host_2".
- Alternate speakers naturally (not strictly one-for-one, but no giant monologue blocks).
- Do not include sound effect cues, music cues, or stage directions — only spoken words.
- Do not use markdown, bullet points, or headers inside dialogue text.
- Do not reference being an AI, a script, or reading from documents — stay in character
  as two podcast hosts discussing a topic they researched.

Evidence pack:
{evidence_pack_json}
""".strip()

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
            "voice_overview": "Two-host podcast-style voice overview discussion",
            "report": "Structured analytical report with findings and conclusion",
            "datatable": "Structured data table extracted from source material",
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
            "voice_overview": "Write a natural two-host spoken conversation, not a summary read aloud.",
            "report": "Write a professional, well-structured report with clear findings and a conclusion.",
            "datatable": "Extract structured, comparable data into a single clean, consistent table.",
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