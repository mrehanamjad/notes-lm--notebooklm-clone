# artifact_evidence_compressor.py
"""Evidence compressor for reducing context size while preserving key information."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from pydantic import ValidationError

from app.core.logger import logger
from app.features.artifacts.artifact_models import EvidencePack
from app.features.artifacts.artifact_prompt_builder import ArtifactPromptBuilder


class ArtifactEvidenceCompressor:
    """
    Compress retrieved source context into a smaller evidence pack.
    This is the main token-saving step for Groq free tier.
    
    The compressor takes raw retrieved context and extracts:
    - Key facts (with importance ratings)
    - Important concepts
    - Definitions
    - Formulas
    - Procedures
    - Examples
    
    This reduces token usage while preserving the most important information
    for artifact generation.
    """

    def __init__(self, llm_client):
        """
        Initialize the compressor with an LLM client.
        
        Args:
            llm_client: Must expose a method like:
                llm_client.generate_text(prompt: str, temperature: float, max_tokens: int) -> str
                or
                llm_client.invoke(prompt: str) -> AIMessage
        """
        self.llm_client = llm_client

    async def compress(
        self,
        artifact_type: str,
        context_text: str,
        topic: str | None = None,
        prompt: str | None = None,
    ) -> EvidencePack:
        """
        Compress context into an evidence pack.
        
        Args:
            artifact_type: Type of artifact being generated
            context_text: The retrieved context text
            topic: Optional user-provided topic
            prompt: Optional user-provided prompt/instructions
            
        Returns:
            EvidencePack: Structured evidence pack with extracted information
            
        Raises:
            Exception: If compression fails, falls back to a simple evidence pack
        """
        if not context_text or context_text == "No documents available.":
            logger.warning("Empty context provided to compressor, returning empty evidence pack")
            return self._fallback_evidence_pack(context_text="No content available.", topic=topic)

        try:
            # Build the compression prompt
            compression_prompt = ArtifactPromptBuilder.build_evidence_compression_prompt(
                artifact_type=artifact_type,
                context_text=context_text,
                topic=topic,
                prompt=prompt,
            )

            logger.info(f"Compressing context for {artifact_type}, context length: {len(context_text)} chars")

            # Generate compression
            raw = await self._generate_compression(compression_prompt)

            # Parse JSON response
            parsed = self._parse_json_response(raw)
            if parsed is None:
                logger.warning("Evidence compression JSON parse failed, using fallback evidence pack")
                return self._fallback_evidence_pack(context_text=context_text, topic=topic)

            # Validate with Pydantic
            try:
                evidence_pack = EvidencePack.model_validate(parsed)
                logger.info(
                    f"Evidence compression successful: "
                    f"facts={len(evidence_pack.facts)}, "
                    f"concepts={len(evidence_pack.concepts)}, "
                    f"definitions={len(evidence_pack.definitions)}, "
                    f"procedures={len(evidence_pack.procedures)}, "
                    f"examples={len(evidence_pack.examples)}, "
                    f"formulas={len(evidence_pack.formulas)}"
                )
                return evidence_pack
            except ValidationError as e:
                logger.warning(f"EvidencePack validation failed: {e}")
                return self._fallback_evidence_pack(context_text=context_text, topic=topic)

        except Exception as e:
            logger.error(f"Evidence compression error: {e}", exc_info=True)
            return self._fallback_evidence_pack(context_text=context_text, topic=topic)

    async def _generate_compression(self, prompt: str) -> str:
        """
        Generate compression using the LLM client.
        
        Supports both sync and async clients with different interfaces.
        """
        try:
            # Try async generation first
            if hasattr(self.llm_client, "agenerate_text"):
                return await self.llm_client.agenerate_text(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=1200,
                )
            elif hasattr(self.llm_client, "ainvoke"):
                # LangChain style
                response = await self.llm_client.ainvoke(prompt)
                return response.content if hasattr(response, "content") else str(response)
            elif hasattr(self.llm_client, "generate_text"):
                # Sync method - run in executor
                import asyncio
                return await asyncio.to_thread(
                    self.llm_client.generate_text,
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=1200,
                )
            elif hasattr(self.llm_client, "invoke"):
                # Sync LangChain style
                response = self.llm_client.invoke(prompt)
                return response.content if hasattr(response, "content") else str(response)
            else:
                # Fallback: assume it's a callable
                result = self.llm_client(prompt)
                if hasattr(result, "content"):
                    return result.content
                return str(result)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback_evidence_pack(
        self, 
        context_text: str, 
        topic: str | None = None
    ) -> EvidencePack:
        """
        If the model returns bad JSON or fails, produce a cheap fallback pack 
        by extracting a few lines from the context.
        """
        if not context_text or context_text == "No documents available.":
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

        lines = [line.strip() for line in context_text.splitlines() if line.strip()]
        facts = []
        seen_facts = set()

        for line in lines:
            # Skip metadata lines
            if line.startswith("#"):
                continue
            if line.startswith("["):
                continue
            if line.startswith("---"):
                continue
            
            # Skip very short lines
            if len(line) < 25:
                continue

            # Skip duplicate or very similar facts
            key = line[:50].lower()
            if key in seen_facts:
                continue
            seen_facts.add(key)

            facts.append({
                "fact": line[:500],  # Truncate to reasonable length
                "source_label": None,
                "importance": "medium",
            })

            if len(facts) >= 12:
                break

        # If we couldn't extract any facts, use the first non-empty line as a fallback
        if not facts:
            for line in lines:
                if len(line) > 20 and not line.startswith("#") and not line.startswith("["):
                    facts.append({
                        "fact": line[:500],
                        "source_label": None,
                        "importance": "medium",
                    })
                    break

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

    # ── JSON parsing helpers ──────────────────────────────────────────────────

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response with multiple fallback strategies.
        
        Supports:
        1. Direct JSON
        2. Markdown fenced code blocks (```json ... ```)
        3. Generic fenced code blocks (``` ... ```)
        4. First JSON object found in text
        """
        text = text.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Strategy 2: Markdown fenced JSON block
        fenced = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            block = fenced.group(1).strip()
            try:
                return json.loads(block)
            except Exception:
                pass

        # Strategy 3: Generic fenced code block
        fenced = re.search(r"```(.*?)```", text, flags=re.DOTALL)
        if fenced:
            block = fenced.group(1).strip()
            try:
                return json.loads(block)
            except Exception:
                pass

        # Strategy 4: Extract first JSON object
        obj = self._extract_first_json_object(text)
        if obj:
            try:
                return json.loads(obj)
            except Exception:
                pass

        # Strategy 5: Try to repair common JSON issues
        try:
            repaired = self._repair_json(text)
            if repaired:
                return json.loads(repaired)
        except Exception:
            pass

        return None

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """
        Extract the first complete JSON object from a string.
        
        Finds the first '{' and then matches the closing '}' with proper nesting.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            char = text[i]

            if escape:
                escape = False
                continue

            if char == '\\':
                escape = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]

        return None

    def _repair_json(self, text: str) -> Optional[str]:
        """
        Attempt to repair common JSON issues.
        
        Handles:
        - Missing quotes around keys
        - Trailing commas
        - Single quotes instead of double quotes
        """
        # Remove trailing commas before closing braces/brackets
        repaired = re.sub(r',\s*}', '}', text)
        repaired = re.sub(r',\s*]', ']', repaired)

        # Replace single quotes with double quotes (careful not to break strings)
        repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)

        # Add quotes around unquoted keys
        repaired = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)

        return repaired

    # ── Utility methods ───────────────────────────────────────────────────────

    def get_compression_stats(self, evidence_pack: EvidencePack) -> Dict[str, int]:
        """Get statistics about the compressed evidence pack."""
        return {
            "total_facts": len(evidence_pack.facts),
            "total_concepts": len(evidence_pack.concepts),
            "total_definitions": len(evidence_pack.definitions),
            "total_formulas": len(evidence_pack.formulas),
            "total_procedures": len(evidence_pack.procedures),
            "total_examples": len(evidence_pack.examples),
            "high_importance_facts": sum(
                1 for f in evidence_pack.facts if f.importance == "high"
            ),
            "medium_importance_facts": sum(
                1 for f in evidence_pack.facts if f.importance == "medium"
            ),
            "low_importance_facts": sum(
                1 for f in evidence_pack.facts if f.importance == "low"
            ),
        }

    def estimate_token_savings(
        self, 
        original_text: str, 
        evidence_pack: EvidencePack
    ) -> Dict[str, int]:
        """
        Estimate token savings from compression.
        
        Uses a rough estimate of 4 chars per token for plain text.
        """
        original_tokens = len(original_text) // 4
        
        # Serialize the evidence pack to JSON
        evidence_json = evidence_pack.model_dump_json()
        compressed_tokens = len(evidence_json) // 4
        
        return {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": original_tokens - compressed_tokens,
            "savings_percentage": int(
                ((original_tokens - compressed_tokens) / max(original_tokens, 1)) * 100
            ),
        }

    def to_readable_format(self, evidence_pack: EvidencePack) -> str:
        """
        Convert an evidence pack to a human-readable format.
        Useful for debugging or logging.
        """
        lines = []
        lines.append(f"# {evidence_pack.title}")
        if evidence_pack.topic:
            lines.append(f"Topic: {evidence_pack.topic}")
        lines.append("")

        if evidence_pack.concepts:
            lines.append("## Concepts")
            for concept in evidence_pack.concepts:
                lines.append(f"• {concept}")
            lines.append("")

        if evidence_pack.definitions:
            lines.append("## Definitions")
            for definition in evidence_pack.definitions:
                lines.append(f"• {definition}")
            lines.append("")

        if evidence_pack.facts:
            lines.append("## Facts")
            for fact in evidence_pack.facts:
                importance_symbol = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(fact.importance, "•")
                source = f" [{fact.source_label}]" if fact.source_label else ""
                lines.append(f"{importance_symbol} {fact.fact}{source}")
            lines.append("")

        if evidence_pack.procedures:
            lines.append("## Procedures")
            for procedure in evidence_pack.procedures:
                lines.append(f"• {procedure}")
            lines.append("")

        if evidence_pack.formulas:
            lines.append("## Formulas")
            for formula in evidence_pack.formulas:
                lines.append(f"• {formula}")
            lines.append("")

        if evidence_pack.examples:
            lines.append("## Examples")
            for example in evidence_pack.examples:
                lines.append(f"• {example}")

        return "\n".join(lines)