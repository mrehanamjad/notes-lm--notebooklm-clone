from dataclasses import dataclass, field
from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.core.logger import logger


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    """Manages a sliding window of recent turns + a rolling summary of older ones."""
    window_size: int = field(default_factory=lambda: settings.MEMORY_WINDOW)
    summarise_after: int = field(default_factory=lambda: settings.SUMMARISE_AFTER)
    turns: List[Turn] = field(default_factory=list)
    rolling_summary: str = ""

    # ── Construct from DB messages ──────────────────────────────────────────────
    @classmethod
    def from_db_messages(
        cls,
        messages: list,
        latest_summary_text: Optional[str] = None,
    ) -> "ConversationMemory":
        memory = cls()
        if latest_summary_text:
            memory.rolling_summary = latest_summary_text

        i = 0
        while i < len(messages):
            if messages[i].role == "human":
                # Look for the immediate next assistant message
                if i + 1 < len(messages) and messages[i + 1].role == "assistant":
                    memory.turns.append(Turn(
                        question=messages[i].content,
                        answer=messages[i + 1].content,
                    ))
                    i += 2 # Skip the matched assistant message
                else:
                    # Dangling human message (no assistant reply followed)
                    i += 1 
            else:
                # Dangling assistant message (e.g., first message in array is an orphan)
                i += 1

        return memory

    # ── Core API ────────────────────────────────────────────────────────────────
    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))

    def should_summarise(self) -> bool:
        return len(self.turns) > self.summarise_after

    def get_recent_turns(self) -> List[Turn]:
        """Return the most recent turns within the window."""
        return self.turns[-self.window_size:]

    def get_turns_to_summarise(self) -> List[Turn]:
        """Return older turns that should be compressed into summary."""
        if len(self.turns) <= self.window_size:
            return []
        return self.turns[:-self.window_size]

    def build_history_block(self) -> str:
        """Build the history block for prompt injection."""
        parts: List[str] = []

        # Include rolling summary if it exists
        if self.rolling_summary:
            parts.append(f"[Summary of earlier conversation]\n{self.rolling_summary}")

        # Include recent turns verbatim
        for turn in self.get_recent_turns():
            parts.append(f"User: {turn.question}\nAssistant: {turn.answer}")

        return "\n\n".join(parts)


# ── Summarisation ──────────────────────────────────────────────────────────────

_SUMMARISE_PROMPT = ChatPromptTemplate.from_template(
    "Below is a conversation history between a user and an AI assistant.\n"
    "Write a concise summary capturing the main topics discussed and conclusions reached.\n"
    "If a previous summary is provided, integrate the new conversation into it to maintain a continuous, updated summary.\n"
    "\n"
    "{previous_summary}"
    "New Conversation:\n{conversation}\n\n"
    "Summary:"
)


def summarise_memory(memory: ConversationMemory, llm: BaseChatModel) -> str:
    """Compress older turns into a rolling summary using the LLM.

    Returns the new summary text (caller is responsible for persisting it).
    """
    turns_to_compress = memory.get_turns_to_summarise()
    if not turns_to_compress:
        return memory.rolling_summary

    conversation_text = "\n".join(
        f"User: {t.question}\nAssistant: {t.answer}" for t in turns_to_compress
    )

    previous_block = ""
    if memory.rolling_summary:
        previous_block = f"Previous summary:\n{memory.rolling_summary}\n\n"

    chain = _SUMMARISE_PROMPT | llm | StrOutputParser()
    new_summary = chain.invoke({
        "previous_summary": previous_block,
        "conversation": conversation_text,
    })

    # Update in-memory state
    memory.rolling_summary = new_summary
    # Keep only the recent window
    memory.turns = memory.get_recent_turns()

    logger.info(f"Memory summarised: compressed {len(turns_to_compress)} older turns")
    return new_summary
