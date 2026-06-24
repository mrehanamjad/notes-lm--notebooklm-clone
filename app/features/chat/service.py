"""Chat service — sessions, DB management, persistent messages."""

from app.core.exceptions import InternalServerError
import json
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ai_clients import get_llm
from app.core.logger import logger
from app.core.exceptions import NotFoundException

from app.features.notebooks.service import NotebookService
from app.features.sources.repository import SourceRepository
from app.features.chat.repository import ChatSessionRepository, ChatMessageRepository, MemorySummaryRepository
from app.features.chat.model import ChatSession, ChatMessage, MemorySummary
from app.features.chat.schema import (
    ChatSessionCreate, ChatSessionResponse, ChatSessionListResponse,
    HumanMessageResponse, AssistantMessageResponse, MessageListResponse, AskResponse, CitationDetail,
    MemoryStatusResponse,
)
from app.features.chat.memory import ConversationMemory, summarise_memory

# Import the extracted AI Engine
from app.features.chat.ai import RAGEngine


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.session_repo = ChatSessionRepository(db)
        self.message_repo = ChatMessageRepository(db)
        self.summary_repo = MemorySummaryRepository(db)
        self.notebook_service = NotebookService(db)
        self.source_repo = SourceRepository(db)

    # ── Session management ─────────────────────────────────────────────────────

    async def create_session(self, data: ChatSessionCreate, user_id: uuid.UUID) -> ChatSession:
        await self.notebook_service.get_notebook(data.notebook_id, user_id)
        session = ChatSession(
            notebook_id=data.notebook_id,
            user_id=user_id,
            title=data.title or "New Chat",
        )
        created = await self.session_repo.create(session)
        logger.info(f"Chat session created: id={created.id}, notebook={data.notebook_id}")
        return created

    async def list_sessions(
        self, notebook_id: uuid.UUID, user_id: uuid.UUID, page: int = 1, size: int = 20
    ) -> ChatSessionListResponse:
        await self.notebook_service.get_notebook(notebook_id, user_id)
        skip = (page - 1) * size
        sessions = await self.session_repo.list_by_notebook(notebook_id, user_id, skip=skip, limit=size)
        total = await self.session_repo.count_by_notebook(notebook_id, user_id)
        return ChatSessionListResponse(
            sessions=[ChatSessionResponse.model_validate(s) for s in sessions],
            total=total, page=page, size=size,
            has_more=(skip + size) < total,
        )

    async def delete_session(self, session_id: uuid.UUID, user_id: uuid.UUID) -> None:
        session = await self.session_repo.get_by_id(session_id, user_id)
        if not session:
            raise NotFoundException(f"Chat session {session_id} not found")
        await self.session_repo.delete(session)
        logger.info(f"Chat session deleted: id={session_id}")

    # ── Messages ───────────────────────────────────────────────────────────────

    async def get_messages(
        self, session_id: uuid.UUID, user_id: uuid.UUID, page: int = 1, size: int = 50
    ) -> MessageListResponse:
        session = await self.session_repo.get_by_id(session_id, user_id)
        if not session:
            raise NotFoundException(f"Chat session {session_id} not found")

        skip = (page - 1) * size
        messages = await self.message_repo.get_messages(session_id, user_id, skip=skip, limit=size)
        total = await self.message_repo.count_messages(session_id, user_id)

        return MessageListResponse(
            messages=[self._format_message(m) for m in messages],
            total=total, page=page, size=size,
            has_more=(skip + size) < total,
        )

    # ── RAG Query ─────────────────────────────────────────────────────────────

    async def send_message(self, session_id: uuid.UUID, user_id: uuid.UUID, question: str, excluded_source_ids: list[str]) -> AskResponse:
        """Full RAG pipeline with conversational memory orchestrating AI logic."""
        session = await self.session_repo.get_by_id(session_id, user_id)
        if not session:
            raise NotFoundException(f"Chat session {session_id} not found")

        notebook_id = session.notebook_id
        all_source_ids = await self.source_repo.get_source_ids_for_notebook(notebook_id, user_id)

        if not all_source_ids:
            raise NotFoundException("No source available for the notebook")

        excluded_set = set(excluded_source_ids)
        source_ids = [sid for sid in all_source_ids if sid not in excluded_set]

        if not source_ids:
           raise NotFoundException("No source selected for the query")

        # 1. Reconstruct Memory
        latest_summary = await self.summary_repo.get_latest(session_id, user_id)
        after_msg_id = latest_summary.summarised_up_to_message_id if latest_summary else None
        active_msgs = await self.message_repo.get_active_messages(
            session_id, user_id, after_message_id=after_msg_id
        )
        memory = ConversationMemory.from_db_messages(
            active_msgs,
            latest_summary.summary_text if latest_summary else None,
        )

        # 2. Auto-summarise if needed
        llm = get_llm()
        if memory.should_summarise():
            new_summary_text = summarise_memory(memory, llm)
            turns_to_compress = len(memory.turns) - memory.window_size
            last_msg_idx = (turns_to_compress * 2) - 1
            if last_msg_idx >= 0 and last_msg_idx < len(active_msgs):
                new_summarised_up_to_id = active_msgs[last_msg_idx].id
            else:
                new_summarised_up_to_id = active_msgs[-1].id if active_msgs else None

            await self.summary_repo.create(MemorySummary(
                session_id=session_id,
                user_id=user_id,
                summary_text=new_summary_text,
                summarised_up_to_message_id=new_summarised_up_to_id,
            ))

        # 3. AI Execution (Delegated to RAGEngine)
        history_block = memory.build_history_block()
        context, citations = RAGEngine.retrieve_context(question, user_id, source_ids)
        answer, used_memory = RAGEngine.generate_answer(question, context, history_block)

        # 4. Save to Database
        human_msg = ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role="human",
            content=question,
            used_memory=False,
        )

        citations_json = json.dumps([c.model_dump() for c in citations]) if citations else None
        assistant_msg = ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=answer,
            citations_json=citations_json,
            used_memory=used_memory,
        )
        try:
            human_msg, assistant_msg = await self.message_repo.save_message_turn(human_msg, assistant_msg)
        except Exception as e:
            logger.error(f"Failed to save chat messages: {e}")
            raise InternalServerError(
                message="Failed to save conversation state due to database transaction failure.",
                details={"reason": str(e)}
            )

        return AskResponse(
            human_message=self._format_message(human_msg),
            assistant_message=self._format_message(assistant_msg),
        )

    # ── Memory management ─────────────────────────────────────────────────────

    async def clear_memory(self, session_id: uuid.UUID, user_id: uuid.UUID) -> dict:
        session = await self.session_repo.get_by_id(session_id, user_id)
        if not session:
            raise NotFoundException(f"Chat session {session_id} not found")
        await self.summary_repo.delete_by_session(session_id, user_id)
        return

    async def get_memory_status(self, session_id: uuid.UUID, user_id: uuid.UUID) -> MemoryStatusResponse:
        session = await self.session_repo.get_by_id(session_id, user_id)
        if not session:
            raise NotFoundException(f"Chat session {session_id} not found")

        total = await self.message_repo.count_messages(session_id, user_id)
        latest_summary = await self.summary_repo.get_latest(session_id, user_id)

        return MemoryStatusResponse(
            session_id=session_id,
            total_messages=total,
            has_summary=latest_summary is not None,
            summary_preview=latest_summary.summary_text[:200] if latest_summary else None,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_message(msg: ChatMessage) -> HumanMessageResponse | AssistantMessageResponse:
        if msg.role == "human":
            return HumanMessageResponse(
                id=msg.id,
                session_id=msg.session_id,
                role="human",
                content=msg.content,
                created_at=msg.created_at,
            )
            
        citations = []
        if msg.citations_json:
            try:
                citations = [CitationDetail(**c) for c in json.loads(msg.citations_json)]
            except (json.JSONDecodeError, TypeError):
                pass

        return AssistantMessageResponse(
            id=msg.id,
            session_id=msg.session_id,
            role="assistant",
            content=msg.content,
            citations=citations,
            used_memory=msg.used_memory,
            created_at=msg.created_at,
        )