"""Chat service — sessions, RAG query with memory, persistent messages."""

import json
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from app.core.config import settings
from app.core.ai_clients import get_qdrant_client, get_embeddings, get_llm
from app.core.logger import logger
from app.core.exceptions import NotFoundException

from app.features.notebooks.service import NotebookService
from app.features.documents.repository import DocumentRepository
from app.features.chat.repository import ChatSessionRepository, ChatMessageRepository, MemorySummaryRepository
from app.features.chat.model import ChatSession, ChatMessage, MemorySummary
from app.features.chat.schema import (
    ChatSessionCreate, ChatSessionResponse, ChatSessionListResponse,
    ChatMessageResponse, MessageListResponse, AskResponse, CitationDetail,
    MemoryStatusResponse,
)
from app.features.chat.memory import ConversationMemory, summarise_memory


_RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_template(
    'You are a professional document analyst with memory of this conversation.\n'
    'Use the conversation history to resolve pronouns and follow-up references.\n'
    'Use the document context to answer factually.\n'
    'Do NOT mention file names or page numbers in your answer.\n'
    'If the answer is not in the context, say "I could not find that in the documents."\n'
    '\n'
    '--- Conversation History ---\n'
    '{history}\n'
    '\n'
    '--- Document Context ---\n'
    '{context}\n'
    '\n'
    'Current question: {question}\n'
    '\n'
    'Answer:'
)

_RAG_PROMPT_NO_HISTORY = ChatPromptTemplate.from_template(
    'You are a precise document analyst.\n'
    'Answer the question using ONLY the document context below.\n'
    'Do NOT mention file names or page numbers in your answer.\n'
    'If the answer is not in the context, say "I could not find that in the documents."\n'
    '\n'
    '--- Document Context ---\n'
    '{context}\n'
    '\n'
    'Question: {question}\n'
    '\n'
    'Answer:'
)


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.session_repo = ChatSessionRepository(db)
        self.message_repo = ChatMessageRepository(db)
        self.summary_repo = MemorySummaryRepository(db)
        self.notebook_service = NotebookService(db)
        self.doc_repo = DocumentRepository(db)

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

    async def send_message(self, session_id: uuid.UUID, question: str, user_id: uuid.UUID) -> AskResponse:
        """Full RAG pipeline with conversational memory."""
        session = await self.session_repo.get_by_id(session_id, user_id)
        if not session:
            raise NotFoundException(f"Chat session {session_id} not found")

        notebook_id = session.notebook_id

        # Get doc_ids for this notebook
        doc_ids = await self.doc_repo.get_doc_ids_for_notebook(notebook_id, user_id)

        # Reconstruct memory from DB
        latest_summary = await self.summary_repo.get_latest(session_id, user_id)
        after_msg_id = latest_summary.summarised_up_to_message_id if latest_summary else None
        active_msgs = await self.message_repo.get_active_messages(
            session_id, user_id, after_message_id=after_msg_id
        )
        memory = ConversationMemory.from_db_messages(
            active_msgs,
            latest_summary.summary_text if latest_summary else None,
        )

        # Auto-summarise if needed
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

        # Semantic search in Qdrant
        citations: list[CitationDetail] = []
        context_parts: list[str] = []

        if doc_ids:
            try:
                embeddings = get_embeddings()
                client = get_qdrant_client()
                collection = settings.QDRANT_COLLECTION

                query_vector = embeddings.embed_query(question)
                # FIXED: user_id as string for Qdrant filter
                search_filter = Filter(must=[
                    FieldCondition(key='user_id', match=MatchValue(value=str(user_id))),
                    FieldCondition(key='doc_id', match=MatchAny(any=doc_ids)),
                ])

                results = client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=settings.RETRIEVER_K,
                    with_payload=True,
                    with_vectors=False,
                )

                for hit in results.points:
                    p = hit.payload
                    citations.append(CitationDetail(
                        file_name=p['file_name'],
                        page_number=p['page_number'],
                        chunk_index=p['chunk_index'],
                        similarity_score=round(hit.score, 4),
                        doc_id=p['doc_id'],
                        is_table=p.get('is_table', False),
                        chunk_text=p['chunk_text'],
                    ))
                    context_parts.append(
                        f"[{p['file_name']} | page {p['page_number']}]\n{p['chunk_text']}"
                    )
            except Exception as e:
                logger.error(f"Qdrant search error: {e}", exc_info=True)

        # Build prompt and call LLM
        context = '\n\n---\n\n'.join(context_parts) if context_parts else "No documents available."
        history_block = memory.build_history_block()
        used_memory = bool(history_block)

        if used_memory:
            chain = _RAG_PROMPT_WITH_HISTORY | llm | StrOutputParser()
            answer = chain.invoke({
                'history': history_block,
                'context': context,
                'question': question,
            })
        else:
            chain = _RAG_PROMPT_NO_HISTORY | llm | StrOutputParser()
            answer = chain.invoke({'context': context, 'question': question})

        # Save messages
        human_msg = await self.message_repo.create(ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role="human",
            content=question,
            used_memory=False,
        ))

        citations_json = json.dumps([c.model_dump() for c in citations]) if citations else None
        assistant_msg = await self.message_repo.create(ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=answer,
            citations_json=citations_json,
            used_memory=used_memory,
        ))

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
        return {"message": "Memory cleared. Messages preserved."}

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
    def _format_message(msg: ChatMessage) -> ChatMessageResponse:
        citations = []
        if msg.citations_json:
            try:
                citations = [CitationDetail(**c) for c in json.loads(msg.citations_json)]
            except (json.JSONDecodeError, TypeError):
                pass

        return ChatMessageResponse(
            id=msg.id,
            session_id=msg.session_id,
            role=msg.role,
            content=msg.content,
            citations=citations,
            used_memory=msg.used_memory,
            created_at=msg.created_at,
        )