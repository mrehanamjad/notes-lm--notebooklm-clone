"""AI RAG Engine — semantic search and LLM generation."""

from app.core.logger import logger
from app.core.exceptions import InternalServerError
import uuid
from typing import Tuple, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from app.core.config import settings
from app.core.ai_clients import get_qdrant_client, get_embeddings, get_llm
from app.features.chat.schema import CitationDetail


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


class RAGEngine:
    """Handles vector search and LLM response generation."""

    @staticmethod
    def retrieve_context(question: str, user_id: uuid.UUID, source_ids: List[str]) -> Tuple[str, List[CitationDetail]]:
        """Queries Qdrant for relevant document chunks."""
        citations: List[CitationDetail] = []
        context_parts: List[str] = []

        if not source_ids:
            return "No documents available.", citations

        try:
            embeddings = get_embeddings()
            client = get_qdrant_client()
            collection = settings.QDRANT_COLLECTION

            query_vector = embeddings.embed_query(question)
            search_filter = Filter(must=[
                FieldCondition(key='user_id', match=MatchValue(value=str(user_id))),
                FieldCondition(key='source_id', match=MatchAny(any=source_ids)),
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
                    source_id=p['source_id'],
                    is_table=p.get('is_table', False),
                    chunk_text=p['chunk_text'],
                ))
                context_parts.append(
                    f"[{p['file_name']} | page {p['page_number']}]\n{p['chunk_text']}"
                )
        except Exception as e:
            logger.error(f"Qdrant search error: {e}", exc_info=True)

        context = '\n\n---\n\n'.join(context_parts) if context_parts else "No documents available."
        return context, citations

    @staticmethod
    def generate_answer(question: str, context: str, history_block: str) -> Tuple[str, bool]:
        """Invokes the LLM with the appropriate prompt chain."""
        llm = get_llm()
        used_memory = bool(history_block)

        try:
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
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}", exc_info=True)
            raise InternalServerError(
                message="The AI engine is currently overloaded or unavailable. Please try again.",
                details={"reason": str(e)}
            )

        return answer, used_memory