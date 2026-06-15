"""Singleton AI client instances.

Lazily initialized on first access — avoids startup cost if not used.
Import and call the getter functions wherever needed.
"""

from functools import lru_cache
from qdrant_client import QdrantClient
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from app.core.config import settings
from app.core.providers.llm import get_llm_provider
from app.core.providers.embeddings import get_embedding_provider
from app.core.logger import logger


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return a singleton QdrantClient connected to cloud."""
    if not settings.QDRANT_URL:
        raise ValueError("QDRANT_URL is not configured in .env")
    logger.info(f"Connecting to Qdrant: {settings.QDRANT_URL}")
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a singleton LLM instance."""
    provider = get_llm_provider()
    return provider.get_llm()


# We cache the provider (not just embeddings) so vector_size is also cached
@lru_cache(maxsize=1)
def _get_cached_embedding_provider():
    return get_embedding_provider()


def get_embeddings() -> Embeddings:
    """Return a singleton embeddings model."""
    return _get_cached_embedding_provider().get_embeddings()


def get_vector_size() -> int:
    """Return the embedding dimensionality."""
    return _get_cached_embedding_provider().get_vector_size()
