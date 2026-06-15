"""Embedding provider abstraction layer.

Supports: HuggingFace, OpenAI. Add new providers by subclassing BaseEmbeddingProvider.
"""

from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings
from app.core.config import settings
from app.core.logger import logger


class BaseEmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        ...

    @abstractmethod
    def get_vector_size(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


class HuggingFaceProvider(BaseEmbeddingProvider):
    def __init__(self):
        self._embeddings = None
        self._vector_size = None

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info(f"Loading HuggingFace embedding model: {settings.HF_MODEL_NAME}")
            self._embeddings = HuggingFaceEmbeddings(model_name=settings.HF_MODEL_NAME)
            # Cache vector size
            test_vec = self._embeddings.embed_query("test")
            self._vector_size = len(test_vec)
            logger.info(f"Embeddings ready — dim={self._vector_size}")
        return self._embeddings

    def get_vector_size(self) -> int:
        if self._vector_size is None:
            self.get_embeddings()  # triggers initialization
        return self._vector_size  # type: ignore


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        self._embeddings = None
        self._vector_size = None

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from langchain_community.embeddings import OpenAIEmbeddings
            logger.info("Loading OpenAI embedding model")
            self._embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            test_vec = self._embeddings.embed_query("test")
            self._vector_size = len(test_vec)
            logger.info(f"OpenAI Embeddings ready — dim={self._vector_size}")
        return self._embeddings

    def get_vector_size(self) -> int:
        if self._vector_size is None:
            self.get_embeddings()
        return self._vector_size  # type: ignore


_PROVIDERS: dict[str, type[BaseEmbeddingProvider]] = {
    "huggingface": HuggingFaceProvider,
    "openai": OpenAIEmbeddingProvider,
}


def get_embedding_provider(provider_name: str | None = None) -> BaseEmbeddingProvider:
    """Factory function — returns an embedding provider instance."""
    name = (provider_name or settings.EMBEDDING_PROVIDER).lower()
    provider_cls = _PROVIDERS.get(name)
    if not provider_cls:
        raise ValueError(f"Unknown embedding provider: {name}. Available: {list(_PROVIDERS.keys())}")
    return provider_cls()
