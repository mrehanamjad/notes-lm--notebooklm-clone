"""LLM provider abstraction layer.

Supports: Groq, OpenAI, OpenRouter. Add new providers by subclassing BaseLLMProvider.
"""

from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from app.core.config import settings
from app.core.logger import logger


class BaseLLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        ...


class GroqProvider(BaseLLMProvider):
    def get_llm(self) -> BaseChatModel:
        from langchain_groq import ChatGroq
        logger.info(f"Initializing Groq LLM: {settings.LLM_MODEL}")
        return ChatGroq(
            model=settings.LLM_MODEL,
            temperature=0.2,
            max_tokens=2048,
            groq_api_key=settings.GROQ_API_KEY,
        )


class OpenAIProvider(BaseLLMProvider):
    def get_llm(self) -> BaseChatModel:
        from langchain_community.chat_models import ChatOpenAI
        logger.info(f"Initializing OpenAI LLM: {settings.LLM_MODEL}")
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.1,
            max_tokens=2048,
            api_key=settings.OPENAI_API_KEY,
        )


class OpenRouterProvider(BaseLLMProvider):
    def get_llm(self) -> BaseChatModel:
        from langchain_community.chat_models import ChatOpenAI
        logger.info(f"Initializing OpenRouter LLM: {settings.LLM_MODEL}")
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.1,
            max_tokens=2048,
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )


_PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
}


def get_llm_provider(provider_name: str | None = None) -> BaseLLMProvider:
    """Factory function — returns an LLM provider instance."""
    name = (provider_name or settings.LLM_PROVIDER).lower()
    provider_cls = _PROVIDERS.get(name)
    if not provider_cls:
        raise ValueError(f"Unknown LLM provider: {name}. Available: {list(_PROVIDERS.keys())}")
    return provider_cls()
