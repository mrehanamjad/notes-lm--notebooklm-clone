from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / '.env')


class Settings(BaseSettings):
    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_URL: str = ''

    # ── Auth ───────────────────────────────────────────────────────────────────
    SECRET_KEY: str = ''
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ── LLM Provider ──────────────────────────────────────────────────────────
    LLM_PROVIDER: str = 'groq'              # groq | openai | openrouter
    GROQ_API_KEY: str = ''
    OPENAI_API_KEY: str = ''
    OPENROUTER_API_KEY: str = ''
    LLM_MODEL: str = 'llama-3.3-70b-versatile'

    # ── Embeddings ─────────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = 'huggingface'  # huggingface | openai
    HF_MODEL_NAME: str = 'all-MiniLM-L6-v2'

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    QDRANT_URL: str = ''
    QDRANT_API_KEY: str = ''
    QDRANT_COLLECTION: str = 'document_butler'

    # ── RAG Tuning ─────────────────────────────────────────────────────────────
    RETRIEVER_K: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Conversation Memory ───────────────────────────────────────────────────
    MEMORY_WINDOW: int = 6       # recent turns kept verbatim
    SUMMARISE_AFTER: int = 6     # trigger summarisation threshold

    # ── Source limits ─────────────────────────────────────────────────────────
    MAX_PLAIN_TEXT_CHARS: int = 50000   # max length for note text

    # ── Storage ────────────────────────────────────────────────────────────────
    STORAGE_PROVIDER: str = 'imagekit'  # imagekit | local
    IMAGEKIT_PRIVATE_KEY: str = ''
    IMAGEKIT_PUBLIC_KEY: str = ''
    IMAGEKIT_URL_ENDPOINT: str = ''


    TAVILY_API_KEY: str = ''
    TAVILY_MAX_RESULTS: int = 5

    # ── YouTube ────────────────────────────────────────────────────────────────
    YOUTUBE_DATA_API_KEY: str = ''

    # ── Helpers ────────────────────────────────────────────────────────────────
    @property
    def async_database_url(self) -> str:
        """Convert psycopg2 URL to asyncpg URL for async operations."""
        from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

        url = self.DATABASE_URL
        if url.startswith('postgresql://'):
            url = url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        elif url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql+asyncpg://', 1)

        # Strip params that asyncpg doesn't support and add ssl=require
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        # Remove psycopg2-specific params
        for key in ['sslmode', 'channel_binding']:
            params.pop(key, None)
        # Add asyncpg-compatible SSL
        params['ssl'] = ['require']
        new_query = urlencode(params, doseq=True)
        url = urlunparse(parsed._replace(query=new_query))
        return url

    class Config:
        env_file = '.env'
        extra = 'ignore'


settings = Settings()