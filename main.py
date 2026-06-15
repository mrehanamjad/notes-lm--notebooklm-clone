from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Import all models so Alembic can discover them ────────────────────────────
from app.features.users.model import User          # noqa: F401
from app.features.notebooks.model import Notebook   # noqa: F401
from app.features.documents.model import Document, DocumentChunk  # noqa: F401
from app.features.chat.model import ChatSession, ChatMessage, MemorySummary  # noqa: F401

# ── Import routers ────────────────────────────────────────────────────────────
from app.features.users.router import router as users_router
from app.features.notebooks.router import router as notebooks_router
from app.features.documents.router import router as documents_router
from app.features.chat.router import router as chat_router

app = FastAPI(
    title="LM Notes API",
    description="NotebookLM-style document RAG with conversational memory",
    version="0.1.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────────────
app.include_router(users_router, prefix="/api/v1/users")
app.include_router(notebooks_router, prefix="/api/v1/notebooks")
app.include_router(documents_router, prefix="/api/v1/documents")
app.include_router(chat_router, prefix="/api/v1/chat")


@app.get("/")
async def root():
    return {"message": "LM Notes API is running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}
