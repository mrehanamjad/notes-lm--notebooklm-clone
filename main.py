from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

# ── Import all models so Alembic can discover them ────────────────────────────
from app.features.users.model import User
from app.features.notebooks.model import Notebook
from app.features.sources.model import Source
from app.features.chat.model import ChatSession, ChatMessage, MemorySummary

# ── Import Core Schemas and Exceptions ────────────────────────────────────────
from app.core.exceptions import AppException
from app.core.schemas import APIErrorResponse

# ── Import routers ────────────────────────────────────────────────────────────
from app.features.users.router import router as users_router
from app.features.notebooks.router import router as notebooks_router
from app.features.sources.router import router as sources_router
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

# ── Global Exception Handlers ─────────────────────────────────────────────────
@app.exception_handler(AppException)    
async def app_exception_handler(request: Request, exc: AppException):
    """Catches all custom application errors and formats them identically."""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIErrorResponse(
            success=False,
            error_type=exc.error_type,
            message=exc.message,
            details=exc.details
        ).model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Overrides default FastAPI 422 errors to match our custom schema."""
    return JSONResponse(
        status_code=422,
        content=APIErrorResponse(
            success=False,
            error_type="VALIDATION_ERROR",
            message="Invalid input data",
            details=exc.errors()
        ).model_dump()
    )

# ── Register routers ─────────────────────────────────────────────────────────
app.include_router(users_router, prefix="/api/v1/users")
app.include_router(notebooks_router, prefix="/api/v1/notebooks")
app.include_router(sources_router, prefix="/api/v1/sources")
app.include_router(chat_router, prefix="/api/v1/chat")


@app.get("/")
async def root():
    return {"message": "LM Notes API is running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}