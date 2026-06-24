import sys
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# ── Add project root to sys.path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Import all models so Alembic sees them ────────────────────────────────────
from app.database.base import Base
from app.features.users.model import User                               # noqa: F401
from app.features.notebooks.model import Notebook                       # noqa: F401
from app.features.sources.model import Source              # noqa: F401
from app.features.chat.model import ChatSession, ChatMessage, MemorySummary  # noqa: F401
from app.features.artifacts.model import Artifact                            # noqa: F401

from app.core.config import settings

# ── Alembic Config ────────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url():
    return settings.DATABASE_URL


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
