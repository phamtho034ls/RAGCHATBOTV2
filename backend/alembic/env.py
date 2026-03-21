"""Alembic env — sync PostgreSQL (psycopg2) từ DATABASE_URL."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def get_sync_url() -> str:
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://legal_bot:legal_bot_pass@localhost:5432/legal_chatbot",
    )
    return url.replace("+asyncpg", "")


def run_migrations_offline() -> None:
    context.configure(
        url=get_sync_url(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(get_sync_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
