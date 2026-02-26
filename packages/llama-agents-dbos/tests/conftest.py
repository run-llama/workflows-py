# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from llama_agents.dbos._store import SQLITE_MIGRATION_SOURCE
from llama_agents.server._store import (
    SQLITE_MIGRATION_SOURCE as SERVER_SQLITE_MIGRATION_SOURCE,
)
from llama_agents.server._store.sqlite.migrate import (
    run_migrations as sqlite_run_migrations,
)
from llama_agents_integration_tests.postgres import (
    get_asyncpg_dsn,
)
from llama_agents_integration_tests.postgres import (
    postgres_container as _postgres_container,
)
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from testcontainers.postgres import PostgresContainer

_SQLITE_SOURCES = [
    SERVER_SQLITE_MIGRATION_SOURCE,
    SQLITE_MIGRATION_SOURCE,
]


@pytest.fixture
def journal_db_path() -> Generator[str]:
    """Create a temporary SQLite database with both server and DBOS migrations."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        conn = sqlite3.connect(db_path)
        try:
            sqlite_run_migrations(conn, sources=_SQLITE_SOURCES)
        finally:
            conn.close()
        yield db_path


@pytest.fixture
def sqlite_engine(journal_db_path: str) -> Engine:
    """Create a SQLAlchemy engine pointing at the migrated test database."""
    engine = create_engine(
        f"sqlite:///{journal_db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine


@pytest.fixture(scope="module")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Module-scoped PostgreSQL container for docker-marked tests."""
    with _postgres_container() as pg:
        yield pg


@pytest.fixture(scope="module")
def postgres_dsn(postgres_container: PostgresContainer) -> str:
    """Return a plain postgresql:// DSN suitable for asyncpg."""
    return get_asyncpg_dsn(postgres_container)
