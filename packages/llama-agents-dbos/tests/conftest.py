# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from llama_agents.server._store.sqlite.sqlite_workflow_store import (
    SqliteWorkflowStore,
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


@pytest.fixture
def journal_db_path() -> Generator[str]:
    """Create a temporary SQLite database with migrations applied."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        SqliteWorkflowStore.run_migrations(db_path)
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
