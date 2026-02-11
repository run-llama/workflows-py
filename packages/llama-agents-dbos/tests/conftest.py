# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import pytest
from llama_agents.dbos.journal.crud import JournalCrud
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool


@pytest.fixture
def sqlite_engine() -> Engine:
    """Create an in-memory SQLite engine with journal table."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    JournalCrud().run_migrations(engine)
    return engine
