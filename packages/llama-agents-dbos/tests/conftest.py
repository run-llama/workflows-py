# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool


@pytest.fixture
def journal_db_path() -> Generator[str]:
    """Create a temporary SQLite database with the journal table."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                seq_num INTEGER NOT NULL,
                task_key TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_workflow_journal_run_id ON workflow_journal (run_id)"
        )
        conn.commit()
        conn.close()
        yield db_path


@pytest.fixture
def sqlite_engine() -> Engine:
    """Create an in-memory SQLite engine with state and journal tables.

    Used by tests that still depend on SqlStateStore (pending Phase 5 cleanup).
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS workflow_state (
                run_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL DEFAULT '{}',
                state_type TEXT NOT NULL DEFAULT 'DictState',
                state_module TEXT NOT NULL DEFAULT 'workflows.context.state_store',
                created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS workflow_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                seq_num INTEGER NOT NULL,
                task_key TEXT NOT NULL
            )
        """)
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_workflow_journal_run_id ON workflow_journal (run_id)"
            )
        )
    return engine
