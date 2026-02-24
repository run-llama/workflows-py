# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Unit tests for RunLifecycleLock implementations."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from llama_agents.dbos.journal.lifecycle import (
    LIFECYCLE_TABLE_NAME,
    RunLifecycleState,
    SqliteRunLifecycleLock,
)


def _make_lock(db_path: str) -> SqliteRunLifecycleLock:
    return SqliteRunLifecycleLock(db_path=db_path)


def _read_state(db_path: str, run_id: str) -> tuple[str, str] | None:
    """Test helper: read raw (state, updated_at) from sqlite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        f"SELECT state, updated_at FROM {LIFECYCLE_TABLE_NAME} WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return row["state"], row["updated_at"]


def _set_updated_at(db_path: str, run_id: str, updated_at: datetime) -> None:
    """Test helper: override updated_at for crash timeout testing."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"UPDATE {LIFECYCLE_TABLE_NAME} SET updated_at = ? WHERE run_id = ?",
        (updated_at.isoformat(), run_id),
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_create_sets_active(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    result = _read_state(journal_db_path, "run-1")
    assert result is not None
    assert result[0] == "active"


@pytest.mark.asyncio
async def test_begin_release_active_to_releasing(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    assert await lock.begin_release("run-1") is True
    result = _read_state(journal_db_path, "run-1")
    assert result is not None
    assert result[0] == "releasing"


@pytest.mark.asyncio
async def test_begin_release_not_active_returns_false(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")
    assert await lock.begin_release("run-1") is False


@pytest.mark.asyncio
async def test_complete_release(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")
    await lock.complete_release("run-1")
    result = _read_state(journal_db_path, "run-1")
    assert result is not None
    assert result[0] == "released"


@pytest.mark.asyncio
async def test_try_begin_resume_no_row(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    assert await lock.try_begin_resume("nonexistent") is None


@pytest.mark.asyncio
async def test_try_begin_resume_active_returns_none(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    assert await lock.try_begin_resume("run-1") is None


@pytest.mark.asyncio
async def test_try_begin_resume_released_transitions_to_active(
    journal_db_path: str,
) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")
    await lock.complete_release("run-1")

    result = await lock.try_begin_resume("run-1")
    assert result == RunLifecycleState.released

    state = _read_state(journal_db_path, "run-1")
    assert state is not None
    assert state[0] == "active"


@pytest.mark.asyncio
async def test_try_begin_resume_releasing_returns_releasing(
    journal_db_path: str,
) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")

    result = await lock.try_begin_resume("run-1")
    assert result == RunLifecycleState.releasing

    state = _read_state(journal_db_path, "run-1")
    assert state is not None
    assert state[0] == "releasing"


@pytest.mark.asyncio
async def test_try_begin_resume_force_resumes_on_crash_timeout(
    journal_db_path: str,
) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")

    # Set updated_at to well past the timeout
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=200)
    _set_updated_at(journal_db_path, "run-1", stale_time)

    result = await lock.try_begin_resume("run-1", crash_timeout_seconds=120.0)
    assert result == RunLifecycleState.released

    state = _read_state(journal_db_path, "run-1")
    assert state is not None
    assert state[0] == "active"


@pytest.mark.asyncio
async def test_try_begin_resume_releasing_no_force_without_timeout(
    journal_db_path: str,
) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")

    # Set stale timestamp but don't pass crash_timeout_seconds
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=200)
    _set_updated_at(journal_db_path, "run-1", stale_time)

    result = await lock.try_begin_resume("run-1")
    assert result == RunLifecycleState.releasing


@pytest.mark.asyncio
async def test_create_is_idempotent(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")
    await lock.complete_release("run-1")
    await lock.create("run-1")
    state = _read_state(journal_db_path, "run-1")
    assert state is not None
    assert state[0] == "active"


@pytest.mark.asyncio
async def test_full_lifecycle(journal_db_path: str) -> None:
    """Test the full active -> releasing -> released -> active cycle."""
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")

    assert await lock.begin_release("run-1") is True
    assert await lock.try_begin_resume("run-1") == RunLifecycleState.releasing

    await lock.complete_release("run-1")
    assert await lock.try_begin_resume("run-1") == RunLifecycleState.released

    state = _read_state(journal_db_path, "run-1")
    assert state is not None
    assert state[0] == "active"


@pytest.mark.asyncio
async def test_run_id_isolation(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.create("run-2")
    await lock.begin_release("run-1")

    state1 = _read_state(journal_db_path, "run-1")
    state2 = _read_state(journal_db_path, "run-2")
    assert state1 is not None and state1[0] == "releasing"
    assert state2 is not None and state2[0] == "active"
