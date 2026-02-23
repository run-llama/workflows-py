# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Unit tests for RunLifecycleLock implementations."""

from __future__ import annotations

import pytest
from llama_agents.dbos.journal.lifecycle import (
    RunLifecycleState,
    SqliteRunLifecycleLock,
)


def _make_lock(db_path: str) -> SqliteRunLifecycleLock:
    return SqliteRunLifecycleLock(db_path=db_path)


@pytest.mark.asyncio
async def test_create_sets_active(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    result = await lock.get_state("run-1")
    assert result is not None
    assert result[0] == RunLifecycleState.active


@pytest.mark.asyncio
async def test_get_state_returns_none_for_missing(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    assert await lock.get_state("nonexistent") is None


@pytest.mark.asyncio
async def test_begin_release_active_to_releasing(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    assert await lock.begin_release("run-1") is True
    result = await lock.get_state("run-1")
    assert result is not None
    assert result[0] == RunLifecycleState.releasing


@pytest.mark.asyncio
async def test_begin_release_not_active_returns_false(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")
    # Already releasing, second attempt should fail
    assert await lock.begin_release("run-1") is False


@pytest.mark.asyncio
async def test_complete_release(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")
    await lock.complete_release("run-1")
    result = await lock.get_state("run-1")
    assert result is not None
    assert result[0] == RunLifecycleState.released


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

    # State should now be active
    state = await lock.get_state("run-1")
    assert state is not None
    assert state[0] == RunLifecycleState.active


@pytest.mark.asyncio
async def test_try_begin_resume_releasing_returns_releasing(
    journal_db_path: str,
) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")

    result = await lock.try_begin_resume("run-1")
    assert result == RunLifecycleState.releasing

    # State should still be releasing
    state = await lock.get_state("run-1")
    assert state is not None
    assert state[0] == RunLifecycleState.releasing


@pytest.mark.asyncio
async def test_force_resume(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")

    await lock.force_resume("run-1")
    state = await lock.get_state("run-1")
    assert state is not None
    assert state[0] == RunLifecycleState.active


@pytest.mark.asyncio
async def test_delete(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.delete("run-1")
    assert await lock.get_state("run-1") is None


@pytest.mark.asyncio
async def test_create_is_idempotent(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.begin_release("run-1")
    await lock.complete_release("run-1")
    # Re-create should reset to active
    await lock.create("run-1")
    state = await lock.get_state("run-1")
    assert state is not None
    assert state[0] == RunLifecycleState.active


@pytest.mark.asyncio
async def test_full_lifecycle(journal_db_path: str) -> None:
    """Test the full active -> releasing -> released -> active cycle."""
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")

    assert await lock.begin_release("run-1") is True
    assert await lock.try_begin_resume("run-1") == RunLifecycleState.releasing

    await lock.complete_release("run-1")
    assert await lock.try_begin_resume("run-1") == RunLifecycleState.released

    # Now active again
    state = await lock.get_state("run-1")
    assert state is not None
    assert state[0] == RunLifecycleState.active


@pytest.mark.asyncio
async def test_run_id_isolation(journal_db_path: str) -> None:
    lock = _make_lock(journal_db_path)
    await lock.create("run-1")
    await lock.create("run-2")
    await lock.begin_release("run-1")

    state1 = await lock.get_state("run-1")
    state2 = await lock.get_state("run-2")
    assert state1 is not None and state1[0] == RunLifecycleState.releasing
    assert state2 is not None and state2[0] == RunLifecycleState.active
