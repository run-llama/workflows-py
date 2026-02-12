# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Unit tests for TaskJournal class."""

from __future__ import annotations

import pytest
from llama_agents.dbos.journal.crud import SqliteJournalCrud
from llama_agents.dbos.journal.task_journal import TaskJournal


def _make_journal(run_id: str, db_path: str) -> TaskJournal:
    crud = SqliteJournalCrud(db_path=db_path)
    return TaskJournal(run_id, crud)


@pytest.mark.asyncio
async def test_fresh_journal_has_no_entries(journal_db_path: str) -> None:
    """Fresh journal returns None for next_expected_key."""
    journal = _make_journal("test-run", journal_db_path)
    await journal.load()

    assert journal.next_expected_key() is None
    assert not journal.is_replaying()


@pytest.mark.asyncio
async def test_record_adds_entry(journal_db_path: str) -> None:
    """Recording a key adds it to the journal."""
    journal = _make_journal("test-run", journal_db_path)
    await journal.load()

    await journal.record("step_a:0")

    # Verify by loading a new journal for the same run
    journal2 = _make_journal("test-run", journal_db_path)
    await journal2.load()
    assert journal2.next_expected_key() == "step_a:0"


@pytest.mark.asyncio
async def test_record_multiple_entries(journal_db_path: str) -> None:
    """Multiple records append to journal in order."""
    journal = _make_journal("test-run", journal_db_path)
    await journal.load()

    await journal.record("step_a:0")
    await journal.record("__pull__:0")
    await journal.record("step_b:1")

    # Verify order by loading a new journal
    journal2 = _make_journal("test-run", journal_db_path)
    await journal2.load()
    assert journal2.next_expected_key() == "step_a:0"
    journal2.advance()
    assert journal2.next_expected_key() == "__pull__:0"
    journal2.advance()
    assert journal2.next_expected_key() == "step_b:1"
    journal2.advance()
    assert journal2.next_expected_key() is None


@pytest.mark.asyncio
async def test_replay_returns_entries_in_order(journal_db_path: str) -> None:
    """Replaying journal returns entries in recorded order."""
    # Set up initial data
    journal1 = _make_journal("replay-run", journal_db_path)
    await journal1.load()
    await journal1.record("step_a:0")
    await journal1.record("step_b:1")
    await journal1.record("__pull__:2")

    # Load fresh journal and replay
    journal = _make_journal("replay-run", journal_db_path)
    await journal.load()

    assert journal.is_replaying()
    assert journal.next_expected_key() == "step_a:0"

    journal.advance()
    assert journal.next_expected_key() == "step_b:1"

    journal.advance()
    assert journal.next_expected_key() == "__pull__:2"

    journal.advance()
    assert journal.next_expected_key() is None
    assert not journal.is_replaying()


@pytest.mark.asyncio
async def test_load_is_idempotent(journal_db_path: str) -> None:
    """Calling load() multiple times doesn't reset state."""
    # Set up initial data
    journal1 = _make_journal("idempotent-run", journal_db_path)
    await journal1.load()
    await journal1.record("step_a:0")

    # Load and advance
    journal = _make_journal("idempotent-run", journal_db_path)
    await journal.load()
    journal.advance()
    assert journal.next_expected_key() is None

    # Load again - should not reset
    await journal.load()
    assert journal.next_expected_key() is None


@pytest.mark.asyncio
async def test_none_crud_works_in_memory() -> None:
    """Journal works without crud (in-memory only)."""
    journal = TaskJournal("memory-run", crud=None)
    await journal.load()

    assert journal.next_expected_key() is None

    await journal.record("step_a:0")
    await journal.record("step_b:1")

    # New journal with same run_id but no crud won't see the entries
    journal2 = TaskJournal("memory-run", crud=None)
    await journal2.load()
    assert journal2.next_expected_key() is None


@pytest.mark.asyncio
async def test_record_advances_index(journal_db_path: str) -> None:
    """Recording advances the replay index to stay in sync."""
    journal = _make_journal("index-run", journal_db_path)
    await journal.load()

    # After recording, index should advance
    await journal.record("step_a:0")
    # Record another
    await journal.record("step_b:1")

    # The journal's internal state shows 2 entries recorded
    assert journal._entries == ["step_a:0", "step_b:1"]
    assert journal._replay_index == 2  # We've advanced past both


@pytest.mark.asyncio
async def test_mixed_replay_and_fresh_execution(journal_db_path: str) -> None:
    """Journal transitions from replay to fresh execution correctly."""
    # Set up initial data
    journal1 = _make_journal("mixed-run", journal_db_path)
    await journal1.load()
    await journal1.record("step_a:0")

    # Load fresh journal
    journal = _make_journal("mixed-run", journal_db_path)
    await journal.load()

    # Replay the existing entry
    assert journal.next_expected_key() == "step_a:0"
    journal.advance()

    # Now fresh execution
    assert journal.next_expected_key() is None
    await journal.record("step_b:1")

    # Verify both entries persisted
    journal2 = _make_journal("mixed-run", journal_db_path)
    await journal2.load()
    assert journal2.next_expected_key() == "step_a:0"
    journal2.advance()
    assert journal2.next_expected_key() == "step_b:1"


@pytest.mark.asyncio
async def test_empty_journal_is_valid(journal_db_path: str) -> None:
    """Empty journal (no entries) is a valid state."""
    journal = _make_journal("empty-run", journal_db_path)
    await journal.load()

    assert journal.next_expected_key() is None
    assert not journal.is_replaying()


@pytest.mark.asyncio
async def test_run_id_isolation(journal_db_path: str) -> None:
    """Journals with different run_ids are isolated."""
    journal1 = _make_journal("run-1", journal_db_path)
    await journal1.load()
    await journal1.record("step_a:0")

    journal2 = _make_journal("run-2", journal_db_path)
    await journal2.load()
    await journal2.record("step_b:0")

    # Each journal sees only its own entries
    check1 = _make_journal("run-1", journal_db_path)
    await check1.load()
    assert check1.next_expected_key() == "step_a:0"

    check2 = _make_journal("run-2", journal_db_path)
    await check2.load()
    assert check2.next_expected_key() == "step_b:0"
