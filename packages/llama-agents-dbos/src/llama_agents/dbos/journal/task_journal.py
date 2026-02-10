# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""TaskJournal for deterministic replay of task completion order."""

from __future__ import annotations

from .crud import JournalCrud


class TaskJournal:
    """Records task completion order for deterministic replay.

    Stores NamedTask string keys directly (e.g., "step_name:0", "__pull__:1").
    During fresh execution, records which tasks complete and in what order.
    During replay, returns the expected task key so the adapter can wait for
    the specific task that completed in the original run.

    Uses a dedicated workflow_journal table with one row per entry for efficient
    append-only storage.
    """

    def __init__(
        self,
        run_id: str,
        crud: JournalCrud | None = None,
    ) -> None:
        """Initialize the task journal.

        Args:
            run_id: Workflow run ID for this journal.
            crud: Journal CRUD operations. If None, operates in-memory only.
        """
        self._run_id = run_id
        self._crud = crud
        self._entries: list[str] | None = None  # Lazy loaded
        self._replay_index: int = 0

    async def load(self) -> None:
        """Load journal from database. Idempotent - only loads once."""
        if self._entries is not None:
            return

        if self._crud is None:
            self._entries = []
            return

        self._entries = await self._crud.load(self._run_id)

    def is_replaying(self) -> bool:
        """True if there are more journal entries to replay."""
        if self._entries is None:
            return False
        return self._replay_index < len(self._entries)

    def next_expected_key(self) -> str | None:
        """Get the next expected task key during replay, or None if fresh execution."""
        if self._entries is None or self._replay_index >= len(self._entries):
            return None
        return self._entries[self._replay_index]

    async def record(self, key: str) -> None:
        """Record a task completion and persist to database."""
        if self._entries is None:
            self._entries = []

        seq_num = len(self._entries)
        self._entries.append(key)
        self._replay_index += 1

        if self._crud is not None:
            await self._crud.insert(self._run_id, seq_num, key)

    def advance(self) -> None:
        """Advance replay index after processing a replayed task."""
        self._replay_index += 1
