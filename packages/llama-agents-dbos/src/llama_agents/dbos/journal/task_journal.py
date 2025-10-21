# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""TaskJournal for deterministic replay of task completion order."""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy.engine import Engine

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
        engine: Engine | None = None,
        crud: JournalCrud | None = None,
    ) -> None:
        """Initialize the task journal.

        Args:
            run_id: Workflow run ID for this journal.
            engine: SQLAlchemy engine. If None, operates in-memory only.
            crud: Journal CRUD operations. If None, uses default JournalCrud().
        """
        self._run_id = run_id
        self._engine = engine
        self._crud = crud or JournalCrud()
        self._entries: list[str] | None = None  # Lazy loaded
        self._replay_index: int = 0

    async def _run_sync(self, fn: Any, *args: Any) -> Any:
        """Run a synchronous function in the default executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    async def load(self) -> None:
        """Load journal from database. Idempotent - only loads once."""
        if self._entries is not None:
            return

        if self._engine is None:
            self._entries = []
            return

        def _load_sync() -> list[str]:
            with self._engine.connect() as conn:  # type: ignore[union-attr]
                return self._crud.load(conn, self._run_id)

        self._entries = await self._run_sync(_load_sync)

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

        if self._engine is not None:

            def _insert_sync() -> None:
                with self._engine.begin() as conn:  # type: ignore[union-attr]
                    self._crud.insert(conn, self._run_id, seq_num, key)

            await self._run_sync(_insert_sync)

    def advance(self) -> None:
        """Advance replay index after processing a replayed task."""
        self._replay_index += 1
