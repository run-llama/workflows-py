# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""CRUD operations for the workflow journal using native database drivers."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod

import asyncpg

JOURNAL_TABLE_NAME = "workflow_journal"


class JournalCrud(ABC):
    """Abstract base for journal CRUD operations."""

    @abstractmethod
    async def insert(self, run_id: str, seq_num: int, task_key: str) -> None: ...

    @abstractmethod
    async def load(self, run_id: str) -> list[str]: ...


class PostgresJournalCrud(JournalCrud):
    """Journal CRUD using asyncpg."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        table_name: str = JOURNAL_TABLE_NAME,
        schema: str | None = None,
    ) -> None:
        self._pool = pool
        self._table_ref = f"{schema}.{table_name}" if schema else table_name

    async def insert(self, run_id: str, seq_num: int, task_key: str) -> None:
        await self._pool.execute(
            f"INSERT INTO {self._table_ref} (run_id, seq_num, task_key) VALUES ($1, $2, $3)",  # noqa: S608
            run_id,
            seq_num,
            task_key,
        )

    async def load(self, run_id: str) -> list[str]:
        rows = await self._pool.fetch(
            f"SELECT task_key FROM {self._table_ref} WHERE run_id = $1 ORDER BY seq_num ASC",  # noqa: S608
            run_id,
        )
        return [row["task_key"] for row in rows]


class SqliteJournalCrud(JournalCrud):
    """Journal CRUD using sqlite3."""

    def __init__(
        self,
        db_path: str,
        table_name: str = JOURNAL_TABLE_NAME,
    ) -> None:
        self._db_path = db_path
        self._table_name = table_name

    async def insert(self, run_id: str, seq_num: int, task_key: str) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                f"INSERT INTO {self._table_name} (run_id, seq_num, task_key) VALUES (?, ?, ?)",  # noqa: S608
                (run_id, seq_num, task_key),
            )
            conn.commit()
        finally:
            conn.close()

    async def load(self, run_id: str) -> list[str]:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                f"SELECT task_key FROM {self._table_name} WHERE run_id = ? ORDER BY seq_num ASC",  # noqa: S608
                (run_id,),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
