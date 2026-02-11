# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""CRUD operations for the workflow journal using native database drivers."""

from __future__ import annotations

import re
import sqlite3
from abc import ABC, abstractmethod

import asyncpg

JOURNAL_TABLE_NAME = "workflow_journal"

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(name: str) -> str:
    """Quote a SQL identifier, raising on invalid names."""
    if not _VALID_IDENTIFIER.match(name):
        msg = f"Invalid SQL identifier: {name!r}"
        raise ValueError(msg)
    return f'"{name}"'


def _qualified_table_ref(table_name: str, schema: str | None = None) -> str:
    """Build a safely-quoted qualified table reference."""
    ref = _quote_identifier(table_name)
    if schema:
        ref = f"{_quote_identifier(schema)}.{ref}"
    return ref


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
        self._table_ref = _qualified_table_ref(table_name, schema)

    async def insert(self, run_id: str, seq_num: int, task_key: str) -> None:
        await self._pool.execute(
            f"INSERT INTO {self._table_ref} (run_id, seq_num, task_key) VALUES ($1, $2, $3)",
            run_id,
            seq_num,
            task_key,
        )

    async def load(self, run_id: str) -> list[str]:
        rows = await self._pool.fetch(
            f"SELECT task_key FROM {self._table_ref} WHERE run_id = $1 ORDER BY seq_num ASC",
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
        self._table_ref = _quote_identifier(table_name)

    async def insert(self, run_id: str, seq_num: int, task_key: str) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                f"INSERT INTO {self._table_ref} (run_id, seq_num, task_key) VALUES (?, ?, ?)",
                (run_id, seq_num, task_key),
            )
            conn.commit()
        finally:
            conn.close()

    async def load(self, run_id: str) -> list[str]:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                f"SELECT task_key FROM {self._table_ref} WHERE run_id = ? ORDER BY seq_num ASC",
                (run_id,),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
