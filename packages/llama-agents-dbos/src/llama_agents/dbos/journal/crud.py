# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""CRUD operations for the workflow journal using native database drivers."""

from __future__ import annotations

import re
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator

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

    @abstractmethod
    async def delete(self, run_id: str) -> None: ...

    @abstractmethod
    async def truncate_from(self, run_id: str, seq_num: int) -> None: ...

    @abstractmethod
    async def purge_operations_from(self, run_id: str, function_id: int) -> None:
        """Delete DBOS operation_outputs rows beyond the given function_id."""
        ...


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
        self._ops_table_ref = _qualified_table_ref("operation_outputs", schema)

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

    async def delete(self, run_id: str) -> None:
        await self._pool.execute(
            f"DELETE FROM {self._table_ref} WHERE run_id = $1",
            run_id,
        )

    async def truncate_from(self, run_id: str, seq_num: int) -> None:
        await self._pool.execute(
            f"DELETE FROM {self._table_ref} WHERE run_id = $1 AND seq_num >= $2",
            run_id,
            seq_num,
        )

    async def purge_operations_from(self, run_id: str, function_id: int) -> None:
        await self._pool.execute(
            f"DELETE FROM {self._ops_table_ref} "
            f"WHERE workflow_uuid = $1 AND function_id > $2",
            run_id,
            function_id,
        )


class SqliteJournalCrud(JournalCrud):
    """Journal CRUD using sqlite3."""

    def __init__(
        self,
        db_path: str,
        table_name: str = JOURNAL_TABLE_NAME,
    ) -> None:
        self._db_path = db_path
        self._table_ref = _quote_identifier(table_name)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
        finally:
            conn.close()

    async def insert(self, run_id: str, seq_num: int, task_key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {self._table_ref} (run_id, seq_num, task_key) VALUES (?, ?, ?)",
                (run_id, seq_num, task_key),
            )
            conn.commit()

    async def load(self, run_id: str) -> list[str]:
        with self._connect() as conn:
            cursor = conn.execute(
                f"SELECT task_key FROM {self._table_ref} WHERE run_id = ? ORDER BY seq_num ASC",
                (run_id,),
            )
            return [row[0] for row in cursor.fetchall()]

    async def delete(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {self._table_ref} WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()

    async def truncate_from(self, run_id: str, seq_num: int) -> None:
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {self._table_ref} WHERE run_id = ? AND seq_num >= ?",
                (run_id, seq_num),
            )
            conn.commit()

    async def purge_operations_from(self, run_id: str, function_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                'DELETE FROM "operation_outputs" '
                "WHERE workflow_uuid = ? AND function_id > ?",
                (run_id, function_id),
            )
            conn.commit()
