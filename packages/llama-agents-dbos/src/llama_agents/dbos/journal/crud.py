# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""CRUD operations and table definitions for the workflow journal."""

from __future__ import annotations

from sqlalchemy import Column, Integer, MetaData, String, Table, text
from sqlalchemy.engine import Connection, Engine

JOURNAL_TABLE_NAME = "workflow_journal"


class JournalCrud:
    """Database operations for the workflow journal table.

    Initialized with table configuration (name, schema), then provides
    methods for inserting, loading, and migrating journal entries.
    """

    def __init__(
        self,
        table_name: str = JOURNAL_TABLE_NAME,
        schema: str | None = None,
    ) -> None:
        self.table_name = table_name
        self.schema = schema

    @property
    def _table_ref(self) -> str:
        if self.schema:
            return f"{self.schema}.{self.table_name}"
        return self.table_name

    def _define_table(self, metadata: MetaData) -> Table:
        return Table(
            self.table_name,
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String(255), nullable=False, index=True),
            Column("seq_num", Integer, nullable=False),
            Column("task_key", String(512), nullable=False),
        )

    def insert(
        self,
        conn: Connection,
        run_id: str,
        seq_num: int,
        task_key: str,
    ) -> None:
        """Insert a new journal entry."""
        conn.execute(
            text(f"""
                INSERT INTO {self._table_ref} (run_id, seq_num, task_key)
                VALUES (:run_id, :seq_num, :task_key)
            """),  # noqa: S608
            {
                "run_id": run_id,
                "seq_num": seq_num,
                "task_key": task_key,
            },
        )

    def load(self, conn: Connection, run_id: str) -> list[str]:
        """Load journal entries for a run, ordered by sequence number."""
        result = conn.execute(
            text(f"""
                SELECT task_key FROM {self._table_ref}
                WHERE run_id = :run_id
                ORDER BY seq_num ASC
            """),  # noqa: S608
            {"run_id": run_id},
        )
        return [row[0] for row in result.fetchall()]

    def run_migrations(self, engine: Engine) -> None:
        """Create the journal table if it doesn't exist."""
        metadata = MetaData(schema=self.schema)
        table = self._define_table(metadata)

        with engine.begin() as conn:
            is_postgres = engine.dialect.name == "postgresql"
            if is_postgres and self.schema:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))  # noqa: S608
            table.create(bind=conn, checkfirst=True)
