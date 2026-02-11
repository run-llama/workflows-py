# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import weakref
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, List, Sequence, cast

import asyncpg
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from workflows.context import JsonSerializer
from workflows.context.serializers import BaseSerializer

from .abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
    StoredEvent,
    StoredTick,
)
from .postgres.migrate import run_migrations as _run_migrations
from .postgres_state_store import PostgresStateStore

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PostgresWorkflowStore(AbstractWorkflowStore):
    """Async Postgres workflow store using asyncpg with LISTEN/NOTIFY."""

    def __init__(
        self,
        dsn: str,
        schema: str | None = None,
        poll_interval: float = 1.0,
        handlers_table_name: str = "wf_handlers",
        events_table_name: str = "wf_events",
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._schema = schema
        self.poll_interval = poll_interval
        self._handlers_table_name = handlers_table_name
        self._events_table_name = events_table_name
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: asyncpg.Pool | None = None
        self._listen_conn: asyncpg.Connection | None = None
        self._conditions: weakref.WeakValueDictionary[str, asyncio.Condition] = (
            weakref.WeakValueDictionary()
        )

    @property
    def _handlers_ref(self) -> str:
        if self._schema:
            return f"{self._schema}.{self._handlers_table_name}"
        return self._handlers_table_name

    @property
    def _events_ref(self) -> str:
        if self._schema:
            return f"{self._schema}.{self._events_table_name}"
        return self._events_table_name

    @property
    def _notify_channel(self) -> str:
        return self._events_table_name

    async def start(self) -> None:
        """Create the connection pool and set up the LISTEN connection."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
        )
        await self._setup_listener()

    async def _setup_listener(self) -> None:
        """Set up a dedicated connection for LISTEN/NOTIFY."""
        assert self._pool is not None
        conn = cast(asyncpg.Connection, await self._pool.acquire())
        await conn.add_listener(self._notify_channel, self._on_notify)
        self._listen_conn = conn

    def _on_notify(
        self,
        connection: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        """Handle NOTIFY callback — wake up subscribers for the given run_id."""
        run_id = payload
        cond = self._conditions.get(run_id)
        if cond is not None:
            # Schedule the notify on the event loop since this callback
            # may fire from a non-async context
            loop = asyncio.get_event_loop()
            loop.create_task(self._notify_condition(cond))

    @staticmethod
    async def _notify_condition(cond: asyncio.Condition) -> None:
        async with cond:
            cond.notify_all()

    async def close(self) -> None:
        """Tear down the LISTEN connection and close the pool."""
        if self._listen_conn is not None:
            try:
                await self._listen_conn.remove_listener(
                    self._notify_channel, self._on_notify
                )
            except Exception:
                logger.debug("Failed to remove listener during close", exc_info=True)
            try:
                await self._pool.release(self._listen_conn)  # type: ignore[union-attr]
            except Exception:
                logger.debug(
                    "Failed to release listen connection during close", exc_info=True
                )
            self._listen_conn = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            await self.start()
        assert self._pool is not None
        return self._pool

    def _get_or_create_condition(self, run_id: str) -> asyncio.Condition:
        cond = self._conditions.get(run_id)
        if cond is None:
            cond = asyncio.Condition()
            self._conditions[run_id] = cond
        return cond

    def create_state_store(
        self,
        run_id: str,
        state_type: type[Any] | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> PostgresStateStore[Any]:
        if self._pool is None:
            raise RuntimeError(
                "PostgresWorkflowStore pool not initialized. Call start() first."
            )
        store = PostgresStateStore(
            pool=self._pool,
            run_id=run_id,
            state_type=state_type,
            schema=self._schema,
        )
        if serialized_state is not None and serializer is not None:
            store._pending_seed = (serialized_state, serializer)
        return store

    # ── Migrations ──────────────────────────────────────────────────────

    async def run_migrations(self) -> None:
        """Apply file-based migrations to create/update schema."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await _run_migrations(cast(asyncpg.Connection, conn), schema=self._schema)

    @staticmethod
    def run_migrations_sync(dsn: str, schema: str | None = None) -> None:
        """Run migrations synchronously, handling event loop detection.

        Safe to call from both sync and async contexts. When called from
        within a running event loop, runs migrations in a background thread.
        """

        async def _migrate() -> None:
            store = PostgresWorkflowStore(dsn=dsn, schema=schema)
            await store.start()
            try:
                await store.run_migrations()
            finally:
                await store.close()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(lambda: asyncio.run(_migrate())).result()
        else:
            asyncio.run(_migrate())

    # ── Handlers ────────────────────────────────────────────────────────

    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        filter_spec = self._build_filters(query)
        if filter_spec is None:
            return []

        clauses, params = filter_spec
        sql = f"""
            SELECT handler_id, workflow_name, status, run_id, error, result,
                   started_at, updated_at, completed_at, idle_since
            FROM {self._handlers_ref}
        """
        if clauses:
            sql = f"{sql} WHERE {' AND '.join(clauses)}"

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [self._row_to_handler(row) for row in rows]

    async def update(self, handler: PersistentHandler) -> None:
        result_json = None
        if handler.result is not None:
            result_json = JsonSerializer().serialize(handler.result)

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._handlers_ref}
                    (handler_id, workflow_name, status, run_id, error, result,
                     started_at, updated_at, completed_at, idle_since)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (handler_id) DO UPDATE SET
                    workflow_name = EXCLUDED.workflow_name,
                    status = EXCLUDED.status,
                    run_id = EXCLUDED.run_id,
                    error = EXCLUDED.error,
                    result = EXCLUDED.result,
                    started_at = EXCLUDED.started_at,
                    updated_at = EXCLUDED.updated_at,
                    completed_at = EXCLUDED.completed_at,
                    idle_since = EXCLUDED.idle_since
                """,
                handler.handler_id,
                handler.workflow_name,
                handler.status,
                handler.run_id,
                handler.error,
                result_json,
                handler.started_at,
                handler.updated_at,
                handler.completed_at,
                handler.idle_since,
            )

    async def delete(self, query: HandlerQuery) -> int:
        filter_spec = self._build_filters(query)
        if filter_spec is None:
            return 0

        clauses, params = filter_spec
        if not clauses:
            return 0

        sql = f"DELETE FROM {self._handlers_ref} WHERE {' AND '.join(clauses)}"
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(sql, *params)
            # asyncpg returns "DELETE N"
            return int(result.split()[-1])

    # ── Events ──────────────────────────────────────────────────────────

    _MAX_SEQUENCE_RETRIES = 5

    async def append_event(self, run_id: str, event: EventEnvelopeWithMetadata) -> None:
        now = _utc_now()
        event_json = event.model_dump_json()

        pool = await self._ensure_pool()
        insert_sql = f"""
            INSERT INTO {self._events_ref} (run_id, sequence, timestamp, event_json)
            VALUES (
                $1,
                COALESCE((SELECT MAX(sequence) FROM {self._events_ref} WHERE run_id = $1::varchar), -1) + 1,
                $2,
                $3
            )
        """
        # Retry on unique constraint violation from concurrent sequence assignment
        for attempt in range(self._MAX_SEQUENCE_RETRIES):
            try:
                async with pool.acquire() as conn:
                    await conn.execute(insert_sql, run_id, now, event_json)
                    await conn.execute(
                        "SELECT pg_notify($1, $2)",
                        self._notify_channel,
                        run_id,
                    )
                    return
            except asyncpg.UniqueViolationError:
                if attempt == self._MAX_SEQUENCE_RETRIES - 1:
                    raise
                logger.debug(
                    "Sequence conflict for run_id=%s, retrying (attempt %d)",
                    run_id,
                    attempt + 1,
                )

    async def query_events(
        self,
        run_id: str,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[StoredEvent]:
        sql = f"""
            SELECT run_id, sequence, timestamp, event_json
            FROM {self._events_ref}
            WHERE run_id = $1
        """
        params: list[Any] = [run_id]
        param_idx = 2

        if after_sequence is not None:
            sql += f" AND sequence > ${param_idx}"
            params.append(after_sequence)
            param_idx += 1

        sql += " ORDER BY sequence"

        if limit is not None:
            sql += f" LIMIT ${param_idx}"
            params.append(limit)

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            StoredEvent(
                run_id=row["run_id"],
                sequence=row["sequence"],
                timestamp=row["timestamp"],
                event=EventEnvelopeWithMetadata.model_validate_json(row["event_json"]),
            )
            for row in rows
        ]

    async def subscribe_events(
        self, run_id: str, after_sequence: int = -1
    ) -> AsyncIterator[StoredEvent]:
        condition = self._get_or_create_condition(run_id)
        cursor = after_sequence
        while True:
            events = await self.query_events(run_id, after_sequence=cursor)
            for event in events:
                yield event
                cursor = event.sequence
                if self._is_terminal_event(event):
                    return
            if not events:
                async with condition:
                    try:
                        await asyncio.wait_for(
                            condition.wait(), timeout=self.poll_interval
                        )
                    except TimeoutError:
                        pass

    # ── Ticks (not supported) ───────────────────────────────────────────

    async def append_tick(self, run_id: str, tick_data: dict[str, Any]) -> None:
        raise NotImplementedError("PostgresWorkflowStore does not support ticks")

    async def get_ticks(self, run_id: str) -> list[StoredTick]:
        raise NotImplementedError("PostgresWorkflowStore does not support ticks")

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_filters(self, query: HandlerQuery) -> tuple[list[str], list[Any]] | None:
        clauses: list[str] = []
        params: list[Any] = []
        param_idx = 1

        def add_in_clause(column: str, values: Sequence[str]) -> None:
            nonlocal param_idx
            placeholders = ", ".join([f"${param_idx + i}" for i in range(len(values))])
            clauses.append(f"{column} IN ({placeholders})")
            params.extend(values)
            param_idx += len(values)

        for field, column in [
            (query.workflow_name_in, "workflow_name"),
            (query.handler_id_in, "handler_id"),
            (query.run_id_in, "run_id"),
            (query.status_in, "status"),
        ]:
            if field is not None:
                if len(field) == 0:
                    return None
                add_in_clause(column, field)

        if query.is_idle is not None:
            if query.is_idle:
                clauses.append("idle_since IS NOT NULL")
            else:
                clauses.append("idle_since IS NULL")

        return clauses, params

    @staticmethod
    def _row_to_handler(row: asyncpg.Record) -> PersistentHandler:
        return PersistentHandler(
            handler_id=row["handler_id"],
            workflow_name=row["workflow_name"],
            status=row["status"],
            run_id=row["run_id"],
            error=row["error"],
            result=json.loads(row["result"]) if row["result"] else None,
            started_at=row["started_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            idle_since=row["idle_since"],
        )
