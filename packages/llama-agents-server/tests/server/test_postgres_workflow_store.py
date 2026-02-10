# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from llama_agents.server._store.abstract_workflow_store import (
    HandlerQuery,
    PersistentHandler,
    Status,
)
from llama_agents.server._store.postgres_workflow_store import PostgresWorkflowStore
from workflows.events import Event, StopEvent

pytestmark = [pytest.mark.no_cover, pytest.mark.asyncio]

# Integration tests require TEST_POSTGRES_DSN env var
POSTGRES_DSN = os.environ.get("TEST_POSTGRES_DSN")
requires_postgres = pytest.mark.skipif(
    POSTGRES_DSN is None,
    reason="TEST_POSTGRES_DSN not set",
)


def _make_event() -> EventEnvelopeWithMetadata:
    class TestEvent(Event):
        key: str = "value"

    return EventEnvelopeWithMetadata.from_event(TestEvent())


def _make_stop_event() -> EventEnvelopeWithMetadata:
    return EventEnvelopeWithMetadata.from_event(StopEvent(result="done"))


def _make_handler(
    handler_id: str = "h1",
    workflow_name: str = "test_workflow",
    status: Status = "running",
    run_id: str = "run-1",
    started_at: datetime | None = None,
    updated_at: datetime | None = None,
    completed_at: datetime | None = None,
    idle_since: datetime | None = None,
    error: str | None = None,
) -> PersistentHandler:
    now = datetime.now(timezone.utc)
    return PersistentHandler(
        handler_id=handler_id,
        workflow_name=workflow_name,
        status=status,
        run_id=run_id,
        started_at=started_at or now,
        updated_at=updated_at or now,
        completed_at=completed_at,
        idle_since=idle_since,
        error=error,
    )


# ── Unit tests (no Postgres needed, test logic with mocks) ──────────


async def test_ticks_raise_not_implemented() -> None:
    store = PostgresWorkflowStore(dsn="postgresql://localhost/test")
    with pytest.raises(NotImplementedError):
        await store.append_tick("run-1", {"step": "init"})
    with pytest.raises(NotImplementedError):
        await store.get_ticks("run-1")


async def test_create_state_store_without_pool_raises() -> None:
    store = PostgresWorkflowStore(dsn="postgresql://localhost/test")
    with pytest.raises(RuntimeError, match="pool not initialized"):
        store.create_state_store("run-1")


async def test_build_filters_empty_in_returns_none() -> None:
    store = PostgresWorkflowStore(dsn="postgresql://localhost/test")
    assert store._build_filters(HandlerQuery(handler_id_in=[])) is None
    assert store._build_filters(HandlerQuery(run_id_in=[])) is None
    assert store._build_filters(HandlerQuery(status_in=[])) is None
    assert store._build_filters(HandlerQuery(workflow_name_in=[])) is None


async def test_build_filters_produces_correct_clauses() -> None:
    store = PostgresWorkflowStore(dsn="postgresql://localhost/test")

    result = store._build_filters(HandlerQuery(handler_id_in=["h1", "h2"]))
    assert result is not None
    clauses, params = result
    assert len(clauses) == 1
    assert "handler_id IN" in clauses[0]
    assert params == ["h1", "h2"]

    result = store._build_filters(HandlerQuery(is_idle=True))
    assert result is not None
    clauses, params = result
    assert "idle_since IS NOT NULL" in clauses[0]
    assert params == []

    result = store._build_filters(HandlerQuery(is_idle=False))
    assert result is not None
    clauses, params = result
    assert "idle_since IS NULL" in clauses[0]


async def test_on_notify_wakes_condition() -> None:
    store = PostgresWorkflowStore(dsn="postgresql://localhost/test")
    cond = store._get_or_create_condition("run-1")

    notified = asyncio.Event()

    async def waiter() -> None:
        async with cond:
            await cond.wait()
        notified.set()

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)

    # Simulate the NOTIFY callback
    store._on_notify(MagicMock(), 0, "wf_events", "run-1")

    await asyncio.wait_for(notified.wait(), timeout=1.0)
    await task


async def test_close_without_start_is_safe() -> None:
    store = PostgresWorkflowStore(dsn="postgresql://localhost/test")
    await store.close()  # Should not raise


# ── Integration tests (require real Postgres) ──────────────────────


@requires_postgres
async def test_integration_migrations_idempotent() -> None:
    from llama_agents.server._store.postgres_workflow_store import PostgresWorkflowStore

    assert POSTGRES_DSN is not None
    store = PostgresWorkflowStore(dsn=POSTGRES_DSN, schema="test_pg_store")
    try:
        await store.start()
        await store.run_migrations()
        await store.run_migrations()  # Should be idempotent
    finally:
        await store.close()


@requires_postgres
async def test_integration_handler_crud() -> None:
    from llama_agents.server._store.postgres_workflow_store import PostgresWorkflowStore

    assert POSTGRES_DSN is not None
    store = PostgresWorkflowStore(dsn=POSTGRES_DSN, schema="test_pg_store")
    try:
        await store.start()
        await store.run_migrations()

        handler = _make_handler(handler_id="pg-h1", run_id="pg-run-1")
        await store.update(handler)

        results = await store.query(HandlerQuery(handler_id_in=["pg-h1"]))
        assert len(results) == 1
        assert results[0].handler_id == "pg-h1"

        count = await store.delete(HandlerQuery(handler_id_in=["pg-h1"]))
        assert count == 1

        results = await store.query(HandlerQuery(handler_id_in=["pg-h1"]))
        assert len(results) == 0
    finally:
        await store.close()


@requires_postgres
async def test_integration_event_append_and_query() -> None:
    from llama_agents.server._store.postgres_workflow_store import PostgresWorkflowStore

    assert POSTGRES_DSN is not None
    store = PostgresWorkflowStore(dsn=POSTGRES_DSN, schema="test_pg_store")
    try:
        await store.start()
        await store.run_migrations()

        await store.append_event("pg-run-ev", _make_event())
        await store.append_event("pg-run-ev", _make_event())
        await store.append_event("pg-run-ev", _make_event())

        events = await store.query_events("pg-run-ev")
        assert len(events) == 3
        assert events[0].sequence == 0
        assert events[1].sequence == 1
        assert events[2].sequence == 2

        events = await store.query_events("pg-run-ev", after_sequence=0, limit=1)
        assert len(events) == 1
        assert events[0].sequence == 1
    finally:
        await store.close()


@requires_postgres
async def test_integration_subscribe_events() -> None:
    from llama_agents.server._store.postgres_workflow_store import PostgresWorkflowStore

    assert POSTGRES_DSN is not None
    store = PostgresWorkflowStore(
        dsn=POSTGRES_DSN, schema="test_pg_store", poll_interval=0.05
    )
    try:
        await store.start()
        await store.run_migrations()

        run_id = "pg-run-sub"

        async def append_events() -> None:
            await asyncio.sleep(0.05)
            await store.append_event(run_id, _make_event())
            await asyncio.sleep(0.05)
            await store.append_event(run_id, _make_event())
            await asyncio.sleep(0.05)
            await store.append_event(run_id, _make_stop_event())

        async def subscribe() -> list[object]:
            collected = []
            async for event in store.subscribe_events(run_id):
                collected.append(event)
            return collected

        append_task = asyncio.create_task(append_events())
        subscribe_task = asyncio.create_task(subscribe())

        collected = await asyncio.wait_for(subscribe_task, timeout=5.0)
        await append_task

        assert len(collected) == 3
    finally:
        await store.close()
