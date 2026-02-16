# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Integration tests for AgentDataStore against the real LlamaCloud staging API.

These tests exercise edge cases that the fake backend might not faithfully
reproduce, particularly the idle_since gt "" workaround and sequence behavior.

Run with:
    set -a && source .env && set +a
    uv run pytest packages/llama-agents-integration-tests/tests/test_agent_data_store_staging.py -v -m llamacloud
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from llama_agents.server import HandlerQuery, PersistentHandler
from llama_agents.server._store.abstract_workflow_store import Status
from llama_agents.server._store.agent_data_state_store import AgentDataStateStore
from llama_agents.server._store.agent_data_store import AgentDataStore
from workflows.context.state_store import DictState
from workflows.events import Event

_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY", "")
_BASE_URL = os.environ.get("LLAMA_CLOUD_BASE_URL", "https://api.cloud.llamaindex.ai")
_PROJECT_ID = os.environ.get("LLAMA_DEPLOY_PROJECT_ID", "")

pytestmark = pytest.mark.llamacloud


def _unique_collection() -> str:
    return f"test_{uuid.uuid4().hex[:12]}"


def _make_store(collection: str | None = None) -> AgentDataStore:
    return AgentDataStore(
        base_url=_BASE_URL,
        api_key=_API_KEY,
        project_id=_PROJECT_ID,
        deployment_name="_public",
        collection=collection or _unique_collection(),
    )


async def _cleanup_store(store: AgentDataStore) -> None:
    """Delete all items from the store's handler, event, and tick collections."""
    for collection in [
        store._collection,
        store._events_collection,
        store._ticks_collection,
        f"{store._collection}_state",
    ]:
        items = await store._client.search(collection, page_size=1000)
        for item in items:
            await store._client.delete_item(item["id"])


def _handler(
    handler_id: str = "h1",
    workflow_name: str = "wf",
    status: Status = "running",
    run_id: str | None = None,
    idle_since: datetime | None = None,
) -> PersistentHandler:
    return PersistentHandler(
        handler_id=handler_id,
        workflow_name=workflow_name,
        status=status,
        run_id=run_id,
        idle_since=idle_since,
    )


def _event_envelope(event: Event) -> EventEnvelopeWithMetadata:
    return EventEnvelopeWithMetadata.from_event(event)


@pytest.fixture()
async def store() -> AsyncGenerator[AgentDataStore, None]:
    s = _make_store()
    yield s
    await _cleanup_store(s)


# ---------------------------------------------------------------------------
# Tests: idle_since filter (the gt "" workaround)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_filter_gt_empty_string(store: AgentDataStore) -> None:
    """The gt '' trick correctly distinguishes idle vs non-idle handlers."""
    now = datetime.now(timezone.utc)

    await store.update(_handler("idle-h", idle_since=now, run_id="r1"))
    await store.update(_handler("active-h", idle_since=None, run_id="r2"))

    idle = await store.query(HandlerQuery(is_idle=True))
    idle_ids = {h.handler_id for h in idle}
    assert "idle-h" in idle_ids
    assert "active-h" not in idle_ids

    active = await store.query(HandlerQuery(is_idle=False))
    active_ids = {h.handler_id for h in active}
    assert "active-h" in active_ids
    assert "idle-h" not in active_ids


@pytest.mark.asyncio
async def test_idle_filter_after_clearing_idle_since(store: AgentDataStore) -> None:
    """Handler transitions from idle to non-idle correctly in queries."""
    now = datetime.now(timezone.utc)

    await store.update(_handler("h1", idle_since=now, run_id="r1"))

    idle = await store.query(HandlerQuery(is_idle=True))
    assert any(h.handler_id == "h1" for h in idle)

    # Clear idle_since (simulate wake-up)
    await store.update(_handler("h1", idle_since=None, run_id="r1"))

    idle = await store.query(HandlerQuery(is_idle=True))
    assert not any(h.handler_id == "h1" for h in idle)

    active = await store.query(HandlerQuery(is_idle=False))
    assert any(h.handler_id == "h1" for h in active)


# ---------------------------------------------------------------------------
# Tests: handler CRUD round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_create_query_update_delete(store: AgentDataStore) -> None:
    """Full handler lifecycle against the real API."""
    h = _handler("crud-h", run_id="r1", status="running")
    await store.update(h)

    results = await store.query(HandlerQuery(handler_id_in=["crud-h"]))
    assert len(results) == 1
    assert results[0].status == "running"

    h.status = "completed"
    await store.update(h)
    results = await store.query(HandlerQuery(handler_id_in=["crud-h"]))
    assert len(results) == 1
    assert results[0].status == "completed"

    deleted = await store.delete(HandlerQuery(handler_id_in=["crud-h"]))
    assert deleted == 1
    results = await store.query(HandlerQuery(handler_id_in=["crud-h"]))
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: event journal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_append_and_query_ordering(store: AgentDataStore) -> None:
    """Events are returned in sequence order from the real API."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    class Msg(Event):
        text: str

    for i in range(5):
        await store.append_event(run_id, _event_envelope(Msg(text=f"msg-{i}")))

    events = await store.query_events(run_id)
    assert len(events) == 5
    sequences = [e.sequence for e in events]
    assert sequences == sorted(sequences)


@pytest.mark.asyncio
async def test_event_query_after_sequence(store: AgentDataStore) -> None:
    """after_sequence filter works against the real API."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    class Ping(Event):
        n: int

    for i in range(5):
        await store.append_event(run_id, _event_envelope(Ping(n=i)))

    events = await store.query_events(run_id, after_sequence=2)
    assert len(events) == 2
    assert events[0].sequence == 3
    assert events[1].sequence == 4


# ---------------------------------------------------------------------------
# Tests: tick journal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_append_and_ordering(store: AgentDataStore) -> None:
    """Ticks round-trip and come back in sequence order."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    for i in range(3):
        await store.append_tick(run_id, {"step": i, "data": f"tick-{i}"})

    ticks = await store.get_ticks(run_id)
    assert len(ticks) == 3
    assert [t.sequence for t in ticks] == [0, 1, 2]
    assert ticks[0].tick_data["step"] == 0
    assert ticks[2].tick_data["data"] == "tick-2"


# ---------------------------------------------------------------------------
# Tests: sequence continuity across store instances
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_sequence_new_store_instance_no_collision() -> None:
    """A new AgentDataStore instance resets in-memory sequence counters.

    This documents the current behavior: a second store instance pointing
    at the same collection will restart sequence numbers from 0, producing
    duplicates. The client-side sort by sequence still works, but sequences
    are not globally unique across instances.
    """
    collection = _unique_collection()
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    class Note(Event):
        text: str

    store1 = AgentDataStore(
        base_url=_BASE_URL,
        api_key=_API_KEY,
        project_id=_PROJECT_ID,
        deployment_name="_public",
        collection=collection,
    )
    try:
        await store1.append_event(run_id, _event_envelope(Note(text="from-store1-0")))
        await store1.append_event(run_id, _event_envelope(Note(text="from-store1-1")))

        store2 = AgentDataStore(
            base_url=_BASE_URL,
            api_key=_API_KEY,
            project_id=_PROJECT_ID,
            deployment_name="_public",
            collection=collection,
        )
        await store2.append_event(run_id, _event_envelope(Note(text="from-store2-0")))

        events = await store2.query_events(run_id)
        sequences = [e.sequence for e in events]
        assert 0 in sequences
        assert sequences.count(0) == 2, (
            "Expected duplicate sequence 0 from two store instances"
        )
    finally:
        await _cleanup_store(store1)


# ---------------------------------------------------------------------------
# Tests: state store round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_store_set_get_round_trip(store: AgentDataStore) -> None:
    """State store persists and loads state from the real API."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    state_store = store.create_state_store(run_id)

    state = await state_store.get_state()
    assert isinstance(state, DictState)

    await state_store.set(path="count", value=42)
    await state_store.set(path="name", value="test")

    assert await state_store.get("count") == 42
    assert await state_store.get("name") == "test"


@pytest.mark.asyncio
async def test_state_store_new_instance_reads_persisted(store: AgentDataStore) -> None:
    """A new AgentDataStateStore with the same collection reads persisted state."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    state_store = store.create_state_store(run_id)

    await state_store.set(path="key", value="value")

    # Create a fresh instance pointing at the same collection
    restored = AgentDataStateStore(
        client=store._client,
        run_id=run_id,
        collection=f"{store._collection}_state",
    )
    val = await restored.get("key")
    assert val == "value"


@pytest.mark.asyncio
async def test_state_store_edit_state_context_manager(store: AgentDataStore) -> None:
    """edit_state atomically loads, mutates, and saves."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    state_store = store.create_state_store(run_id)

    await state_store.set(path="x", value=1)
    assert await state_store.get("x") == 1

    async with state_store.edit_state() as state:
        current = state["x"]
        state["x"] = current + 10

    assert await state_store.get("x") == 11


# ---------------------------------------------------------------------------
# Tests: compound queries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_by_status_and_workflow_name(store: AgentDataStore) -> None:
    """Multi-field queries work against the real API."""
    await store.update(
        _handler("h1", workflow_name="wf-a", status="running", run_id="r1")
    )
    await store.update(
        _handler("h2", workflow_name="wf-a", status="completed", run_id="r2")
    )
    await store.update(
        _handler("h3", workflow_name="wf-b", status="running", run_id="r3")
    )

    results = await store.query(
        HandlerQuery(
            workflow_name_in=["wf-a"],
            status_in=["running"],
        )
    )
    assert len(results) == 1
    assert results[0].handler_id == "h1"


@pytest.mark.asyncio
async def test_query_by_multiple_run_ids(store: AgentDataStore) -> None:
    """includes filter works for multi-value queries."""
    await store.update(_handler("h1", run_id="r1"))
    await store.update(_handler("h2", run_id="r2"))
    await store.update(_handler("h3", run_id="r3"))

    results = await store.query(HandlerQuery(run_id_in=["r1", "r3"]))
    ids = {h.handler_id for h in results}
    assert ids == {"h1", "h3"}
