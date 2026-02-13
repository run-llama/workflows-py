# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from llama_agents.server import (
    HandlerQuery,
    PersistentHandler,
)
from llama_agents.server._store.abstract_workflow_store import Status, StoredEvent
from llama_agents.server._store.agent_data_store import AgentDataStore
from llama_agents_integration_tests.fake_agent_data import (
    FakeAgentDataBackend,
    create_agent_data_store,
)
from workflows.context.state_store import DictState, InMemoryStateStore
from workflows.events import Event, StopEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend() -> FakeAgentDataBackend:
    return FakeAgentDataBackend()


@pytest.fixture()
def store(
    backend: FakeAgentDataBackend, monkeypatch: pytest.MonkeyPatch
) -> AgentDataStore:
    return create_agent_data_store(backend, monkeypatch)


def make_handler(
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


def make_envelope(
    event: Event | None = None,
    seq_label: int = 0,
) -> EventEnvelopeWithMetadata:
    if event is None:
        event = Event(data=f"seq-{seq_label}")
    return EventEnvelopeWithMetadata.from_event(event, include_qualified_name=False)


# ---------------------------------------------------------------------------
# Handler CRUD tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_and_query_returns_handler(store: AgentDataStore) -> None:
    handler = make_handler(handler_id="h1", run_id="run-1")
    await store.update(handler)

    result = await store.query(HandlerQuery(handler_id_in=["h1"]))
    assert len(result) == 1
    assert result[0].handler_id == "h1"
    assert result[0].run_id == "run-1"


@pytest.mark.asyncio
async def test_update_overwrites_existing(store: AgentDataStore) -> None:
    await store.update(make_handler(handler_id="h1", status="running"))
    await store.update(make_handler(handler_id="h1", status="completed"))

    result = await store.query(HandlerQuery(handler_id_in=["h1"]))
    assert len(result) == 1
    assert result[0].status == "completed"


@pytest.mark.asyncio
async def test_query_filters_by_run_id(store: AgentDataStore) -> None:
    await store.update(make_handler(handler_id="h1", run_id="run-1"))
    await store.update(make_handler(handler_id="h2", run_id="run-2"))

    result = await store.query(HandlerQuery(run_id_in=["run-1"]))
    assert len(result) == 1
    assert result[0].handler_id == "h1"


@pytest.mark.asyncio
async def test_query_filters_by_workflow_name(store: AgentDataStore) -> None:
    await store.update(make_handler(handler_id="h1", workflow_name="wf-a"))
    await store.update(make_handler(handler_id="h2", workflow_name="wf-b"))

    result = await store.query(HandlerQuery(workflow_name_in=["wf-a"]))
    assert len(result) == 1
    assert result[0].handler_id == "h1"


@pytest.mark.asyncio
async def test_query_filters_by_status(store: AgentDataStore) -> None:
    await store.update(make_handler(handler_id="h1", status="running"))
    await store.update(make_handler(handler_id="h2", status="completed"))

    result = await store.query(HandlerQuery(status_in=["completed"]))
    assert len(result) == 1
    assert result[0].handler_id == "h2"


@pytest.mark.asyncio
async def test_query_with_empty_filter_returns_nothing(store: AgentDataStore) -> None:
    await store.update(make_handler(handler_id="h1"))
    result = await store.query(HandlerQuery(handler_id_in=[]))
    assert result == []


@pytest.mark.asyncio
async def test_query_no_filters_returns_all(store: AgentDataStore) -> None:
    await store.update(make_handler(handler_id="h1"))
    await store.update(make_handler(handler_id="h2"))

    result = await store.query(HandlerQuery())
    assert len(result) == 2


@pytest.mark.asyncio
async def test_query_filters_by_is_idle(store: AgentDataStore) -> None:
    now = datetime.now(timezone.utc)
    await store.update(make_handler(handler_id="h1", idle_since=now))
    await store.update(make_handler(handler_id="h2", idle_since=None))

    idle = await store.query(HandlerQuery(is_idle=True))
    assert len(idle) == 1
    assert idle[0].handler_id == "h1"

    not_idle = await store.query(HandlerQuery(is_idle=False))
    assert len(not_idle) == 1
    assert not_idle[0].handler_id == "h2"


@pytest.mark.asyncio
async def test_delete_removes_matching_handlers(store: AgentDataStore) -> None:
    await store.update(make_handler(handler_id="h1", workflow_name="wf-a"))
    await store.update(make_handler(handler_id="h2", workflow_name="wf-b"))

    count = await store.delete(HandlerQuery(workflow_name_in=["wf-a"]))
    assert count == 1

    remaining = await store.query(HandlerQuery())
    assert len(remaining) == 1
    assert remaining[0].handler_id == "h2"


@pytest.mark.asyncio
async def test_delete_invalidates_cache(store: AgentDataStore) -> None:
    await store.update(make_handler(handler_id="h1"))
    # Verify it's cached
    assert store._id_cache.get("h1") is not None

    await store.delete(HandlerQuery(handler_id_in=["h1"]))
    assert store._id_cache.get("h1") is None


@pytest.mark.asyncio
async def test_query_multiple_run_ids(store: AgentDataStore) -> None:
    await store.update(make_handler(handler_id="h1", run_id="run-1"))
    await store.update(make_handler(handler_id="h2", run_id="run-2"))
    await store.update(make_handler(handler_id="h3", run_id="run-3"))

    result = await store.query(HandlerQuery(run_id_in=["run-1", "run-3"]))
    ids = {h.handler_id for h in result}
    assert ids == {"h1", "h3"}


# ---------------------------------------------------------------------------
# Event journal tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_event_and_query(store: AgentDataStore) -> None:
    await store.append_event("run-1", make_envelope(seq_label=0))

    result = await store.query_events("run-1")
    assert len(result) == 1
    assert result[0].run_id == "run-1"
    assert result[0].sequence == 0
    assert result[0].event.type == "Event"


@pytest.mark.asyncio
async def test_append_multiple_events(store: AgentDataStore) -> None:
    for i in range(5):
        await store.append_event("run-1", make_envelope(seq_label=i))

    result = await store.query_events("run-1")
    assert len(result) == 5
    assert [e.sequence for e in result] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_query_events_after_sequence(store: AgentDataStore) -> None:
    for i in range(5):
        await store.append_event("run-1", make_envelope(seq_label=i))

    result = await store.query_events("run-1", after_sequence=2)
    assert len(result) == 2
    assert [e.sequence for e in result] == [3, 4]


@pytest.mark.asyncio
async def test_query_events_with_limit(store: AgentDataStore) -> None:
    for i in range(5):
        await store.append_event("run-1", make_envelope(seq_label=i))

    result = await store.query_events("run-1", limit=3)
    assert len(result) == 3
    assert [e.sequence for e in result] == [0, 1, 2]


@pytest.mark.asyncio
async def test_query_events_nonexistent_run(store: AgentDataStore) -> None:
    result = await store.query_events("nonexistent")
    assert result == []


@pytest.mark.asyncio
async def test_events_isolated_by_run_id(store: AgentDataStore) -> None:
    for i in range(3):
        await store.append_event("run-a", make_envelope(seq_label=i))
    for i in range(2):
        await store.append_event("run-b", make_envelope(seq_label=i))

    result_a = await store.query_events("run-a")
    result_b = await store.query_events("run-b")

    assert len(result_a) == 3
    assert len(result_b) == 2


# ---------------------------------------------------------------------------
# Event subscription tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_events_receives_appended(store: AgentDataStore) -> None:
    collected: list[StoredEvent] = []

    async def consumer() -> None:
        async for event in store.subscribe_events("run-1"):
            collected.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)

    await store.append_event("run-1", make_envelope(seq_label=0))
    await store.append_event("run-1", make_envelope(seq_label=1))
    await store.append_event("run-1", make_envelope(event=StopEvent(data="done")))

    await asyncio.wait_for(task, timeout=2.0)
    assert len(collected) == 3
    assert [e.sequence for e in collected] == [0, 1, 2]


@pytest.mark.asyncio
async def test_subscribe_events_terminates_on_stop(store: AgentDataStore) -> None:
    collected: list[StoredEvent] = []

    async def consumer() -> None:
        async for event in store.subscribe_events("run-1"):
            collected.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)

    await store.append_event("run-1", make_envelope(seq_label=0))
    await store.append_event("run-1", make_envelope(event=StopEvent(data="done")))

    await asyncio.wait_for(task, timeout=2.0)
    assert len(collected) == 2
    assert collected[-1].event.type == "StopEvent"


@pytest.mark.asyncio
async def test_subscribe_events_already_terminated(store: AgentDataStore) -> None:
    await store.append_event("run-1", make_envelope(seq_label=0))
    await store.append_event("run-1", make_envelope(event=StopEvent(data="done")))

    collected: list[StoredEvent] = []
    async for event in store.subscribe_events("run-1"):
        collected.append(event)

    assert len(collected) == 2
    assert collected[-1].event.type == "StopEvent"


@pytest.mark.asyncio
async def test_subscribe_events_with_after_sequence(store: AgentDataStore) -> None:
    for i in range(3):
        await store.append_event("run-1", make_envelope(seq_label=i))

    collected: list[StoredEvent] = []

    async def consumer() -> None:
        async for event in store.subscribe_events("run-1", after_sequence=1):
            collected.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)

    await store.append_event("run-1", make_envelope(event=StopEvent(data="done")))

    await asyncio.wait_for(task, timeout=2.0)
    assert [e.sequence for e in collected] == [2, 3]


# ---------------------------------------------------------------------------
# Tick journal tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_tick_and_get(store: AgentDataStore) -> None:
    await store.append_tick("run-1", {"step": "a", "state": {}})
    await store.append_tick("run-1", {"step": "b", "state": {}})

    ticks = await store.get_ticks("run-1")
    assert len(ticks) == 2
    assert ticks[0].sequence == 0
    assert ticks[1].sequence == 1
    assert ticks[0].tick_data["step"] == "a"
    assert ticks[1].tick_data["step"] == "b"


@pytest.mark.asyncio
async def test_get_ticks_empty(store: AgentDataStore) -> None:
    ticks = await store.get_ticks("nonexistent")
    assert ticks == []


@pytest.mark.asyncio
async def test_ticks_isolated_by_run_id(store: AgentDataStore) -> None:
    await store.append_tick("run-a", {"step": "a1"})
    await store.append_tick("run-b", {"step": "b1"})
    await store.append_tick("run-b", {"step": "b2"})

    assert len(await store.get_ticks("run-a")) == 1
    assert len(await store.get_ticks("run-b")) == 2


# ---------------------------------------------------------------------------
# State store tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_state_store_returns_in_memory(store: AgentDataStore) -> None:
    state_store = store.create_state_store("run-1")
    assert isinstance(state_store, InMemoryStateStore)


@pytest.mark.asyncio
async def test_create_state_store_with_type(store: AgentDataStore) -> None:
    state_store = store.create_state_store("run-1", state_type=DictState)
    assert isinstance(state_store, InMemoryStateStore)
    state = await state_store.get_state()
    assert isinstance(state, DictState)


# ---------------------------------------------------------------------------
# LRU cache behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_uses_cache_on_second_call(store: AgentDataStore) -> None:
    """After the first update caches the ID, subsequent updates use it."""
    await store.update(make_handler(handler_id="h1", status="running"))
    cached = store._id_cache.get("h1")
    assert cached is not None

    # Second update should use cached ID (no search)
    await store.update(make_handler(handler_id="h1", status="completed"))
    result = await store.query(HandlerQuery(handler_id_in=["h1"]))
    assert result[0].status == "completed"
