# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from llama_agents.server import (
    AbstractWorkflowStore,
    MemoryWorkflowStore,
    SqliteWorkflowStore,
    StoredEvent,
)
from workflows.events import (
    Event,
    StopEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
)


def make_envelope(
    event: Event | None = None,
    seq_label: int = 0,
) -> EventEnvelopeWithMetadata:
    """Create an EventEnvelopeWithMetadata by serializing a real Event."""
    if event is None:
        event = Event(data=f"seq-{seq_label}")
    return EventEnvelopeWithMetadata.from_event(event, include_qualified_name=False)


def _stop(seq_label: int = 0) -> StopEvent:
    return StopEvent(data=f"seq-{seq_label}")


def _failed() -> WorkflowFailedEvent:
    return WorkflowFailedEvent(
        step_name="test_step",
        exception_type="ValueError",
        exception_message="boom",
        traceback="",
        attempts=1,
        elapsed_seconds=0.0,
    )


def _cancelled() -> WorkflowCancelledEvent:
    return WorkflowCancelledEvent()


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> AbstractWorkflowStore:
    if request.param == "memory":
        return MemoryWorkflowStore()
    else:
        return SqliteWorkflowStore(str(tmp_path / "test.sqlite"), poll_interval=0.1)


@pytest.mark.asyncio
async def test_append_single_event_and_query_it_back(
    store: AbstractWorkflowStore,
) -> None:
    await store.append_event("run-1", make_envelope(seq_label=0))

    result = await store.query_events("run-1")
    assert len(result) == 1
    assert result[0].run_id == "run-1"
    assert result[0].sequence == 0
    assert result[0].event.type == "Event"


@pytest.mark.asyncio
async def test_append_multiple_events_and_query_all(
    store: AbstractWorkflowStore,
) -> None:
    for i in range(5):
        await store.append_event("run-1", make_envelope(seq_label=i))

    result = await store.query_events("run-1")
    assert len(result) == 5
    assert [e.sequence for e in result] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_query_events_after_sequence_filters_correctly(
    store: AbstractWorkflowStore,
) -> None:
    for i in range(5):
        await store.append_event("run-1", make_envelope(seq_label=i))

    # after_sequence=2 should return events with sequence > 2
    result = await store.query_events("run-1", after_sequence=2)
    assert len(result) == 2
    assert [e.sequence for e in result] == [3, 4]


@pytest.mark.asyncio
async def test_query_events_with_limit(store: AbstractWorkflowStore) -> None:
    for i in range(5):
        await store.append_event("run-1", make_envelope(seq_label=i))

    result = await store.query_events("run-1", limit=3)
    assert len(result) == 3
    assert [e.sequence for e in result] == [0, 1, 2]


@pytest.mark.asyncio
async def test_query_events_with_after_sequence_and_limit(
    store: AbstractWorkflowStore,
) -> None:
    for i in range(10):
        await store.append_event("run-1", make_envelope(seq_label=i))

    # after_sequence=3 gives sequences 4..9, then limit=2 gives [4, 5]
    result = await store.query_events("run-1", after_sequence=3, limit=2)
    assert len(result) == 2
    assert [e.sequence for e in result] == [4, 5]


@pytest.mark.asyncio
async def test_query_events_for_nonexistent_run_id_returns_empty(
    store: AbstractWorkflowStore,
) -> None:
    result = await store.query_events("nonexistent-run")
    assert result == []


@pytest.mark.asyncio
async def test_events_from_different_run_ids_are_isolated(
    store: AbstractWorkflowStore,
) -> None:
    for i in range(3):
        await store.append_event("run-a", make_envelope(seq_label=i))
    for i in range(2):
        await store.append_event("run-b", make_envelope(seq_label=i))

    result_a = await store.query_events("run-a")
    result_b = await store.query_events("run-b")

    assert len(result_a) == 3
    assert all(e.run_id == "run-a" for e in result_a)

    assert len(result_b) == 2
    assert all(e.run_id == "run-b" for e in result_b)


@pytest.mark.asyncio
async def test_subscribe_events_receives_appended_events(
    store: AbstractWorkflowStore,
) -> None:
    """Appending events wakes a waiting subscriber."""
    collected: list[StoredEvent] = []

    async def consumer() -> None:
        async for event in store.subscribe_events("run-1"):
            collected.append(event)

    task = asyncio.create_task(consumer())

    # Give subscriber time to start waiting
    await asyncio.sleep(0.01)

    await store.append_event("run-1", make_envelope(seq_label=0))
    await store.append_event("run-1", make_envelope(seq_label=1))
    # Append a terminal event to end the subscription
    await store.append_event("run-1", make_envelope(event=_stop(seq_label=2)))

    await asyncio.wait_for(task, timeout=2.0)

    assert len(collected) == 3
    assert [e.sequence for e in collected] == [0, 1, 2]


@pytest.mark.asyncio
async def test_subscribe_events_terminates_on_terminal_event(
    store: AbstractWorkflowStore,
) -> None:
    """Subscriber terminates after receiving a terminal event."""
    collected: list[StoredEvent] = []

    async def consumer() -> None:
        async for event in store.subscribe_events("run-1"):
            collected.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)

    await store.append_event("run-1", make_envelope())
    await store.append_event("run-1", make_envelope(event=_failed()))

    await asyncio.wait_for(task, timeout=2.0)

    assert len(collected) == 2
    assert collected[-1].event.type == "WorkflowFailedEvent"


@pytest.mark.asyncio
async def test_subscribe_events_terminates_on_cancelled_event(
    store: AbstractWorkflowStore,
) -> None:
    """Subscriber terminates on WorkflowCancelledEvent."""
    collected: list[StoredEvent] = []

    async def consumer() -> None:
        async for event in store.subscribe_events("run-1"):
            collected.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)

    await store.append_event("run-1", make_envelope(event=_cancelled()))

    await asyncio.wait_for(task, timeout=2.0)

    assert len(collected) == 1
    assert collected[0].event.type == "WorkflowCancelledEvent"


@pytest.mark.asyncio
async def test_subscribe_events_multiple_concurrent_subscribers(
    store: AbstractWorkflowStore,
) -> None:
    """Multiple concurrent subscribers on the same run each receive all events."""
    collected_a: list[StoredEvent] = []
    collected_b: list[StoredEvent] = []

    async def consumer_a() -> None:
        async for event in store.subscribe_events("run-1"):
            collected_a.append(event)

    async def consumer_b() -> None:
        async for event in store.subscribe_events("run-1"):
            collected_b.append(event)

    task_a = asyncio.create_task(consumer_a())
    task_b = asyncio.create_task(consumer_b())
    await asyncio.sleep(0.01)

    await store.append_event("run-1", make_envelope(seq_label=0))
    await store.append_event("run-1", make_envelope(seq_label=1))
    await store.append_event("run-1", make_envelope(event=_stop(seq_label=2)))

    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=2.0)

    assert len(collected_a) == 3
    assert len(collected_b) == 3
    assert [e.sequence for e in collected_a] == [0, 1, 2]
    assert [e.sequence for e in collected_b] == [0, 1, 2]


@pytest.mark.asyncio
async def test_subscribe_events_with_after_sequence(
    store: AbstractWorkflowStore,
) -> None:
    """Subscriber can resume from a specific sequence position."""
    # Pre-populate some events
    await store.append_event("run-1", make_envelope(seq_label=0))
    await store.append_event("run-1", make_envelope(seq_label=1))
    await store.append_event("run-1", make_envelope(seq_label=2))

    collected: list[StoredEvent] = []

    async def consumer() -> None:
        async for event in store.subscribe_events("run-1", after_sequence=1):
            collected.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)

    # Append a terminal event
    await store.append_event("run-1", make_envelope(event=_stop(seq_label=3)))

    await asyncio.wait_for(task, timeout=2.0)

    # Should have events 2, 3 (skipping 0 and 1)
    assert len(collected) == 2
    assert [e.sequence for e in collected] == [2, 3]


@pytest.mark.asyncio
async def test_subscribe_events_already_terminated(
    store: AbstractWorkflowStore,
) -> None:
    """If events already contain a terminal event, subscriber terminates immediately."""
    await store.append_event("run-1", make_envelope(seq_label=0))
    await store.append_event("run-1", make_envelope(event=_stop(seq_label=1)))

    collected: list[StoredEvent] = []
    async for event in store.subscribe_events("run-1"):
        collected.append(event)

    assert len(collected) == 2
    assert collected[-1].event.type == "StopEvent"
