# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from llama_agents.server import (
    AbstractWorkflowStore,
    HandlerQuery,
    MemoryWorkflowStore,
    PersistentHandler,
    Status,
    StoredEvent,
)
from workflows.events import (
    Event,
    StopEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
)


@pytest.mark.asyncio
async def test_update_and_query_returns_inserted_handler() -> None:
    store = MemoryWorkflowStore()

    handler = PersistentHandler(
        handler_id="h1",
        workflow_name="wf_a",
        status="running",
    )

    await store.update(handler)

    # Filter by workflow_name list
    result = await store.query(
        HandlerQuery(workflow_name_in=["wf_a"], status_in=["running"])
    )

    assert len(result) == 1
    found = result[0]
    assert found.handler_id == "h1"
    assert found.workflow_name == "wf_a"
    assert found.status == "running"


@pytest.mark.asyncio
async def test_update_on_conflict_overwrites_existing_row() -> None:
    store = MemoryWorkflowStore()

    # Initial insert (in-progress)
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf_b",
            status="running",
        )
    )

    # Update same handler_id (completed)
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf_b",
            status="completed",
        )
    )

    # Should not be returned for status=running
    result_in_progress = await store.query(
        HandlerQuery(workflow_name_in=["wf_b"], status_in=["running"])
    )
    assert result_in_progress == []

    # Should be returned for status=completed with latest values
    result_completed = await store.query(
        HandlerQuery(workflow_name_in=["wf_b"], status_in=["completed"])
    )
    assert len(result_completed) == 1
    found = result_completed[0]
    assert found.handler_id == "h2"
    assert found.workflow_name == "wf_b"
    assert found.status == "completed"


@pytest.mark.asyncio
async def test_delete_filters_by_query() -> None:
    store = MemoryWorkflowStore()

    await store.update(
        PersistentHandler(
            handler_id="delete-me",
            workflow_name="wf_delete",
            status="completed",
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="keep-me",
            workflow_name="wf_keep",
            status="running",
        )
    )

    deleted = await store.delete(HandlerQuery(handler_id_in=["delete-me"]))

    assert deleted == 1
    remaining = await store.query(HandlerQuery())
    ids = {handler.handler_id for handler in remaining}
    assert ids == {"keep-me"}


@pytest.mark.asyncio
async def test_delete_noop_on_empty_filter() -> None:
    store = MemoryWorkflowStore()

    await store.update(
        PersistentHandler(
            handler_id="delete-me",
            workflow_name="wf_delete",
            status="completed",
        )
    )

    deleted = await store.delete(HandlerQuery(handler_id_in=[]))

    assert deleted == 0
    remaining = await store.query(HandlerQuery())
    assert len(remaining) == 1
    assert remaining[0].handler_id == "delete-me"


@pytest.mark.asyncio
async def test_query_filters_by_handler_id_and_empty_lists() -> None:
    store = MemoryWorkflowStore()

    # Seed three handlers
    for hid, wf in [("h1", "wf_a"), ("h2", "wf_a"), ("h3", "wf_b")]:
        await store.update(
            PersistentHandler(
                handler_id=hid,
                workflow_name=wf,
                status="running",
            )
        )

    # Filter by specific handler ids
    result = await store.query(HandlerQuery(handler_id_in=["h1", "h3"]))
    ids = {h.handler_id for h in result}
    assert ids == {"h1", "h3"}

    # Empty handler_id list short-circuits to []
    result_empty_ids = await store.query(HandlerQuery(handler_id_in=[]))
    assert result_empty_ids == []

    # Empty workflow_name list short-circuits to []
    result_empty_wf = await store.query(HandlerQuery(workflow_name_in=[]))
    assert result_empty_wf == []

    # No filters returns all
    all_rows = await store.query(HandlerQuery())
    assert {h.handler_id for h in all_rows} == {"h1", "h2", "h3"}


@pytest.mark.asyncio
async def test_query_filters_by_multiple_statuses() -> None:
    store = MemoryWorkflowStore()

    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf",
            status="running",
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf",
            status="completed",
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h3",
            workflow_name="wf",
            status="failed",
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h4",
            workflow_name="wf",
            status="cancelled",
        )
    )

    # Query for multiple statuses
    result = await store.query(HandlerQuery(status_in=["completed", "failed"]))
    ids = {h.handler_id for h in result}
    assert ids == {"h2", "h3"}

    # Empty status list returns nothing
    result_empty = await store.query(HandlerQuery(status_in=[]))
    assert result_empty == []


@pytest.mark.asyncio
async def test_query_filters_by_workflow_name() -> None:
    store = MemoryWorkflowStore()

    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf_a",
            status="running",
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf_b",
            status="running",
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h3",
            workflow_name="wf_a",
            status="completed",
        )
    )

    # Query for specific workflow
    result = await store.query(HandlerQuery(workflow_name_in=["wf_a"]))
    ids = {h.handler_id for h in result}
    assert ids == {"h1", "h3"}

    # Query for multiple workflows
    result_multi = await store.query(HandlerQuery(workflow_name_in=["wf_a", "wf_b"]))
    ids_multi = {h.handler_id for h in result_multi}
    assert ids_multi == {"h1", "h2", "h3"}


@pytest.mark.asyncio
async def test_query_combines_multiple_filters() -> None:
    store = MemoryWorkflowStore()

    # Seed multiple handlers with different combinations
    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf_a",
            status="running",
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf_a",
            status="completed",
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h3",
            workflow_name="wf_b",
            status="running",
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h4",
            workflow_name="wf_b",
            status="completed",
        )
    )

    # Combine workflow and status filters
    result = await store.query(
        HandlerQuery(workflow_name_in=["wf_a"], status_in=["running"])
    )
    ids = {h.handler_id for h in result}
    assert ids == {"h1"}

    # Combine all three filters
    result_triple = await store.query(
        HandlerQuery(
            handler_id_in=["h2", "h4"],
            workflow_name_in=["wf_a"],
            status_in=["completed"],
        )
    )
    ids_triple = {h.handler_id for h in result_triple}
    assert ids_triple == {"h2"}


@pytest.mark.asyncio
async def test_delete_removes_multiple_matching_handlers() -> None:
    store = MemoryWorkflowStore()

    # Seed multiple handlers
    for i in range(5):
        await store.update(
            PersistentHandler(
                handler_id=f"h{i}",
                workflow_name="wf",
                status="completed" if i % 2 == 0 else "running",
            )
        )

    # Delete all completed handlers
    deleted = await store.delete(HandlerQuery(status_in=["completed"]))
    assert deleted == 3  # h0, h2, h4

    remaining = await store.query(HandlerQuery())
    ids = {h.handler_id for h in remaining}
    assert ids == {"h1", "h3"}


@pytest.mark.asyncio
async def test_store_handles_all_datetime_fields() -> None:
    store = MemoryWorkflowStore()

    now = datetime.now(timezone.utc)
    handler = PersistentHandler(
        handler_id="h1",
        workflow_name="wf",
        status="completed",
        run_id="run123",
        error=None,
        result=StopEvent(result={"output": "success"}),
        started_at=now,
        updated_at=now,
        completed_at=now,
    )

    await store.update(handler)

    result = await store.query(HandlerQuery(handler_id_in=["h1"]))
    assert len(result) == 1
    found = result[0]
    assert found.run_id == "run123"
    assert found.result == StopEvent(result={"output": "success"})
    assert found.started_at == now
    assert found.updated_at == now
    assert found.completed_at == now


@pytest.mark.asyncio
async def test_store_handles_error_field() -> None:
    store = MemoryWorkflowStore()

    handler = PersistentHandler(
        handler_id="h1",
        workflow_name="wf",
        status="failed",
        error="Something went wrong",
    )

    await store.update(handler)

    result = await store.query(HandlerQuery(handler_id_in=["h1"]))
    assert len(result) == 1
    assert result[0].error == "Something went wrong"


@pytest.mark.asyncio
async def test_empty_store_returns_empty_results() -> None:
    store = MemoryWorkflowStore()

    # Query empty store
    result = await store.query(HandlerQuery())
    assert result == []

    # Delete from empty store
    deleted = await store.delete(HandlerQuery(handler_id_in=["nonexistent"]))
    assert deleted == 0


@pytest.mark.asyncio
async def test_update_handler_status_with_nonexistent_run_id() -> None:
    store = MemoryWorkflowStore()
    # Should not raise when run_id does not exist
    await store.update_handler_status("nonexistent-run-id", status="completed")


@pytest.mark.asyncio
async def test_update_handler_status_sets_status_and_completed_at() -> None:
    store = MemoryWorkflowStore()
    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf",
            status="running",
            run_id="run-1",
        )
    )

    await store.update_handler_status("run-1", status="completed")

    result = await store.query(HandlerQuery(run_id_in=["run-1"]))
    assert len(result) == 1
    handler = result[0]
    assert handler.status == "completed"
    assert handler.updated_at is not None
    assert handler.completed_at is not None


@pytest.mark.asyncio
async def test_update_handler_status_with_result() -> None:
    store = MemoryWorkflowStore()
    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf",
            status="running",
            run_id="run-1",
        )
    )

    stop = StopEvent(result={"answer": 42})
    await store.update_handler_status("run-1", status="completed", result=stop)

    result = await store.query(HandlerQuery(run_id_in=["run-1"]))
    handler = result[0]
    assert handler.status == "completed"
    assert handler.result == stop


@pytest.mark.asyncio
async def test_update_handler_status_with_error() -> None:
    store = MemoryWorkflowStore()
    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf",
            status="running",
            run_id="run-1",
        )
    )

    await store.update_handler_status("run-1", status="failed", error="boom")

    result = await store.query(HandlerQuery(run_id_in=["run-1"]))
    handler = result[0]
    assert handler.status == "failed"
    assert handler.error == "boom"
    assert handler.completed_at is not None


@pytest.mark.asyncio
async def test_update_handler_status_idle_since_explicit_none_clears() -> None:
    store = MemoryWorkflowStore()
    now = datetime.now(timezone.utc)
    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf",
            status="running",
            run_id="run-1",
            idle_since=now,
        )
    )

    # Passing idle_since=None explicitly should clear it
    await store.update_handler_status("run-1", idle_since=None)

    result = await store.query(HandlerQuery(run_id_in=["run-1"]))
    assert result[0].idle_since is None


@pytest.mark.asyncio
async def test_update_handler_status_idle_since_unset_preserves() -> None:
    store = MemoryWorkflowStore()
    now = datetime.now(timezone.utc)
    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf",
            status="running",
            run_id="run-1",
            idle_since=now,
        )
    )

    # Not passing idle_since at all should preserve the existing value
    await store.update_handler_status("run-1", status="running")

    result = await store.query(HandlerQuery(run_id_in=["run-1"]))
    assert result[0].idle_since == now


@pytest.mark.asyncio
async def test_update_handler_status_non_terminal_does_not_set_completed_at() -> None:
    store = MemoryWorkflowStore()
    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf",
            status="running",
            run_id="run-1",
        )
    )

    # Update status to "running" (non-terminal) should not set completed_at
    await store.update_handler_status("run-1", status="running")

    result = await store.query(HandlerQuery(run_id_in=["run-1"]))
    assert result[0].completed_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
async def test_update_handler_status_terminal_sets_completed_at(
    terminal_status: Status,
) -> None:
    store = MemoryWorkflowStore()
    await store.update(
        PersistentHandler(
            handler_id="h1",
            workflow_name="wf",
            status="running",
            run_id="run-1",
        )
    )

    await store.update_handler_status("run-1", status=terminal_status)

    result = await store.query(HandlerQuery(run_id_in=["run-1"]))
    assert result[0].completed_at is not None


def _make_stored_event(event: Event, run_id: str = "run-1") -> StoredEvent:
    return StoredEvent(
        run_id=run_id,
        sequence=0,
        timestamp=datetime.now(timezone.utc),
        event=EventEnvelopeWithMetadata.from_event(event),
    )


def test_is_terminal_event_stop_event() -> None:
    stored = _make_stored_event(StopEvent(result="done"))
    assert AbstractWorkflowStore._is_terminal_event(stored) is True


def test_is_terminal_event_regular_event() -> None:
    stored = _make_stored_event(Event())
    assert AbstractWorkflowStore._is_terminal_event(stored) is False


def test_is_terminal_event_workflow_failed_event() -> None:
    # WorkflowFailedEvent extends StopEvent, so it should be terminal
    event = WorkflowFailedEvent(
        step_name="my_step",
        exception_type="ValueError",
        exception_message="bad value",
        traceback="",
        attempts=1,
        elapsed_seconds=0.1,
    )
    stored = _make_stored_event(event)
    assert AbstractWorkflowStore._is_terminal_event(stored) is True


def test_is_terminal_event_workflow_cancelled_event() -> None:
    # WorkflowCancelledEvent extends StopEvent, so it should be terminal
    stored = _make_stored_event(WorkflowCancelledEvent())
    assert AbstractWorkflowStore._is_terminal_event(stored) is True
