import pytest
from datetime import datetime, timezone

from workflows.events import StopEvent
from workflows.server.abstract_workflow_store import HandlerQuery, PersistentHandler
from workflows.server.memory_workflow_store import MemoryWorkflowStore


@pytest.mark.asyncio
async def test_update_and_query_returns_inserted_handler() -> None:
    store = MemoryWorkflowStore()

    handler = PersistentHandler(
        handler_id="h1",
        workflow_name="wf_a",
        status="running",
        ctx={"state": {"x": 1, "y": [1, 2, 3]}},
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
    assert found.ctx == {"state": {"x": 1, "y": [1, 2, 3]}}


@pytest.mark.asyncio
async def test_update_on_conflict_overwrites_existing_row() -> None:
    store = MemoryWorkflowStore()

    # Initial insert (in-progress)
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf_b",
            status="running",
            ctx={"k": "v1"},
        )
    )

    # Update same handler_id (completed) with new ctx
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf_b",
            status="completed",
            ctx={"k": "v2", "n": 42},
        )
    )

    # Should not be returned for status=running
    result_in_progress = await store.query(
        HandlerQuery(workflow_name_in=["wf_b"], status_in=["running"])
    )
    assert result_in_progress == []

    # Should be returned for status=completed with latest ctx
    result_completed = await store.query(
        HandlerQuery(workflow_name_in=["wf_b"], status_in=["completed"])
    )
    assert len(result_completed) == 1
    found = result_completed[0]
    assert found.handler_id == "h2"
    assert found.workflow_name == "wf_b"
    assert found.status == "completed"
    assert found.ctx == {"k": "v2", "n": 42}


@pytest.mark.asyncio
async def test_delete_filters_by_query() -> None:
    store = MemoryWorkflowStore()

    await store.update(
        PersistentHandler(
            handler_id="delete-me",
            workflow_name="wf_delete",
            status="completed",
            ctx={"val": 1},
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="keep-me",
            workflow_name="wf_keep",
            status="running",
            ctx={"val": 2},
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
            ctx={},
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
                ctx={"seed": hid},
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
            ctx={},
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf",
            status="completed",
            ctx={},
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h3",
            workflow_name="wf",
            status="failed",
            ctx={},
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h4",
            workflow_name="wf",
            status="cancelled",
            ctx={},
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
            ctx={},
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf_b",
            status="running",
            ctx={},
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h3",
            workflow_name="wf_a",
            status="completed",
            ctx={},
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
            ctx={},
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h2",
            workflow_name="wf_a",
            status="completed",
            ctx={},
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h3",
            workflow_name="wf_b",
            status="running",
            ctx={},
        )
    )
    await store.update(
        PersistentHandler(
            handler_id="h4",
            workflow_name="wf_b",
            status="completed",
            ctx={},
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
                ctx={},
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
        ctx={"data": "value"},
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
        ctx={},
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
