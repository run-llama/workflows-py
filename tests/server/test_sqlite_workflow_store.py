from pathlib import Path

import pytest

from workflows.server.abstract_workflow_store import HandlerQuery, PersistentHandler
from workflows.server.sqlite_workflow_store import SqliteWorkflowStore


@pytest.mark.asyncio
async def test_update_and_query_returns_inserted_handler(tmp_path: Path) -> None:
    db_path: str = str(tmp_path / "handlers.db")
    store = SqliteWorkflowStore(db_path)

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
async def test_update_on_conflict_overwrites_existing_row(tmp_path: Path) -> None:
    db_path: str = str(tmp_path / "handlers.db")
    store = SqliteWorkflowStore(db_path)

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

    # Should not be returned for completed=False
    result_in_progress = await store.query(
        HandlerQuery(workflow_name_in=["wf_b"], status_in=["running"])
    )
    assert result_in_progress == []

    # Should be returned for completed=True with latest ctx
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
async def test_query_filters_by_handler_id_and_empty_lists(tmp_path: Path) -> None:
    db_path: str = str(tmp_path / "handlers.db")
    store = SqliteWorkflowStore(db_path)

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
