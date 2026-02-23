# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Unit tests for DBOSIdleReleaseDecorator."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_agents.dbos.idle_release import (
    DBOSIdleReleaseDecorator,
    DBOSIdleReleaseExternalRunAdapter,
    _DBOSIdleReleaseInternalRunAdapter,
)
from llama_agents.dbos.journal.crud import JournalCrud
from llama_agents.server._runtime.persistence_runtime import TickPersistenceDecorator
from llama_agents.server._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    PersistentHandler,
)
from pydantic import BaseModel
from workflows.events import WorkflowIdleEvent
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    Runtime,
    WaitResultTick,
)
from workflows.runtime.types.ticks import TickIdleRelease, WorkflowTick


# Helper to access mock methods on decorator._decorated (typed as Runtime but actually MagicMock)
def _inner(decorator: DBOSIdleReleaseDecorator) -> MagicMock:
    assert isinstance(decorator._decorated, MagicMock)
    return decorator._decorated


@pytest.fixture()
def mock_store() -> AsyncMock:
    store = AsyncMock(spec=AbstractWorkflowStore)
    store.update_handler_status = AsyncMock()
    store.query = AsyncMock(return_value=[])
    store.update = AsyncMock()
    return store


@pytest.fixture()
def mock_inner_runtime() -> MagicMock:
    runtime = MagicMock(spec=Runtime)
    runtime.get_internal_adapter = MagicMock()
    runtime.get_external_adapter = MagicMock()
    runtime.run_workflow = MagicMock()
    return runtime


@pytest.fixture()
def mock_tick_persistence() -> MagicMock:
    return MagicMock(spec=TickPersistenceDecorator)


@pytest.fixture()
def mock_journal_crud() -> AsyncMock:
    return AsyncMock(spec=JournalCrud)


@pytest.fixture()
def decorator(
    mock_inner_runtime: MagicMock,
    mock_store: AsyncMock,
    mock_tick_persistence: MagicMock,
    mock_journal_crud: AsyncMock,
) -> DBOSIdleReleaseDecorator:
    return DBOSIdleReleaseDecorator(
        mock_inner_runtime,
        mock_store,
        idle_timeout=0.1,
        tick_persistence=mock_tick_persistence,
        journal_crud=mock_journal_crud,
    )


@pytest.mark.asyncio()
async def test_idle_event_stamps_idle_since_and_spawns_release(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """WorkflowIdleEvent should stamp idle_since and schedule deferred release."""
    inner_adapter = MagicMock(spec=InternalRunAdapter)
    inner_adapter.run_id = "run-1"
    inner_adapter.write_to_event_stream = AsyncMock()

    adapter = _DBOSIdleReleaseInternalRunAdapter(inner_adapter, decorator, mock_store)
    event = WorkflowIdleEvent()

    await adapter.write_to_event_stream(event)

    # Should have stamped idle_since
    mock_store.update_handler_status.assert_called_once()
    call_kwargs = mock_store.update_handler_status.call_args
    assert call_kwargs[0][0] == "run-1"
    assert call_kwargs[1]["idle_since"] is not None

    # Should have forwarded the event
    inner_adapter.write_to_event_stream.assert_called_once_with(event)

    # Should have a deferred release task tracked by run_id
    assert "run-1" in decorator._deferred_release_tasks


@pytest.mark.asyncio()
async def test_release_sends_tick_idle_release(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """Release should send TickIdleRelease via the inner external adapter."""
    idle_since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    handler = PersistentHandler(
        handler_id="handler-1",
        workflow_name="test_wf",
        status="running",
        idle_since=idle_since,
    )
    mock_store.query.return_value = [handler]

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).get_external_adapter.return_value = mock_inner_external

    await decorator._release_idle_handler("run-1")

    # Should have sent TickIdleRelease via inner external adapter
    mock_inner_external.send_event.assert_called_once()
    tick_arg = mock_inner_external.send_event.call_args[0][0]
    assert isinstance(tick_arg, TickIdleRelease)


@pytest.mark.asyncio()
async def test_release_skips_if_not_idle(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """Release should skip if handler is no longer idle."""
    handler = PersistentHandler(
        handler_id="handler-1",
        workflow_name="test_wf",
        status="running",
    )
    mock_store.query.return_value = [handler]

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).get_external_adapter.return_value = mock_inner_external

    await decorator._release_idle_handler("run-1")

    mock_inner_external.send_event.assert_not_called()


@pytest.mark.asyncio()
async def test_send_event_triggers_resume_when_idle(
    decorator: DBOSIdleReleaseDecorator,
    mock_store: AsyncMock,
    mock_tick_persistence: MagicMock,
    mock_journal_crud: AsyncMock,
) -> None:
    """send_event should resume workflow if store shows handler is idle."""
    handler = PersistentHandler(
        handler_id="handler-1",
        workflow_name="test_wf",
        status="running",
        idle_since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        started_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    mock_store.query.return_value = [handler]

    mock_workflow = MagicMock()
    mock_tick_persistence.get_tracked_workflow.return_value = mock_workflow

    mock_new_adapter = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).run_workflow.return_value = mock_new_adapter

    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")

    mock_rebuilt_state = MagicMock()
    with patch.object(
        decorator, "_broker_state_from_ticks", new_callable=AsyncMock
    ) as mock_broker:
        mock_broker.return_value = MagicMock()
        with (
            patch("llama_agents.dbos.idle_release.infer_state_type", return_value=None),
            patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos,
            patch(
                "llama_agents.dbos.idle_release.rebuild_state_from_ticks",
                return_value=mock_rebuilt_state,
            ) as mock_rebuild,
        ):
            mock_dbos.delete_workflow_async = AsyncMock()
            tick = MagicMock(spec=WorkflowTick)
            await adapter.send_event(tick)

    # Pending tick should have been included in the rebuilt state
    mock_rebuild.assert_called_once()
    assert mock_rebuild.call_args[0][1] == [tick]

    # Should have started a new workflow run with the same run_id
    _inner(decorator).run_workflow.assert_called_once()
    call_args = _inner(decorator).run_workflow.call_args
    assert call_args[0][0] == "run-1"  # same run_id reused
    # The init_state should be the one with the pending tick applied
    assert call_args[0][2] is mock_rebuilt_state
    # Handler should have been updated in store
    mock_store.update.assert_called_once()
    updated = mock_store.update.call_args[0][0]
    assert updated.idle_since is None
    assert updated.status == "running"
    assert updated.run_id == "run-1"  # run_id stays the same
    # The tick is baked into init_state, so send_event is NOT called on the adapter
    mock_new_adapter.send_event.assert_not_called()
    # Journal and DBOS operation outputs should have been purged for the run_id
    mock_journal_crud.purge_dbos_operation_outputs.assert_called_once_with("run-1")
    mock_journal_crud.delete.assert_called_once_with("run-1")


@pytest.mark.asyncio()
async def test_send_event_skips_resume_when_not_idle(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """send_event should not resume if store shows handler is not idle."""
    handler = PersistentHandler(
        handler_id="handler-1",
        workflow_name="test_wf",
        status="running",
    )
    mock_store.query.return_value = [handler]

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).get_external_adapter.return_value = mock_inner_external

    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")

    await adapter.send_event(MagicMock(spec=WorkflowTick))

    # Should not have started a new workflow
    _inner(decorator).run_workflow.assert_not_called()
    # Should have forwarded the event
    mock_inner_external.send_event.assert_called_once()


@pytest.mark.asyncio()
async def test_wait_receive_cancels_pending_release_timer(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """wait_receive returning a tick should cancel the pending release timer."""
    # Schedule a deferred release
    decorator._schedule_deferred_release("run-1")
    assert "run-1" in decorator._deferred_release_tasks
    task = decorator._deferred_release_tasks["run-1"]

    # Create internal adapter with a mock that returns a tick
    inner_adapter = MagicMock(spec=InternalRunAdapter)
    inner_adapter.run_id = "run-1"
    tick_result = WaitResultTick(tick=MagicMock(spec=WorkflowTick))
    inner_adapter.wait_receive = AsyncMock(return_value=tick_result)

    adapter = _DBOSIdleReleaseInternalRunAdapter(inner_adapter, decorator, mock_store)

    result = await adapter.wait_receive(timeout_seconds=5.0)

    assert result is tick_result
    # Timer should have been cancelled (allow event loop to process cancellation)
    await asyncio.sleep(0)
    assert task.cancelled()
    assert "run-1" not in decorator._deferred_release_tasks


@pytest.mark.asyncio()
async def test_no_destroy_or_shutdown_cancellation(
    decorator: DBOSIdleReleaseDecorator,
) -> None:
    """Decorator should not have destroy/shutdown methods that cancel workflows."""
    assert not hasattr(decorator, "stop_task")
    assert not hasattr(decorator, "_on_server_stop")
    assert not hasattr(decorator, "_close_internal_adapter")
    assert not hasattr(decorator, "_active_run_ids")
    assert not hasattr(decorator, "_internal_adapters")


@pytest.mark.asyncio()
async def test_await_and_purge_deletes_journal_and_operation_outputs(
    decorator: DBOSIdleReleaseDecorator,
    mock_journal_crud: AsyncMock,
) -> None:
    """_await_and_purge should delete journal entries and DBOS operation outputs."""
    external = AsyncMock(spec=ExternalRunAdapter)
    external.get_result = AsyncMock(return_value=None)

    with patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos:
        mock_dbos.delete_workflow_async = AsyncMock()
        await decorator._await_and_purge("run-1", external)

    mock_dbos.delete_workflow_async.assert_called_once_with("run-1")
    mock_journal_crud.purge_dbos_operation_outputs.assert_called_once_with("run-1")
    mock_journal_crud.delete.assert_called_once_with("run-1")


@pytest.mark.asyncio()
async def test_ensure_active_run_carries_over_serialized_state(
    decorator: DBOSIdleReleaseDecorator,
    mock_store: AsyncMock,
    mock_tick_persistence: MagicMock,
    mock_journal_crud: AsyncMock,
) -> None:
    """_ensure_active_run_locked should pass serialized_state from the old state store."""

    class MyState(BaseModel):
        counter: int = 0

    handler = PersistentHandler(
        handler_id="handler-1",
        workflow_name="stateful_wf",
        status="running",
        idle_since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        started_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    mock_store.query.return_value = [handler]

    mock_workflow = MagicMock()
    mock_tick_persistence.get_tracked_workflow.return_value = mock_workflow

    # Set up state store to return a state object
    mock_state_store = AsyncMock()
    mock_state_store.get_state = AsyncMock(return_value=MyState(counter=42))
    mock_store.create_state_store = MagicMock(return_value=mock_state_store)

    mock_new_adapter = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).run_workflow.return_value = mock_new_adapter

    with (
        patch.object(
            decorator, "_broker_state_from_ticks", new_callable=AsyncMock
        ) as mock_broker,
        patch(
            "llama_agents.dbos.idle_release.infer_state_type",
            return_value=MyState,
        ),
        patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos,
    ):
        mock_broker.return_value = MagicMock()
        mock_dbos.delete_workflow_async = AsyncMock()
        await decorator._ensure_active_run_locked("run-1")

    # Should have created state store with the right type
    mock_store.create_state_store.assert_called_once_with("run-1", state_type=MyState)

    # serialized_state kwarg should be passed through to run_workflow
    call_kwargs = _inner(decorator).run_workflow.call_args
    serialized_state = call_kwargs.kwargs.get("serialized_state")
    assert serialized_state is not None
    assert "counter" in serialized_state["state_data"]
