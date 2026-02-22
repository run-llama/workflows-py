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
from llama_agents.server._runtime.persistence_runtime import TickPersistenceDecorator
from llama_agents.server._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    PersistentHandler,
)
from workflows.events import WorkflowCancelledEvent, WorkflowIdleEvent
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    Runtime,
    WaitResultTick,
)
from workflows.runtime.types.ticks import TickCancelRun, WorkflowTick


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
def decorator(
    mock_inner_runtime: MagicMock,
    mock_store: AsyncMock,
    mock_tick_persistence: MagicMock,
) -> DBOSIdleReleaseDecorator:
    return DBOSIdleReleaseDecorator(
        mock_inner_runtime,
        mock_store,
        idle_timeout=0.1,
        tick_persistence=mock_tick_persistence,
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
async def test_release_sends_tick_cancel_run(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """Release should send TickCancelRun via the inner external adapter."""
    idle_since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = idle_since
    mock_store.query.return_value = [handler]

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).get_external_adapter.return_value = mock_inner_external

    await decorator._release_idle_handler("run-1")

    # Should have sent TickCancelRun via inner external adapter
    mock_inner_external.send_event.assert_called_once()
    tick_arg = mock_inner_external.send_event.call_args[0][0]
    assert isinstance(tick_arg, TickCancelRun)

    # run_id should be tracked as releasing
    assert "run-1" in decorator._idle_releasing


@pytest.mark.asyncio()
async def test_release_skips_if_not_idle(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """Release should skip if handler is no longer idle."""
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = None
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
) -> None:
    """send_event should resume workflow if store shows handler is idle."""
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    handler.handler_id = "handler-1"
    handler.workflow_name = "test_wf"
    handler.error = None
    handler.result = None
    handler.started_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    handler.completed_at = None
    mock_store.query.return_value = [handler]

    mock_workflow = MagicMock()
    mock_tick_persistence.get_tracked_workflow.return_value = mock_workflow

    mock_new_adapter = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).run_workflow.return_value = mock_new_adapter

    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")

    with patch.object(
        decorator, "_broker_state_from_ticks", new_callable=AsyncMock
    ) as mock_broker:
        mock_broker.return_value = MagicMock()
        # Mock state carry-over: no state type
        with patch(
            "llama_agents.dbos.idle_release.infer_state_type", return_value=None
        ):
            await adapter.send_event(MagicMock(spec=WorkflowTick))

    # Should have started a new workflow run
    _inner(decorator).run_workflow.assert_called_once()
    # Handler should have been updated in store
    mock_store.update.assert_called_once()
    updated = mock_store.update.call_args[0][0]
    assert updated.idle_since is None
    assert updated.status == "running"
    # run_id should have changed
    assert updated.run_id != "run-1"
    # The new adapter (from run_workflow) should have received the tick
    mock_new_adapter.send_event.assert_called_once()


@pytest.mark.asyncio()
async def test_send_event_skips_resume_when_not_idle(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """send_event should not resume if store shows handler is not idle."""
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = None
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
async def test_cancelled_event_during_idle_release_overwrites_status_back_to_running(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """When ServerRuntimeDecorator sets status='cancelled' during idle release,
    the idle release adapter should overwrite it back to 'running' so the
    handler remains resumable (not treated as terminal)."""
    inner_adapter = MagicMock(spec=InternalRunAdapter)
    inner_adapter.run_id = "run-1"
    inner_adapter.write_to_event_stream = AsyncMock()

    adapter = _DBOSIdleReleaseInternalRunAdapter(inner_adapter, decorator, mock_store)

    # Simulate that this run_id is being idle-released
    decorator._idle_releasing.add("run-1")

    await adapter.write_to_event_stream(WorkflowCancelledEvent())

    # Should have forwarded the event
    inner_adapter.write_to_event_stream.assert_called_once()

    # Should have overwritten status back to "running"
    mock_store.update_handler_status.assert_called_once_with("run-1", status="running")


@pytest.mark.asyncio()
async def test_cancelled_event_without_idle_release_does_not_overwrite(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """A WorkflowCancelledEvent from a user cancel (not idle release) should
    NOT overwrite the status."""
    inner_adapter = MagicMock(spec=InternalRunAdapter)
    inner_adapter.run_id = "run-1"
    inner_adapter.write_to_event_stream = AsyncMock()

    adapter = _DBOSIdleReleaseInternalRunAdapter(inner_adapter, decorator, mock_store)

    # run-1 is NOT in _idle_releasing
    await adapter.write_to_event_stream(WorkflowCancelledEvent())

    inner_adapter.write_to_event_stream.assert_called_once()
    mock_store.update_handler_status.assert_not_called()


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
