# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Unit tests for DBOSIdleReleaseDecorator."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_agents.dbos.idle_release import (
    CRASH_TIMEOUT_SECONDS,
    DBOSIdleReleaseDecorator,
    DBOSIdleReleaseExternalRunAdapter,
    _DBOSIdleReleaseInternalRunAdapter,
)
from llama_agents.dbos.journal.crud import JournalCrud
from llama_agents.dbos.journal.lifecycle import RunLifecycleLock, RunLifecycleState
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
def mock_journal_crud() -> AsyncMock:
    return AsyncMock(spec=JournalCrud)


@pytest.fixture()
def mock_lifecycle_lock() -> AsyncMock:
    return AsyncMock(spec=RunLifecycleLock)


@pytest.fixture()
def decorator(
    mock_inner_runtime: MagicMock,
    mock_store: AsyncMock,
    mock_journal_crud: AsyncMock,
    mock_lifecycle_lock: AsyncMock,
) -> DBOSIdleReleaseDecorator:
    return DBOSIdleReleaseDecorator(
        mock_inner_runtime,
        mock_store,
        idle_timeout=0.1,
        journal_crud=lambda: mock_journal_crud,
        lifecycle_lock=lambda: mock_lifecycle_lock,
    )


@pytest.mark.asyncio()
async def test_idle_event_schedules_release_without_stamping_idle_since(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """WorkflowIdleEvent should schedule deferred release but NOT stamp idle_since."""
    inner_adapter = MagicMock(spec=InternalRunAdapter)
    inner_adapter.run_id = "run-1"
    inner_adapter.write_to_event_stream = AsyncMock()

    adapter = _DBOSIdleReleaseInternalRunAdapter(inner_adapter, decorator, mock_store)
    event = WorkflowIdleEvent()

    await adapter.write_to_event_stream(event)

    # Should NOT have stamped idle_since (idle_since is now set only after release)
    mock_store.update_handler_status.assert_not_called()

    # Should have forwarded the event
    inner_adapter.write_to_event_stream.assert_called_once_with(event)

    # Should have a deferred release task tracked by run_id
    assert "run-1" in decorator._deferred_release_tasks


@pytest.mark.asyncio()
async def test_release_uses_lifecycle_lock(
    decorator: DBOSIdleReleaseDecorator,
    mock_store: AsyncMock,
    mock_lifecycle_lock: AsyncMock,
) -> None:
    """Release should use begin_release and send TickIdleRelease."""
    mock_lifecycle_lock.begin_release.return_value = True

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).get_external_adapter.return_value = mock_inner_external

    await decorator._release_idle_handler("run-1")

    mock_lifecycle_lock.begin_release.assert_called_once_with("run-1")
    mock_inner_external.send_event.assert_called_once()
    tick_arg = mock_inner_external.send_event.call_args[0][0]
    assert isinstance(tick_arg, TickIdleRelease)


@pytest.mark.asyncio()
async def test_release_skips_if_begin_release_fails(
    decorator: DBOSIdleReleaseDecorator,
    mock_lifecycle_lock: AsyncMock,
) -> None:
    """Release should skip if begin_release returns False."""
    mock_lifecycle_lock.begin_release.return_value = False

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).get_external_adapter.return_value = mock_inner_external

    await decorator._release_idle_handler("run-1")

    _inner(decorator).get_external_adapter.assert_not_called()


@pytest.mark.asyncio()
async def test_await_and_mark_released_sets_idle_since(
    decorator: DBOSIdleReleaseDecorator,
    mock_store: AsyncMock,
    mock_lifecycle_lock: AsyncMock,
) -> None:
    """After workflow completes, idle_since should be set and state marked released."""
    external = AsyncMock(spec=ExternalRunAdapter)
    external.get_result = AsyncMock(return_value=None)

    await decorator._await_and_mark_released("run-1", external)

    mock_lifecycle_lock.complete_release.assert_called_once_with("run-1")
    mock_store.update_handler_status.assert_called_once()
    call_kwargs = mock_store.update_handler_status.call_args
    assert call_kwargs[0][0] == "run-1"
    assert call_kwargs[1]["idle_since"] is not None


@pytest.mark.asyncio()
async def test_send_event_resumes_when_released(
    decorator: DBOSIdleReleaseDecorator,
    mock_store: AsyncMock,
    mock_journal_crud: AsyncMock,
    mock_lifecycle_lock: AsyncMock,
) -> None:
    """send_event should resume workflow if lifecycle state is released."""
    mock_lifecycle_lock.try_begin_resume.return_value = RunLifecycleState.released

    handler = PersistentHandler(
        handler_id="handler-1",
        workflow_name="test_wf",
        status="running",
        idle_since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        started_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    mock_store.query.return_value = [handler]

    mock_workflow = MagicMock()
    mock_workflow.workflow_name = "test_wf"
    decorator.track_workflow(mock_workflow)

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
            mock_dbos.retrieve_workflow_async = AsyncMock()
            mock_handle = AsyncMock()
            mock_dbos.retrieve_workflow_async.return_value = mock_handle
            tick = MagicMock(spec=WorkflowTick)
            await adapter.send_event(tick)

    # Pending tick should have been included in the rebuilt state
    mock_rebuild.assert_called_once()
    assert mock_rebuild.call_args[0][1] == [tick]

    # Should have started a new workflow run with the same run_id
    _inner(decorator).run_workflow.assert_called_once()
    call_args = _inner(decorator).run_workflow.call_args
    assert call_args[0][0] == "run-1"
    assert call_args[0][2] is mock_rebuilt_state

    # Handler should have been updated in store
    mock_store.update.assert_called_once()
    updated = mock_store.update.call_args[0][0]
    assert updated.idle_since is None
    assert updated.status == "running"
    assert updated.run_id == "run-1"

    # Journal and DBOS operation outputs should have been purged
    mock_journal_crud.purge_dbos_operation_outputs.assert_called_once_with("run-1")
    mock_journal_crud.delete.assert_called_once_with("run-1")


@pytest.mark.asyncio()
async def test_send_event_sends_normally_when_active(
    decorator: DBOSIdleReleaseDecorator,
    mock_store: AsyncMock,
    mock_lifecycle_lock: AsyncMock,
) -> None:
    """send_event should send normally if lifecycle returns None (active)."""
    mock_lifecycle_lock.try_begin_resume.return_value = None

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).get_external_adapter.return_value = mock_inner_external

    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")
    await adapter.send_event(MagicMock(spec=WorkflowTick))

    _inner(decorator).run_workflow.assert_not_called()
    mock_inner_external.send_event.assert_called_once()


@pytest.mark.asyncio()
async def test_send_event_waits_on_releasing_then_resumes(
    decorator: DBOSIdleReleaseDecorator,
    mock_store: AsyncMock,
    mock_journal_crud: AsyncMock,
    mock_lifecycle_lock: AsyncMock,
) -> None:
    """send_event should poll when releasing, then resume when released."""
    # First call: releasing. Second call: released.
    mock_lifecycle_lock.try_begin_resume.side_effect = [
        RunLifecycleState.releasing,
        RunLifecycleState.released,
    ]
    mock_lifecycle_lock.get_state.return_value = (
        RunLifecycleState.releasing,
        datetime.now(timezone.utc),
    )

    handler = PersistentHandler(
        handler_id="handler-1",
        workflow_name="test_wf",
        status="running",
        idle_since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        started_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    mock_store.query.return_value = [handler]

    mock_workflow = MagicMock()
    mock_workflow.workflow_name = "test_wf"
    decorator.track_workflow(mock_workflow)

    mock_new_adapter = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).run_workflow.return_value = mock_new_adapter

    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")

    with (
        patch.object(
            decorator, "_broker_state_from_ticks", new_callable=AsyncMock
        ) as mock_broker,
        patch("llama_agents.dbos.idle_release.infer_state_type", return_value=None),
        patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos,
        patch("llama_agents.dbos.idle_release.rebuild_state_from_ticks"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_broker.return_value = MagicMock()
        mock_dbos.delete_workflow_async = AsyncMock()
        mock_dbos.retrieve_workflow_async = AsyncMock()
        mock_dbos.retrieve_workflow_async.return_value = AsyncMock()
        await adapter.send_event(MagicMock(spec=WorkflowTick))

    assert mock_lifecycle_lock.try_begin_resume.call_count == 2
    _inner(decorator).run_workflow.assert_called_once()


@pytest.mark.asyncio()
async def test_send_event_force_resumes_on_crash_timeout(
    decorator: DBOSIdleReleaseDecorator,
    mock_store: AsyncMock,
    mock_journal_crud: AsyncMock,
    mock_lifecycle_lock: AsyncMock,
) -> None:
    """send_event should force resume if releasing state is stale."""
    mock_lifecycle_lock.try_begin_resume.return_value = RunLifecycleState.releasing
    stale_time = datetime.now(timezone.utc) - timedelta(
        seconds=CRASH_TIMEOUT_SECONDS + 10
    )
    mock_lifecycle_lock.get_state.return_value = (
        RunLifecycleState.releasing,
        stale_time,
    )

    handler = PersistentHandler(
        handler_id="handler-1",
        workflow_name="test_wf",
        status="running",
        idle_since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        started_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    mock_store.query.return_value = [handler]

    mock_workflow = MagicMock()
    mock_workflow.workflow_name = "test_wf"
    decorator.track_workflow(mock_workflow)

    mock_new_adapter = AsyncMock(spec=ExternalRunAdapter)
    _inner(decorator).run_workflow.return_value = mock_new_adapter

    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")

    with (
        patch.object(
            decorator, "_broker_state_from_ticks", new_callable=AsyncMock
        ) as mock_broker,
        patch("llama_agents.dbos.idle_release.infer_state_type", return_value=None),
        patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos,
        patch("llama_agents.dbos.idle_release.rebuild_state_from_ticks"),
    ):
        mock_broker.return_value = MagicMock()
        mock_dbos.delete_workflow_async = AsyncMock()
        mock_dbos.retrieve_workflow_async = AsyncMock()
        mock_dbos.retrieve_workflow_async.return_value = AsyncMock()
        await adapter.send_event(MagicMock(spec=WorkflowTick))

    mock_lifecycle_lock.force_resume.assert_called_once_with("run-1")
    _inner(decorator).run_workflow.assert_called_once()


@pytest.mark.asyncio()
async def test_wait_receive_cancels_pending_release_timer(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """wait_receive returning a tick should cancel the pending release timer."""
    decorator._schedule_deferred_release("run-1")
    assert "run-1" in decorator._deferred_release_tasks
    task = decorator._deferred_release_tasks["run-1"]

    inner_adapter = MagicMock(spec=InternalRunAdapter)
    inner_adapter.run_id = "run-1"
    tick_result = WaitResultTick(tick=MagicMock(spec=WorkflowTick))
    inner_adapter.wait_receive = AsyncMock(return_value=tick_result)

    adapter = _DBOSIdleReleaseInternalRunAdapter(inner_adapter, decorator, mock_store)

    result = await adapter.wait_receive(timeout_seconds=5.0)

    assert result is tick_result
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
async def test_do_resume_carries_over_serialized_state(
    decorator: DBOSIdleReleaseDecorator,
    mock_store: AsyncMock,
    mock_journal_crud: AsyncMock,
) -> None:
    """_do_resume should pass serialized_state from the old state store."""

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
    mock_workflow.workflow_name = "stateful_wf"
    decorator.track_workflow(mock_workflow)

    mock_state_store = AsyncMock()
    mock_state_store.to_dict = MagicMock(
        return_value={
            "state_type": "MyState",
            "state_data": {"counter": 42},
        }
    )
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
        mock_dbos.retrieve_workflow_async = AsyncMock()
        mock_dbos.retrieve_workflow_async.return_value = AsyncMock()
        await decorator._do_resume("run-1")

    mock_store.create_state_store.assert_called_once_with("run-1", state_type=MyState)

    call_kwargs = _inner(decorator).run_workflow.call_args
    serialized_state = call_kwargs.kwargs.get("serialized_state")
    assert serialized_state is not None
    assert "counter" in serialized_state["state_data"]
