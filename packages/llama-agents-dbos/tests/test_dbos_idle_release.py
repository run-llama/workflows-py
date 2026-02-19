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
from llama_agents.server._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    PersistentHandler,
)
from workflows.events import WorkflowIdleEvent
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    Runtime,
)
from workflows.runtime.types.ticks import WorkflowTick


@pytest.fixture()
def mock_store() -> AsyncMock:
    store = AsyncMock(spec=AbstractWorkflowStore)
    store.update_handler_status = AsyncMock()
    store.query = AsyncMock(return_value=[])
    return store


@pytest.fixture()
def mock_inner_runtime() -> MagicMock:
    runtime = MagicMock(spec=Runtime)
    runtime.get_internal_adapter = MagicMock()
    runtime.get_external_adapter = MagicMock()
    runtime.run_workflow = MagicMock()
    return runtime


@pytest.fixture()
def decorator(
    mock_inner_runtime: MagicMock, mock_store: AsyncMock
) -> DBOSIdleReleaseDecorator:
    return DBOSIdleReleaseDecorator(mock_inner_runtime, mock_store, idle_timeout=0.1)


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
async def test_release_only_cancels_workflow(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """Release should only call cancel_workflow_async — no close, no shutdown signal."""
    # Set up store to return a handler that's been idle long enough
    idle_since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = idle_since
    mock_store.query.return_value = [handler]

    with patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos:
        mock_dbos.cancel_workflow_async = AsyncMock()
        await decorator._release_idle_handler("run-1")

    mock_dbos.cancel_workflow_async.assert_called_once_with("run-1")


@pytest.mark.asyncio()
async def test_release_skips_if_not_idle(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """Release should skip if handler is no longer idle."""
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = None
    mock_store.query.return_value = [handler]

    with patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos:
        mock_dbos.cancel_workflow_async = AsyncMock()
        await decorator._release_idle_handler("run-1")

    mock_dbos.cancel_workflow_async.assert_not_called()


@pytest.mark.asyncio()
async def test_send_event_triggers_resume_when_idle(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """send_event should resume workflow if store shows handler is idle."""
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    mock_store.query.return_value = [handler]

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)

    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")

    with (
        patch.object(
            decorator._decorated,
            "get_external_adapter",
            return_value=mock_inner_external,
        ),
        patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos,
    ):
        mock_dbos.resume_workflow_async = AsyncMock()
        await adapter.send_event(MagicMock(spec=WorkflowTick))

    mock_dbos.resume_workflow_async.assert_called_once_with("run-1")
    mock_inner_external.send_event.assert_called_once()


@pytest.mark.asyncio()
async def test_send_event_skips_resume_when_not_idle(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """send_event should not resume if store shows handler is not idle."""
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = None
    mock_store.query.return_value = [handler]

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)

    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")

    with (
        patch.object(
            decorator._decorated,
            "get_external_adapter",
            return_value=mock_inner_external,
        ),
        patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos,
    ):
        mock_dbos.resume_workflow_async = AsyncMock()
        await adapter.send_event(MagicMock(spec=WorkflowTick))

    mock_dbos.resume_workflow_async.assert_not_called()
    mock_inner_external.send_event.assert_called_once()


@pytest.mark.asyncio()
async def test_send_event_cancels_pending_release_timer(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """send_event should cancel any pending deferred release timer for the run."""
    # Schedule a deferred release
    decorator._schedule_deferred_release("run-1")
    assert "run-1" in decorator._deferred_release_tasks
    task = decorator._deferred_release_tasks["run-1"]

    # Set up store to return non-idle handler
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = None
    mock_store.query.return_value = [handler]

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)
    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")

    with patch.object(
        decorator._decorated,
        "get_external_adapter",
        return_value=mock_inner_external,
    ):
        await adapter.send_event(MagicMock(spec=WorkflowTick))

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
