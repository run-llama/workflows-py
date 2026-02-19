# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Unit tests for DBOSIdleReleaseDecorator."""

from __future__ import annotations

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

    # Should have spawned a background task
    assert len(decorator._background_tasks) == 1


@pytest.mark.asyncio()
async def test_run_workflow_tracks_active_run_id(
    decorator: DBOSIdleReleaseDecorator, mock_inner_runtime: MagicMock
) -> None:
    """run_workflow should add run_id to _active_run_ids."""
    mock_inner_runtime.run_workflow.return_value = MagicMock(spec=ExternalRunAdapter)

    decorator.run_workflow("run-1", MagicMock(), MagicMock())

    assert "run-1" in decorator._active_run_ids


@pytest.mark.asyncio()
async def test_release_cancels_and_closes(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """Release should call cancel_workflow_async and close the internal adapter."""
    decorator._active_run_ids.add("run-1")

    # Set up a mock internal adapter
    mock_internal = AsyncMock(spec=InternalRunAdapter)
    decorator._internal_adapters["run-1"] = mock_internal

    # Set up store to return a handler that's been idle long enough
    idle_since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = idle_since
    mock_store.query.return_value = [handler]

    with patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos:
        mock_dbos.cancel_workflow_async = AsyncMock()
        await decorator._release_idle_handler("run-1")

    assert "run-1" not in decorator._active_run_ids
    mock_dbos.cancel_workflow_async.assert_called_once_with("run-1")
    mock_internal.close.assert_called_once()


@pytest.mark.asyncio()
async def test_release_skips_if_not_idle(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """Release should skip if handler is no longer idle."""
    decorator._active_run_ids.add("run-1")

    handler = MagicMock(spec=PersistentHandler)
    handler.idle_since = None
    mock_store.query.return_value = [handler]

    with patch("llama_agents.dbos.idle_release.DBOS") as mock_dbos:
        mock_dbos.cancel_workflow_async = AsyncMock()
        await decorator._release_idle_handler("run-1")

    # Should not have cancelled — handler wasn't idle
    assert "run-1" in decorator._active_run_ids
    mock_dbos.cancel_workflow_async.assert_not_called()


@pytest.mark.asyncio()
async def test_send_event_triggers_resume_when_not_active(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """send_event on external adapter should resume workflow if not active."""
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

    mock_dbos.resume_workflow_async.assert_called_once_with("run-1")
    assert "run-1" in decorator._active_run_ids
    mock_inner_external.send_event.assert_called_once()


@pytest.mark.asyncio()
async def test_send_event_clears_idle_when_active(
    decorator: DBOSIdleReleaseDecorator, mock_store: AsyncMock
) -> None:
    """send_event should clear idle_since when run is active."""
    decorator._active_run_ids.add("run-1")

    mock_inner_external = AsyncMock(spec=ExternalRunAdapter)

    adapter = DBOSIdleReleaseExternalRunAdapter(decorator, "run-1")

    with patch.object(
        decorator._decorated,
        "get_external_adapter",
        return_value=mock_inner_external,
    ):
        await adapter.send_event(MagicMock(spec=WorkflowTick))

    mock_store.update_handler_status.assert_called_once_with("run-1", idle_since=None)
    mock_inner_external.send_event.assert_called_once()
