# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from llama_agents.server import (
    HandlerQuery,
    MemoryWorkflowStore,
    PersistentHandler,
    WorkflowServer,
)
from llama_agents.server._service import EventSendError, HandlerCompletedError
from server_test_fixtures import (  # type: ignore[import]
    ErrorWorkflow,
    ExternalEvent,
    wait_for_passing,
)
from workflows import Workflow


@pytest.mark.asyncio
async def test_cancel_running_handler(
    memory_store: MemoryWorkflowStore, interactive_workflow: Workflow
) -> None:
    """Start an interactive workflow, cancel it, and verify status becomes cancelled."""
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow(
        "interactive", interactive_workflow, additional_events=[ExternalEvent]
    )

    async with server.contextmanager():
        handler_data = await server._service.start_workflow(
            interactive_workflow, "cancel-test-1"
        )
        assert handler_data.run_id is not None

        result = await server._service.cancel_handler("cancel-test-1")
        assert result == "cancelled"

        async def status_is_cancelled() -> None:
            persisted = await memory_store.query(
                HandlerQuery(handler_id_in=["cancel-test-1"])
            )
            assert len(persisted) == 1
            assert persisted[0].status == "cancelled"

        await wait_for_passing(status_is_cancelled, max_duration=2.0, interval=0.01)


@pytest.mark.asyncio
async def test_cancel_handler_with_purge(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    """Start and complete a workflow, then purge it from the store."""
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("simple", simple_test_workflow)

    async with server.contextmanager():
        await server._service.start_workflow(simple_test_workflow, "purge-test-1")

        # Wait for completion
        async def handler_completed() -> None:
            persisted = await memory_store.query(
                HandlerQuery(handler_id_in=["purge-test-1"])
            )
            assert len(persisted) == 1
            assert persisted[0].status == "completed"

        await wait_for_passing(handler_completed, max_duration=2.0, interval=0.01)

        result = await server._service.cancel_handler("purge-test-1", purge=True)
        assert result == "deleted"

        # Handler should be gone from store
        persisted = await memory_store.query(
            HandlerQuery(handler_id_in=["purge-test-1"])
        )
        assert len(persisted) == 0


@pytest.mark.asyncio
async def test_cancel_handler_not_found(memory_store: MemoryWorkflowStore) -> None:
    """Cancelling a nonexistent handler returns None."""
    server = WorkflowServer(workflow_store=memory_store)

    async with server.contextmanager():
        result = await server._service.cancel_handler("nonexistent")
        assert result is None


@pytest.mark.asyncio
async def test_send_event_workflow_not_registered(
    memory_store: MemoryWorkflowStore,
) -> None:
    """Sending an event to a handler whose workflow is not registered raises EventSendError."""
    server = WorkflowServer(workflow_store=memory_store)

    # Seed store with a handler for an unregistered workflow
    await memory_store.update(
        PersistentHandler(
            handler_id="orphan-handler",
            workflow_name="unregistered",
            status="running",
            run_id="some-run-id",
            started_at=datetime.now(timezone.utc),
        )
    )

    async with server.contextmanager():
        with pytest.raises(EventSendError, match="not registered"):
            await server._service.send_event(
                "orphan-handler", ExternalEvent(response="hello")
            )


@pytest.mark.asyncio
async def test_send_event_no_run_id(
    memory_store: MemoryWorkflowStore, interactive_workflow: Workflow
) -> None:
    """Sending an event to a handler with no run_id raises EventSendError."""
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow(
        "interactive", interactive_workflow, additional_events=[ExternalEvent]
    )

    # Seed store with a handler that has no run_id
    await memory_store.update(
        PersistentHandler(
            handler_id="no-run-handler",
            workflow_name="interactive",
            status="running",
            run_id=None,
            started_at=datetime.now(timezone.utc),
        )
    )

    async with server.contextmanager():
        with pytest.raises(EventSendError, match="no run ID"):
            await server._service.send_event(
                "no-run-handler", ExternalEvent(response="hello")
            )


@pytest.mark.asyncio
async def test_start_workflow_happy_path(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    """start_workflow returns HandlerData with correct initial fields."""
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("simple", simple_test_workflow)

    async with server.contextmanager():
        handler_data = await server._service.start_workflow(
            simple_test_workflow, "start-hp-1"
        )
        assert handler_data.handler_id == "start-hp-1"
        assert handler_data.workflow_name == "simple"
        assert handler_data.run_id is not None
        assert handler_data.status == "running"


@pytest.mark.asyncio
async def test_await_workflow_happy_path(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    """await_workflow returns completed HandlerData."""
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("simple", simple_test_workflow)

    async with server.contextmanager():
        handler_data = await server._service.start_workflow(
            simple_test_workflow, "await-hp-1"
        )
        result = await server._service.await_workflow(handler_data)
        assert result.status == "completed"
        assert result.handler_id == "await-hp-1"


@pytest.mark.asyncio
async def test_await_workflow_error_returns_failed(
    memory_store: MemoryWorkflowStore,
) -> None:
    """await_workflow on an ErrorWorkflow returns failed status, not an exception."""
    error_wf = ErrorWorkflow()
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("error", error_wf)

    async with server.contextmanager():
        handler_data = await server._service.start_workflow(error_wf, "await-err-1")
        result = await server._service.await_workflow(handler_data)
        assert result.status == "failed"


@pytest.mark.asyncio
async def test_resolve_handler_raises_on_completed(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    """resolve_handler raises HandlerCompletedError for a terminal handler."""
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("simple", simple_test_workflow)

    async with server.contextmanager():
        await server._service.start_workflow(simple_test_workflow, "resolve-done-1")

        async def handler_completed() -> None:
            persisted = await memory_store.query(
                HandlerQuery(handler_id_in=["resolve-done-1"])
            )
            assert len(persisted) == 1
            assert persisted[0].status == "completed"

        await wait_for_passing(handler_completed, max_duration=2.0, interval=0.01)

        with pytest.raises(HandlerCompletedError):
            await server._service.resolve_handler("resolve-done-1")


@pytest.mark.asyncio
async def test_send_event_happy_path(
    memory_store: MemoryWorkflowStore, interactive_workflow: Workflow
) -> None:
    """send_event delivers an event and the workflow completes."""
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow(
        "interactive", interactive_workflow, additional_events=[ExternalEvent]
    )

    async with server.contextmanager():
        handler_data = await server._service.start_workflow(
            interactive_workflow, "send-hp-1"
        )

        # Wait for the workflow to emit the InputRequiredEvent (meaning it's waiting)
        run_id = handler_data.run_id
        assert run_id is not None

        async def handler_emitted_input_required() -> None:
            events = await memory_store.query_events(run_id)
            assert any(e.event.type == "RequestedExternalEvent" for e in events)

        await wait_for_passing(
            handler_emitted_input_required, max_duration=2.0, interval=0.01
        )

        await server._service.send_event("send-hp-1", ExternalEvent(response="pong"))

        async def handler_completed() -> None:
            persisted = await memory_store.query(
                HandlerQuery(handler_id_in=["send-hp-1"])
            )
            assert len(persisted) == 1
            assert persisted[0].status == "completed"

        await wait_for_passing(handler_completed, max_duration=2.0, interval=0.01)


@pytest.mark.asyncio
async def test_cancel_terminal_handler_without_purge(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    """cancel_handler on an already-completed handler without purge returns None."""
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("simple", simple_test_workflow)

    async with server.contextmanager():
        await server._service.start_workflow(simple_test_workflow, "cancel-term-1")

        async def handler_completed() -> None:
            persisted = await memory_store.query(
                HandlerQuery(handler_id_in=["cancel-term-1"])
            )
            assert len(persisted) == 1
            assert persisted[0].status == "completed"

        await wait_for_passing(handler_completed, max_duration=2.0, interval=0.01)

        result = await server._service.cancel_handler("cancel-term-1", purge=False)
        assert result is None

        # Handler should still exist unchanged
        persisted = await memory_store.query(
            HandlerQuery(handler_id_in=["cancel-term-1"])
        )
        assert len(persisted) == 1
        assert persisted[0].status == "completed"
