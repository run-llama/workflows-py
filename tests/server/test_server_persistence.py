# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
import asyncio
from typing import AsyncGenerator

import pytest

from tests.server.conftest import ExternalEvent, RequestedExternalEvent
from workflows.events import Event, InternalDispatchEvent
from workflows.server import WorkflowServer
from workflows import Context
from workflows.workflow import Workflow

from .memory_workflow_store import MemoryWorkflowStore
from .util import wait_for_passing


@pytest.fixture
def memory_store() -> MemoryWorkflowStore:
    return MemoryWorkflowStore()


@pytest.fixture
async def server_with_store(
    memory_store: MemoryWorkflowStore, interactive_workflow: Workflow
) -> AsyncGenerator[WorkflowServer, None]:
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("test", interactive_workflow)
    yield server
    await server._close()


@pytest.mark.asyncio
async def test_store_is_updated_on_step_completion(
    server_with_store: WorkflowServer, memory_store: MemoryWorkflowStore
) -> None:
    server = server_with_store

    # Start a workflow through internal runner to exercise persistence updates
    handler_id = "persist-1"
    handler = server._workflows["test"].run()
    server._run_workflow_handler(handler_id, "test", handler)
    handler = server._handlers[handler_id]

    # wait for first step to complete
    async def get_non_internal_event() -> Event:
        item = await server._handlers[handler_id].queue.get()
        if isinstance(item, InternalDispatchEvent):
            raise ValueError("Internal event received. Try again")
        return item

    item = await wait_for_passing(get_non_internal_event)
    assert isinstance(item, RequestedExternalEvent)

    # much sure its stored and running
    assert handler_id in memory_store.handlers
    persistent = memory_store.handlers[handler_id]
    assert persistent.workflow_name == "test"
    assert persistent.status == "running"
    assert isinstance(persistent.ctx, dict)

    # now, validate that the workflow completes when responding
    handler = server._handlers[handler_id].run_handler
    ctx = handler.ctx
    assert ctx is not None
    ctx.send_event(ExternalEvent(response="pong"))
    result = await handler
    # wait for event loop to resolve all tasks
    await server._handlers[handler_id].task
    await asyncio.sleep(0)  # let even loop resolve other waiters on the internal
    assert result == "received: pong"
    updated = memory_store.handlers[handler_id]
    assert updated.status == "completed"


@pytest.mark.asyncio
async def test_resume_active_handlers_across_server_restart(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    # Seed the store with a valid serialized Context using public API
    from workflows.server.abstract_workflow_store import PersistentHandler

    handler_id = "resume-1"
    initial_ctx = Context(simple_test_workflow).to_dict()
    await memory_store.update(
        PersistentHandler(
            handler_id=handler_id,
            workflow_name="test",
            status="running",
            ctx=initial_ctx,
        )
    )

    # Second server: same store and workflow, explicitly initialize active handlers
    server2 = WorkflowServer(workflow_store=memory_store)
    server2.add_workflow("test", simple_test_workflow)
    await server2._initialize_active_handlers()

    # The handler should be registered under the same id
    assert handler_id in server2._handlers

    # Await its completion through internal result future
    result = await server2._handlers[handler_id].run_handler
    assert result == "processed: default"


@pytest.mark.asyncio
async def test_store_is_updated_on_workflow_failure(
    memory_store: MemoryWorkflowStore, error_workflow: Workflow
) -> None:
    # Build a server with a failing workflow and the in-memory store
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("error", error_workflow)

    # Start a workflow through internal runner to exercise persistence updates
    handler_id = "fail-1"
    handler = server._workflows["error"].run()
    server._run_workflow_handler(handler_id, "error", handler)

    # Await the failure of the handler itself
    with pytest.raises(ValueError, match="Test error"):
        await handler

    # Ensure the background streaming task has completed and persisted status
    await server._handlers[handler_id].task
    await asyncio.sleep(0)

    # Verify store captured the failed status and has a context snapshot
    assert handler_id in memory_store.handlers
    persistent = memory_store.handlers[handler_id]
    assert persistent.workflow_name == "error"
    assert persistent.status == "failed"
    assert isinstance(persistent.ctx, dict)
    await server._close()


@pytest.mark.asyncio
async def test_startup_ignores_unregistered_workflows(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    from workflows.server.abstract_workflow_store import PersistentHandler

    # Unknown workflow entry
    await memory_store.update(
        PersistentHandler(
            handler_id="unknown-1", workflow_name="unknown", status="running", ctx={}
        )
    )

    # Known workflow entry to be resumed
    await memory_store.update(
        PersistentHandler(
            handler_id="known-1",
            workflow_name="test",
            status="running",
            ctx=Context(simple_test_workflow).to_dict(),
        )
    )

    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("test", simple_test_workflow)
    await server._initialize_active_handlers()

    assert "unknown-1" not in server._handlers
    assert "known-1" in server._handlers

    # Await completion of the resumed known handler
    result = await server._handlers["known-1"].run_handler
    assert result == "processed: default"
