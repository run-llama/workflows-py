# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from tests.server.conftest import ExternalEvent, RequestedExternalEvent
from workflows.server import WorkflowServer
from workflows import Context
from workflows.workflow import Workflow

from .memory_workflow_store import MemoryWorkflowStore


@pytest.fixture
def memory_store() -> MemoryWorkflowStore:
    return MemoryWorkflowStore()


@pytest_asyncio.fixture
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
    item = await server._handlers[handler_id].queue.get()
    assert isinstance(item, RequestedExternalEvent)
    assert handler_id in memory_store.handlers
    persistent = memory_store.handlers[handler_id]
    assert persistent.workflow_name == "test"
    assert persistent.completed is False
    assert isinstance(persistent.ctx, dict)
    handler = server._handlers[handler_id].handler
    ctx = handler.ctx

    # validate that the workflow completes
    assert ctx is not None
    ctx.send_event(ExternalEvent(response="pong"))
    result = await handler
    assert result == "received: pong"
    updated = memory_store.handlers[handler_id]
    assert updated.completed is True


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
            completed=False,
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
    result = await server2._handlers[handler_id].handler
    assert result == "processed: default"


@pytest.mark.asyncio
async def test_startup_ignores_unregistered_workflows(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    from workflows.server.abstract_workflow_store import PersistentHandler

    # Unknown workflow entry
    await memory_store.update(
        PersistentHandler(
            handler_id="unknown-1", workflow_name="unknown", completed=False, ctx={}
        )
    )

    # Known workflow entry to be resumed
    await memory_store.update(
        PersistentHandler(
            handler_id="known-1",
            workflow_name="test",
            completed=False,
            ctx=Context(simple_test_workflow).to_dict(),
        )
    )

    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("test", simple_test_workflow)
    await server._initialize_active_handlers()

    assert "unknown-1" not in server._handlers
    assert "known-1" in server._handlers

    # Await completion of the resumed known handler
    result = await server._handlers["known-1"].handler
    assert result == "processed: default"
