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
from workflows.server.abstract_workflow_store import PersistentHandler

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
    async with server.contextmanager():
        yield server


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
    async with server2.contextmanager():  # start and stop it
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
    async with server.contextmanager():
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


@pytest.mark.asyncio
async def test_startup_ignores_unregistered_workflows(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
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
    async with server.contextmanager():  # start and stop it
        assert "unknown-1" not in server._handlers
        assert "known-1" in server._handlers

        # Await completion of the resumed known handler
        result = await server._handlers["known-1"].run_handler
        assert result == "processed: default"


def patch_store_update_to_fail(
    monkeypatch: pytest.MonkeyPatch, store: MemoryWorkflowStore, fail_count: int
) -> dict[str, int]:
    """Monkeypatch `store.update` to fail `fail_count` times, then delegate to original.

    Returns a dict with a mutable `count` key to inspect attempt count.
    """
    attempts: dict[str, int] = {"count": 0}
    original_update = store.update

    async def wrapped(handler: PersistentHandler) -> None:
        attempts["count"] += 1
        if attempts["count"] <= fail_count:
            raise Exception(
                f"Simulated persistence failure (attempt {attempts['count']})"
            )
        await original_update(handler)

    monkeypatch.setattr(store, "update", wrapped)
    return attempts


@pytest.mark.asyncio
async def test_persistence_retries_on_failure(
    simple_test_workflow: Workflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that persistence operations are retried according to backoff configuration."""
    # Create a store that fails twice then succeeds
    store = MemoryWorkflowStore()
    attempts = patch_store_update_to_fail(monkeypatch, store, fail_count=2)

    # Configure server with custom backoff (shorter for testing)
    server = WorkflowServer(
        workflow_store=store,
        persistence_backoff=[0.0, 0.0],  # Very short backoffs for testing
    )
    server.add_workflow("test", simple_test_workflow)

    async with server.contextmanager():
        # Start a workflow to trigger persistence
        handler_id = "retry-test"
        handler = server._workflows["test"].run()
        server._run_workflow_handler(handler_id, "test", handler)

        # Wait for workflow completion
        result = await server._handlers[handler_id].run_handler
        assert result == "processed: default"

        # Wait for background streaming task to complete, ignoring its expected exception
        task = server._handlers[handler_id].task
        try:
            await task
        except Exception:
            pass
        await asyncio.sleep(0)

        # Verify that retries occurred and eventually succeeded
        # There can be multiple checkpoints (initial running, step completion, final completion)
        # so attempts will be >= initial + retries
        assert attempts["count"] >= 3  # Initial + 2 retries
        assert handler_id in store.handlers
        persistent = store.handlers[handler_id]
        assert persistent.status == "completed"


@pytest.mark.asyncio
async def test_workflow_cancelled_after_all_retries_fail(
    streaming_workflow: Workflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that workflow is cancelled when all persistence retries are exhausted."""
    # Create a store that always fails
    store = MemoryWorkflowStore()
    attempts = patch_store_update_to_fail(
        monkeypatch, store, fail_count=10
    )  # Fail more than backoff attempts

    # Configure server with custom backoff (shorter for testing)
    server = WorkflowServer(
        workflow_store=store,
        persistence_backoff=[0.0, 0.0],  # 2 retries, very short backoffs
    )
    server.add_workflow("test", streaming_workflow)
    async with server.contextmanager():
        try:
            # Start a workflow to trigger persistence
            handler_id = "cancel-test"
            handler = server._workflows["test"].run()
            server._run_workflow_handler(handler_id, "test", handler)

            # The handler should be cancelled due to persistence failures
            with pytest.raises(asyncio.CancelledError):
                await server._handlers[handler_id].run_handler

            # Wait for background streaming task to complete; it raises due to failed checkpoint
            task = server._handlers[handler_id].task
            with pytest.raises(Exception):
                await task
            await asyncio.sleep(0)

            # Verify that all retry attempts were made
            assert attempts["count"] == 3  # Initial + 2 retries
            # Handler should not be successfully stored due to persistent failures
            assert handler_id not in store.handlers
        finally:  # clean up log noise. Wait for underlying cancelled workflows to fully resolve
            tasks = [
                handler.run_handler._run_task
                for handler in server._handlers.values()
                if handler.run_handler._run_task
            ]
            for task in tasks:
                try:
                    task.cancel()
                except Exception:
                    pass
            await asyncio.sleep(0)
