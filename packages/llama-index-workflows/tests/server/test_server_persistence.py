# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
import asyncio
import pytest
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport


from .conftest import ExternalEvent, RequestedExternalEvent
from workflows.events import Event, InternalDispatchEvent, StopEvent
from workflows.server import WorkflowServer
from workflows import Context
from workflows.workflow import Workflow
from workflows.server.abstract_workflow_store import HandlerQuery, PersistentHandler

from workflows.server.memory_workflow_store import MemoryWorkflowStore
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


@pytest.fixture
async def server_with_store_and_simple_workflow(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> AsyncGenerator[WorkflowServer, None]:
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("test", simple_test_workflow)
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
    await server._run_workflow_handler(handler_id, "test", handler)
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
    persistent_list = await memory_store.query(HandlerQuery(handler_id_in=[handler_id]))
    assert persistent_list
    persistent = persistent_list[0]
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
    task = server._handlers[handler_id].task
    assert task is not None
    await task
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
async def test_startup_marks_invalid_persisted_context_as_failed(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    """Server should not crash on invalid persisted context; it should mark it failed."""
    # Seed an invalid context payload that will fail Context.from_dict
    # Make the context structurally valid but with an invalid streaming_queue JSON
    invalid_ctx = {
        "state": {},
        "streaming_queue": "[]",
        "queues": {"process": "not-deserializable-as-a-queue"},
        "event_buffers": {},
        "in_progress": {},
        "accepted_events": [],
        "broker_log": [],
        "is_running": True,
        "waiting_ids": [],
    }

    handler_id = "bad-ctx-1"
    await memory_store.update(
        PersistentHandler(
            handler_id=handler_id,
            workflow_name="test",
            status="running",
            ctx=invalid_ctx,
        )
    )

    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("test", simple_test_workflow)
    async with server.contextmanager():
        # Invalid handler should not be registered
        assert handler_id not in server._handlers

    # After startup attempt, it should be marked as failed in the store
    persisted = memory_store.handlers[handler_id]
    assert persisted.status == "failed"


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
        await server._run_workflow_handler(handler_id, "error", handler)

        # Await the failure of the handler itself
        with pytest.raises(ValueError, match="Test error"):
            await handler

        # Ensure the background streaming task has completed and persisted status
        task = server._handlers[handler_id].task
        assert task is not None
        await task
        await asyncio.sleep(0)

        # Verify store captured the failed status and has a context snapshot
        persistent_list = await memory_store.query(
            HandlerQuery(handler_id_in=[handler_id])
        )
        assert persistent_list
        persistent = persistent_list[0]
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
        await server._run_workflow_handler(handler_id, "test", handler)

        # Wait for workflow completion
        result = await server._handlers[handler_id].run_handler
        assert result == "processed: default"

        # Wait for background streaming task to complete, ignoring its expected exception
        task = server._handlers[handler_id].task
        try:
            assert task is not None
            await task
        except Exception:
            pass
        await asyncio.sleep(0)

        # Verify that retries occurred and eventually succeeded
        # There can be multiple checkpoints (initial running, step completion, final completion)
        # so attempts will be >= initial + retries
        assert attempts["count"] >= 3  # Initial + 2 retries
        persistent_list = await store.query(HandlerQuery(handler_id_in=[handler_id]))
        assert persistent_list
        persistent = persistent_list[0]
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
        # Start a workflow to trigger persistence
        handler_id = "cancel-test"
        handler = server._workflows["test"].run()

        # Should raise HTTPException if persistence fails on initial checkpoint
        with pytest.raises(Exception):
            await server._run_workflow_handler(handler_id, "test", handler)

        # Verify retry attempts and no registration/persistence on failure
        assert attempts["count"] == 3  # Initial + 2 retries
        assert handler_id not in server._handlers
        persistent_list = await store.query(HandlerQuery(handler_id_in=[handler_id]))
        assert not persistent_list


@pytest.mark.asyncio
async def test_resume_across_runs(
    memory_store: MemoryWorkflowStore, cumulative_workflow: Workflow
) -> None:
    """Test that workflow context accumulates data across multiple runs using handler_id continuation."""
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("cumulative", cumulative_workflow)

    async with server.contextmanager():
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First run - should start with count=0, increment by 5
            response = await client.post(
                "/workflows/cumulative/run", json={"start_event": {"increment": 5}}
            )
            assert response.status_code == 200
            resp_data = response.json()
            assert resp_data["result"]["value"]["result"] == "count: 5, runs: 1"

            # Get the handler id for that run
            handler_id = resp_data["handler_id"]

            # Wait for the handler to be fully persisted as completed
            await asyncio.sleep(0.1)

            # Verify it's persisted in the store as completed
            persisted_list = await memory_store.query(
                HandlerQuery(handler_id_in=[handler_id])
            )
            assert persisted_list
            persisted = persisted_list[0]
            assert persisted.status == "completed"

            # Second run - should start with count=5, increment by 3
            response2 = await client.post(
                "/workflows/cumulative/run",
                json={"handler_id": handler_id, "start_event": {"increment": 3}},
            )
            assert response2.status_code == 200
            resp_data2 = response2.json()
            assert resp_data2["result"]["value"]["result"] == "count: 8, runs: 2"

            # Verify the handler id is the same
            assert resp_data2["handler_id"] == handler_id

            # Wait for the handler to be fully persisted as completed
            await asyncio.sleep(0.1)

            # Verify memory store has only one handler
            assert len(memory_store.handlers) == 1
            assert memory_store.handlers[handler_id].status == "completed"


@pytest.mark.asyncio
async def test_result_for_completed_persisted_handler_without_runtime_registration(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    """A completed handler persisted in the store but not registered in memory should still return its result."""
    handler_id = "store-completed-1"

    # Seed a completed handler directly in the store (server won't load completed handlers at startup)
    await memory_store.update(
        PersistentHandler(
            handler_id=handler_id,
            workflow_name="test",
            status="completed",
            result=StopEvent(result="processed: default"),
            ctx=Context(simple_test_workflow).to_dict(),
        )
    )

    # Start a server with the same store and workflow registered
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("test", simple_test_workflow)

    async with server.contextmanager():
        # Ensure the handler is not registered in runtime memory
        assert handler_id not in server._handlers

        # But the API should still return the persisted result
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/handlers/{handler_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["result"]["value"]["result"] == "processed: default"
