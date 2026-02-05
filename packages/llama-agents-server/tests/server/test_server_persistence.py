# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from llama_agents.server import WorkflowServer
from llama_agents.server.abstract_workflow_store import (
    HandlerQuery,
    PersistentHandler,
)
from llama_agents.server.memory_workflow_store import MemoryWorkflowStore
from server_test_fixtures import (  # type: ignore[import]
    ExternalEvent,
    RequestedExternalEvent,
    wait_for_passing,
)
from workflows.context.context_types import SerializedContext
from workflows.events import Event, InternalDispatchEvent, StopEvent
from workflows.runtime.types.ticks import TickAddEvent
from workflows.workflow import Workflow


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

    handler_id = "persist-1"
    adapter = await server._service.start_workflow(
        workflow_name="test",
        workflow=server._service._workflows["test"],
        handler_id=handler_id,
    )

    # Wait for the first step to complete by streaming published events
    # and looking for the RequestedExternalEvent (non-internal)
    async def get_non_internal_event() -> Event:
        async for event in adapter.stream_published_events():
            if isinstance(event, InternalDispatchEvent):
                continue
            return event
        raise ValueError("Stream ended without a non-internal event")

    item = await wait_for_passing(get_non_internal_event)
    assert isinstance(item, RequestedExternalEvent)

    # Verify the handler is stored and running
    persistent_list = await memory_store.query(HandlerQuery(handler_id_in=[handler_id]))
    assert persistent_list
    persistent = persistent_list[0]
    assert persistent.workflow_name == "test"
    assert persistent.status == "running"
    assert isinstance(persistent.ctx, dict)

    # Send a response event to complete the workflow
    await adapter.send_event(
        TickAddEvent(event=ExternalEvent(response="pong"), step_name=None)
    )

    # Wait for the workflow to complete
    result = await adapter.get_result()
    assert result.result == "received: pong"

    # Wait for the completion callback to persist status
    await asyncio.sleep(0.1)

    updated = memory_store.handlers[handler_id]
    assert updated.status == "completed"


@pytest.mark.asyncio
async def test_resume_active_handlers_across_server_restart(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    # Seed the store with a valid serialized Context using public API
    handler_id = "resume-1"
    initial_ctx = SerializedContext().model_dump(mode="python")
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
    async with server2.contextmanager():
        # The handler should be registered as an active adapter
        adapter = server2._service.get_adapter_for_handler(handler_id)
        assert adapter is not None

        # Await its completion through the adapter
        result = await adapter.get_result()
        assert result.result == "processed: default"


@pytest.mark.asyncio
async def test_startup_marks_invalid_persisted_context_as_failed(
    memory_store: MemoryWorkflowStore, simple_test_workflow: Workflow
) -> None:
    """Server should not crash on invalid persisted context; it should mark it failed."""
    # Seed an invalid context payload that will fail Context.from_dict
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
        assert server._service.get_adapter_for_handler(handler_id) is None

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
        handler_id = "fail-1"
        adapter = await server._service.start_workflow(
            workflow_name="error",
            workflow=server._service._workflows["error"],
            handler_id=handler_id,
        )

        # Await the failure of the workflow via the adapter
        with pytest.raises(ValueError, match="Test error"):
            await adapter.get_result()

        # Wait for the completion callback to persist the failed status
        await asyncio.sleep(0.1)

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
            ctx=SerializedContext().model_dump(mode="python"),
        )
    )

    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("test", simple_test_workflow)
    async with server.contextmanager():
        assert server._service.get_adapter_for_handler("unknown-1") is None
        adapter = server._service.get_adapter_for_handler("known-1")
        assert adapter is not None

        # Await completion of the resumed known handler
        result = await adapter.get_result()
        assert result.result == "processed: default"


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
    store = MemoryWorkflowStore()

    # Configure server with custom backoff (shorter for testing)
    server = WorkflowServer(
        workflow_store=store,
        persistence_backoff=[0.0, 0.0],  # Very short backoffs for testing
    )
    server.add_workflow("test", simple_test_workflow)

    async with server.contextmanager():
        handler_id = "retry-test"
        adapter = await server._service.start_workflow(
            workflow_name="test",
            workflow=server._service._workflows["test"],
            handler_id=handler_id,
        )

        # Apply the monkeypatch AFTER start_workflow so the initial checkpoint
        # succeeds, but subsequent checkpoints (after_step_completed, _on_complete) fail.
        attempts = patch_store_update_to_fail(monkeypatch, store, fail_count=2)

        # Wait for workflow completion
        result = await adapter.get_result()
        assert result.result == "processed: default"

        # Wait for the completion callback to persist
        await asyncio.sleep(0.1)

        # Verify that retries occurred and eventually succeeded
        assert attempts["count"] >= 3  # 2 failures + 1 success
        persistent_list = await store.query(HandlerQuery(handler_id_in=[handler_id]))
        assert persistent_list
        persistent = persistent_list[0]
        assert persistent.status == "completed"


@pytest.mark.asyncio
async def test_workflow_cancelled_after_all_retries_fail(
    streaming_workflow: Workflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that workflow is cancelled when all persistence retries are exhausted.

    In the new architecture, the initial checkpoint happens in start_workflow.
    If it fails, start_workflow raises directly.
    """
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
        handler_id = "cancel-test"

        # Should raise if persistence fails on initial checkpoint
        with pytest.raises(Exception):
            await server._service.start_workflow(
                workflow_name="test",
                workflow=server._service._workflows["test"],
                handler_id=handler_id,
            )

        # Verify retry attempts occurred
        assert attempts["count"] >= 1
        # The handler should not remain registered as active
        # (since start_workflow raised, the adapter may or may not have been cleaned up,
        # but the handler should not be usable)
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

            # Wait for the completion callback to persist the handler
            async def handler_is_completed() -> None:
                persisted_list = await memory_store.query(
                    HandlerQuery(handler_id_in=[handler_id])
                )
                assert persisted_list
                assert persisted_list[0].status == "completed"

            await wait_for_passing(handler_is_completed)

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

            # Wait for persistence to complete
            async def handler_completed_after_second_run() -> None:
                persisted_list = await memory_store.query(
                    HandlerQuery(handler_id_in=[handler_id])
                )
                assert persisted_list
                assert persisted_list[0].status == "completed"

            await wait_for_passing(handler_completed_after_second_run)

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
            ctx=SerializedContext().model_dump(mode="python"),
        )
    )

    # Start a server with the same store and workflow registered
    server = WorkflowServer(workflow_store=memory_store)
    server.add_workflow("test", simple_test_workflow)

    async with server.contextmanager():
        # Ensure the handler is not registered in runtime memory
        assert server._service.get_adapter_for_handler(handler_id) is None

        # But the API should still return the persisted result
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/handlers/{handler_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["result"]["value"]["result"] == "processed: default"
