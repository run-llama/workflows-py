# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for idle workflow suspend/resume functionality."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from llama_agents.server._server_runtime import ServerExternalAdapter, ServerRuntimeDecorator
from llama_agents.server.abstract_workflow_store import (
    HandlerQuery,
    PersistentHandler,
    Status,
)
from llama_agents.server.memory_workflow_store import MemoryWorkflowStore
from llama_agents.server.server import WorkflowServer
from server_test_fixtures import async_yield, wait_for_passing
from workflows import Context, Workflow, step
from workflows.context.context_types import SerializedContext
from workflows.events import HumanResponseEvent, StartEvent, StopEvent
from workflows.runtime.types.ticks import TickAddEvent


class WaitableExternalEvent(HumanResponseEvent):
    """Event sent from external sources."""

    response: str


class WaitingWorkflow(Workflow):
    """Workflow that uses ctx.wait_for_event() to properly become idle."""

    @step
    async def start_and_wait(self, ctx: Context, ev: StartEvent) -> StopEvent:
        external = await ctx.wait_for_event(WaitableExternalEvent)
        return StopEvent(result=f"received: {external.response}")


def get_adapter(server: WorkflowServer, handler_id: str) -> ServerExternalAdapter:
    """Get the active adapter for a handler, asserting it exists."""
    adapter = server._service.get_adapter_for_handler(handler_id)
    assert adapter is not None, f"Adapter for {handler_id} not found"
    return adapter


def get_decorator(server: WorkflowServer, workflow_name: str) -> ServerRuntimeDecorator:
    """Get the runtime decorator for a workflow."""
    decorator = server._service.get_decorator(workflow_name)
    assert decorator is not None
    return decorator


def assert_adapter_exists(server: WorkflowServer, handler_id: str) -> None:
    """Assert a handler has an active adapter."""
    get_adapter(server, handler_id)


def get_idle_since(server: WorkflowServer, handler_id: str) -> datetime | None:
    """Get idle_since from the runtime metadata for a handler."""
    for decorator in server._service._decorators.values():
        meta = decorator.get_run_metadata(handler_id)
        if meta is not None:
            return meta.idle_since
    return None


def make_server(
    memory_store: MemoryWorkflowStore,
    waiting_workflow: Workflow,
    idle_release_timeout: timedelta | None,
    *,
    persistence_backoff: list[float] | None = None,
) -> WorkflowServer:
    if persistence_backoff is None:
        server = WorkflowServer(
            workflow_store=memory_store,
            idle_release_timeout=idle_release_timeout,
        )
    else:
        server = WorkflowServer(
            workflow_store=memory_store,
            idle_release_timeout=idle_release_timeout,
            persistence_backoff=persistence_backoff,
        )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )
    return server


async def start_waiting_handler(
    server: WorkflowServer, handler_id: str
) -> ServerExternalAdapter:
    """Start a workflow and wait for it to become idle."""
    adapter = await server._service.start_workflow(
        workflow_name="test",
        workflow=server._service._workflows["test"],
        handler_id=handler_id,
    )

    async def is_idle() -> None:
        idle_since = get_idle_since(server, handler_id)
        assert idle_since is not None, "Workflow not yet idle"

    await wait_for_passing(is_idle, interval=0.01, max_duration=2.0)
    return adapter


async def seed_persistent_handler(
    store: MemoryWorkflowStore,
    handler_id: str,
    *,
    idle_since: datetime | None,
    ctx: dict[str, object],
    status: Status = "running",
) -> None:
    await store.update(
        PersistentHandler(
            handler_id=handler_id,
            workflow_name="test",
            status=status,
            idle_since=idle_since,
            ctx=ctx,
        )
    )


@pytest.fixture
def memory_store() -> MemoryWorkflowStore:
    return MemoryWorkflowStore()


@pytest.fixture
def waiting_workflow() -> Workflow:
    return WaitingWorkflow()


# Use short real-time timeouts for idle tests
FAST_IDLE_TIMEOUT = timedelta(milliseconds=50)


@pytest.mark.asyncio
async def test_is_idle_query_filter_memory_store() -> None:
    """Test that is_idle filter works in MemoryWorkflowStore."""
    store = MemoryWorkflowStore()
    now = datetime.now(timezone.utc)

    await seed_persistent_handler(
        store, "idle-1", idle_since=now - timedelta(minutes=5), ctx={}
    )
    await seed_persistent_handler(store, "active-1", idle_since=None, ctx={})
    await seed_persistent_handler(
        store, "idle-2", idle_since=now - timedelta(seconds=10), ctx={}
    )

    idle_results = await store.query(HandlerQuery(is_idle=True))
    assert len(idle_results) == 2
    assert {r.handler_id for r in idle_results} == {"idle-1", "idle-2"}

    active_results = await store.query(HandlerQuery(is_idle=False))
    assert len(active_results) == 1
    assert active_results[0].handler_id == "active-1"

    all_results = await store.query(HandlerQuery())
    assert len(all_results) == 3


@pytest.mark.asyncio
async def test_workflow_becomes_idle_and_is_suspended(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that a workflow becomes idle after waiting and gets suspended after timeout."""
    server = make_server(memory_store, waiting_workflow, FAST_IDLE_TIMEOUT)

    async with server.contextmanager():
        handler_id = "release-test-1"
        adapter = await start_waiting_handler(server, handler_id)

        # Wait for suspension (real-time timer will fire after 50ms)
        async def check_suspended() -> None:
            assert adapter.is_suspended, "Adapter should be suspended"

        await wait_for_passing(check_suspended, interval=0.01, max_duration=3.0)

        # Should still exist in the store with status "running"
        persisted = await memory_store.query(
            HandlerQuery(handler_id_in=[handler_id])
        )
        assert len(persisted) == 1
        assert persisted[0].status == "running"
        assert persisted[0].idle_since is not None


@pytest.mark.asyncio
async def test_suspended_workflow_resumes_on_event(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that a suspended workflow is resumed when an event is sent."""
    server = make_server(memory_store, waiting_workflow, FAST_IDLE_TIMEOUT)

    async with server.contextmanager():
        handler_id = "reload-test-1"
        adapter = await start_waiting_handler(server, handler_id)

        # Wait for suspension
        async def check_suspended() -> None:
            assert adapter.is_suspended

        await wait_for_passing(check_suspended, interval=0.01, max_duration=3.0)

        # Send an event - this should trigger resume
        await adapter.send_event(
            TickAddEvent(event=WaitableExternalEvent(response="hello"), step_name=None)
        )

        # The adapter should no longer be suspended
        assert not adapter.is_suspended

        # Wait for the workflow to complete
        result = await adapter.get_result()
        assert result.result == "received: hello"


@pytest.mark.asyncio
async def test_idle_handlers_not_resumed_on_server_start(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that idle handlers are not loaded into memory on server start."""
    idle_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = SerializedContext().model_dump(mode="python")

    # Seed an idle handler
    await seed_persistent_handler(
        memory_store, "idle-on-start-1", idle_since=idle_time, ctx=ctx
    )
    # Seed an active (non-idle) handler
    await seed_persistent_handler(
        memory_store, "active-on-start-1", idle_since=None, ctx=ctx
    )

    server = make_server(memory_store, waiting_workflow, timedelta(minutes=5))

    async with server.contextmanager():
        # The idle handler should NOT be in memory (startup skips idle handlers)
        assert server._service.get_adapter_for_handler("idle-on-start-1") is None

        # The active handler SHOULD be in memory
        assert_adapter_exists(server, "active-on-start-1")

        # The idle handler should still exist in the store
        persisted = await memory_store.query(
            HandlerQuery(handler_id_in=["idle-on-start-1"])
        )
        assert len(persisted) == 1
        assert persisted[0].status == "running"


@pytest.mark.asyncio
async def test_idle_release_disabled_when_timeout_none(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that no timer is started when idle_release_timeout is None."""
    server = make_server(memory_store, waiting_workflow, None)

    async with server.contextmanager():
        handler_id = "no-release-test"
        adapter = await start_waiting_handler(server, handler_id)

        # Give it some real time to ensure no suspension happens
        await asyncio.sleep(0.1)

        # Timer should not exist since idle_release_timeout is None
        assert adapter._idle_release_timer is None
        assert not adapter.is_suspended


@pytest.mark.asyncio
async def test_idle_release_cancels_inner_runtime(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Idle release should stop the workflow inner runtime."""
    server = make_server(memory_store, waiting_workflow, FAST_IDLE_TIMEOUT)

    async with server.contextmanager():
        handler_id = "runtime-cancel-test"
        adapter = await start_waiting_handler(server, handler_id)

        # Wait for suspension
        async def check_suspended() -> None:
            assert adapter.is_suspended

        await wait_for_passing(check_suspended, interval=0.01, max_duration=3.0)

        # The adapter should be suspended (inner run cancelled)
        assert adapter.is_suspended


@pytest.mark.asyncio
async def test_try_reload_resumes_persisted_handler(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that try_reload_handler can load a handler from persistence."""
    server = make_server(memory_store, waiting_workflow, timedelta(minutes=5))

    # Seed a persisted idle handler
    idle_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = SerializedContext().model_dump(mode="python")
    await seed_persistent_handler(
        memory_store, "reload-test", idle_since=idle_time, ctx=ctx
    )

    async with server.contextmanager():
        # Handler should not be in memory (idle handlers skipped at startup)
        assert server._service.get_adapter_for_handler("reload-test") is None

        # Reload it
        adapter, persisted = await server._service.try_reload_handler("reload-test")
        assert adapter is not None
        assert persisted is not None
        assert persisted.status == "running"

        # Should now be active
        assert_adapter_exists(server, "reload-test")


@pytest.mark.asyncio
async def test_try_reload_is_singleton_under_concurrency(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Concurrent reload requests should only create one handler."""
    server = make_server(memory_store, waiting_workflow, timedelta(minutes=5))

    # Seed a persisted idle handler
    idle_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = SerializedContext().model_dump(mode="python")
    handler_id = "concurrent-reload-test"
    await seed_persistent_handler(
        memory_store, handler_id, idle_since=idle_time, ctx=ctx
    )

    async with server.contextmanager():
        assert server._service.get_adapter_for_handler(handler_id) is None

        results: list[ServerExternalAdapter | None] = []

        async def reload_and_track() -> None:
            adapter, _ = await server._service.try_reload_handler(handler_id)
            results.append(adapter)

        # Fire 5 concurrent reload requests
        await asyncio.gather(*[reload_and_track() for _ in range(5)])

        # At least one should succeed
        successful = [r for r in results if r is not None]
        assert len(successful) >= 1

        # All successful results should be the same adapter instance
        unique_adapters = set(id(a) for a in successful)
        assert len(unique_adapters) == 1, (
            f"{len(unique_adapters)} different adapter instances were created "
            f"by concurrent reloads. Only one should be created."
        )


@pytest.mark.asyncio
async def test_send_event_via_http_resumes_suspended_workflow(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that posting an event via HTTP can resume a suspended workflow and complete it."""
    server = make_server(memory_store, waiting_workflow, FAST_IDLE_TIMEOUT)

    async with server.contextmanager():
        transport = ASGITransport(app=server.app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Start a workflow via HTTP
            response = await client.post("/workflows/test/run-nowait", json={})
            assert response.status_code == 200
            handler_id = response.json()["handler_id"]

            # Wait for workflow to become idle then suspended
            adapter = server._service.get_adapter_for_handler(handler_id)
            assert adapter is not None

            async def check_suspended() -> None:
                assert adapter.is_suspended

            await wait_for_passing(check_suspended, interval=0.01, max_duration=3.0)

            # Send event via HTTP to resume
            response = await client.post(
                f"/events/{handler_id}",
                json={
                    "event": {
                        "type": "WaitableExternalEvent",
                        "value": {"response": "hello-http"},
                    },
                },
            )
            assert response.status_code == 200

            # Verify workflow completed
            async def check_completed() -> None:
                resp = await client.get(f"/handlers/{handler_id}")
                data = resp.json()
                assert data["status"] in ("completed", "failed"), f"status={data['status']}"
                assert data["status"] == "completed"

            await wait_for_passing(check_completed, interval=0.05, max_duration=5.0)


@pytest.mark.asyncio
async def test_suspend_persists_checkpoint_before_cancelling(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Verify that suspend does a final checkpoint before cancelling the inner run."""
    server = make_server(memory_store, waiting_workflow, FAST_IDLE_TIMEOUT)

    async with server.contextmanager():
        handler_id = "checkpoint-before-suspend"
        adapter = await start_waiting_handler(server, handler_id)

        # Wait for suspension
        async def check_suspended() -> None:
            assert adapter.is_suspended

        await wait_for_passing(check_suspended, interval=0.01, max_duration=3.0)

        # Verify the store has a checkpoint
        persisted = await memory_store.query(
            HandlerQuery(handler_id_in=[handler_id])
        )
        assert len(persisted) == 1
        assert persisted[0].status == "running"
        # Context should be saved
        assert isinstance(persisted[0].ctx, dict)
        assert len(persisted[0].ctx) > 0
