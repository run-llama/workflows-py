# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
"""Tests for idle workflow release and reload functionality."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest
from workflows import Context, Workflow, step
from workflows.events import HumanResponseEvent, StartEvent, StopEvent
from workflows.server import WorkflowServer
from workflows.server.abstract_workflow_store import HandlerQuery, PersistentHandler
from workflows.server.memory_workflow_store import MemoryWorkflowStore

from .util import wait_for_passing  # type: ignore[import]


class WaitableExternalEvent(HumanResponseEvent):
    """Event sent from external sources."""

    response: str


class WaitingWorkflow(Workflow):
    """Workflow that uses ctx.wait_for_event() to properly become idle."""

    @step
    async def start_and_wait(self, ctx: Context, ev: StartEvent) -> StopEvent:
        # Use ctx.wait_for_event() to create a waiter - this makes the workflow idle
        external = await ctx.wait_for_event(WaitableExternalEvent)
        return StopEvent(result=f"received: {external.response}")


@pytest.fixture
def memory_store() -> MemoryWorkflowStore:
    return MemoryWorkflowStore()


@pytest.fixture
def waiting_workflow() -> Workflow:
    return WaitingWorkflow()


@pytest.fixture
async def server_with_idle_release(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> AsyncGenerator[WorkflowServer, None]:
    """Server with a short idle release timeout for testing."""
    server = WorkflowServer(
        workflow_store=memory_store,
        idle_release_timeout=timedelta(milliseconds=50),
    )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )
    async with server.contextmanager():
        yield server


@pytest.mark.asyncio
async def test_is_idle_query_filter_memory_store() -> None:
    """Test that is_idle filter works in MemoryWorkflowStore."""
    store = MemoryWorkflowStore()

    now = datetime.now(timezone.utc)

    # Handler that is idle (has idle_since set)
    await store.update(
        PersistentHandler(
            handler_id="idle-1",
            workflow_name="test",
            status="running",
            idle_since=now - timedelta(minutes=5),
            ctx={},
        )
    )

    # Handler that is not idle (no idle_since)
    await store.update(
        PersistentHandler(
            handler_id="active-1",
            workflow_name="test",
            status="running",
            idle_since=None,
            ctx={},
        )
    )

    # Another idle handler
    await store.update(
        PersistentHandler(
            handler_id="idle-2",
            workflow_name="test",
            status="running",
            idle_since=now - timedelta(seconds=10),
            ctx={},
        )
    )

    # Query for idle handlers
    idle_results = await store.query(HandlerQuery(is_idle=True))
    assert len(idle_results) == 2
    assert {r.handler_id for r in idle_results} == {"idle-1", "idle-2"}

    # Query for non-idle handlers
    active_results = await store.query(HandlerQuery(is_idle=False))
    assert len(active_results) == 1
    assert active_results[0].handler_id == "active-1"

    # Query without filter returns all
    all_results = await store.query(HandlerQuery())
    assert len(all_results) == 3


@pytest.mark.asyncio
async def test_workflow_becomes_idle_and_is_released(
    server_with_idle_release: WorkflowServer, memory_store: MemoryWorkflowStore
) -> None:
    """Test that a workflow becomes idle after a step and can be released."""
    server = server_with_idle_release

    # Start a workflow
    handler_id = "release-test-1"
    handler = server._workflows["test"].run()
    await server._run_workflow_handler(handler_id, "test", handler)

    # Wait for the WorkflowIdleEvent to be processed (workflow becomes idle when waiting)
    async def check_idle() -> None:
        wrapper = server._handlers.get(handler_id)
        if wrapper is None or wrapper.idle_since is None:
            raise ValueError("Not idle yet")

    await wait_for_passing(check_idle)

    # The handler should now be idle (waiting for external event)
    wrapper = server._handlers[handler_id]
    assert wrapper.idle_since is not None

    # Wait for the idle release mechanism to kick in
    await asyncio.sleep(0.15)  # Wait longer than the release timeout + check interval

    # The handler should be released from memory
    assert handler_id not in server._handlers

    # But should still exist in the store with status "running"
    persisted = await memory_store.query(HandlerQuery(handler_id_in=[handler_id]))
    assert len(persisted) == 1
    assert persisted[0].status == "running"
    assert persisted[0].idle_since is not None


@pytest.mark.asyncio
async def test_released_workflow_is_reloaded_on_event(
    server_with_idle_release: WorkflowServer, memory_store: MemoryWorkflowStore
) -> None:
    """Test that a released workflow is reloaded when an event is sent."""
    server = server_with_idle_release

    # Start a workflow
    handler_id = "reload-test-1"
    handler = server._workflows["test"].run()
    await server._run_workflow_handler(handler_id, "test", handler)

    # Wait for the workflow to become idle
    async def check_idle() -> None:
        wrapper = server._handlers.get(handler_id)
        if wrapper is None or wrapper.idle_since is None:
            raise ValueError("Not idle yet")

    await wait_for_passing(check_idle)

    # Wait for release
    await asyncio.sleep(0.15)
    assert handler_id not in server._handlers

    # Now reload by using _try_reload_handler
    reloaded = await server._try_reload_handler(handler_id)
    assert reloaded is not None
    assert handler_id in server._handlers

    # Send the event to complete the workflow
    ctx = reloaded.run_handler.ctx
    assert ctx is not None
    ctx.send_event(WaitableExternalEvent(response="hello"))

    result = await reloaded.run_handler
    assert result == "received: hello"


@pytest.mark.asyncio
async def test_idle_release_restores_idle_since_on_reload(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that idle_since is preserved when reloading via _try_reload_handler."""
    server = WorkflowServer(
        workflow_store=memory_store,
        idle_release_timeout=timedelta(minutes=5),  # Long timeout
    )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )

    # Seed a persisted handler with idle_since
    idle_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = Context(waiting_workflow).to_dict()
    await memory_store.update(
        PersistentHandler(
            handler_id="idle-restore-1",
            workflow_name="test",
            status="running",
            idle_since=idle_time,
            ctx=ctx,
        )
    )

    async with server.contextmanager():
        # Idle handler should NOT be in memory on startup
        assert "idle-restore-1" not in server._handlers

        # Reload it (simulating an event arriving)
        wrapper = await server._try_reload_handler("idle-restore-1")
        assert wrapper is not None
        assert wrapper.idle_since == idle_time


@pytest.mark.asyncio
async def test_active_stream_consumer_prevents_release(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that a workflow with an active stream consumer is not released."""
    server = WorkflowServer(
        workflow_store=memory_store,
        idle_release_timeout=timedelta(milliseconds=10),
    )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )

    async with server.contextmanager():
        # Start workflow
        handler_id = "consumer-test-1"
        handler = server._workflows["test"].run()
        await server._run_workflow_handler(handler_id, "test", handler)

        wrapper = server._handlers[handler_id]

        # Simulate an active consumer by locking the mutex
        async with wrapper.consumer_mutex:
            # Force idle state and trigger timer
            wrapper.idle_since = datetime.now(timezone.utc)
            wrapper._start_idle_release_timer()

            # Wait for timer to fire (should be blocked by consumer_mutex)
            await asyncio.sleep(0.05)

            # Should NOT be released because consumer_mutex is locked
            assert handler_id in server._handlers


@pytest.mark.asyncio
async def test_reloaded_idle_workflow_is_released_again(
    server_with_idle_release: WorkflowServer, memory_store: MemoryWorkflowStore
) -> None:
    """Test that a reloaded workflow that stays idle gets released again."""
    server = server_with_idle_release

    # Start a workflow
    handler_id = "reload-idle-test-1"
    handler = server._workflows["test"].run()
    await server._run_workflow_handler(handler_id, "test", handler)

    # Wait for workflow to become idle
    async def check_idle() -> None:
        wrapper = server._handlers.get(handler_id)
        if wrapper is None or wrapper.idle_since is None:
            raise ValueError("Not idle yet")

    await wait_for_passing(check_idle)

    # Wait for release
    await asyncio.sleep(0.1)
    assert handler_id not in server._handlers

    # Reload the workflow (simulating an event arriving)
    reloaded = await server._try_reload_handler(handler_id)
    assert reloaded is not None
    assert handler_id in server._handlers

    # The reloaded handler should have idle_since restored
    assert reloaded.idle_since is not None

    # Since the workflow is still idle (no event woke it up),
    # it should have a release timer running
    assert reloaded._idle_release_timer is not None

    # Wait for it to be released again
    await asyncio.sleep(0.1)
    assert handler_id not in server._handlers


@pytest.mark.asyncio
async def test_idle_handlers_not_resumed_on_server_start(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that idle handlers are not loaded into memory on server start."""
    # Seed the store with an idle handler
    idle_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = Context(waiting_workflow).to_dict()
    await memory_store.update(
        PersistentHandler(
            handler_id="idle-on-start-1",
            workflow_name="test",
            status="running",
            idle_since=idle_time,
            ctx=ctx,
        )
    )

    # Also seed an active (non-idle) handler
    await memory_store.update(
        PersistentHandler(
            handler_id="active-on-start-1",
            workflow_name="test",
            status="running",
            idle_since=None,  # Not idle
            ctx=ctx,
        )
    )

    server = WorkflowServer(
        workflow_store=memory_store,
        idle_release_timeout=timedelta(minutes=5),
    )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )

    async with server.contextmanager():
        # The idle handler should NOT be in memory
        assert "idle-on-start-1" not in server._handlers

        # The active handler SHOULD be in memory
        assert "active-on-start-1" in server._handlers

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
    server = WorkflowServer(
        workflow_store=memory_store,
        idle_release_timeout=None,  # Disabled
    )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )

    async with server.contextmanager():
        # Start a workflow
        handler_id = "no-release-test"
        handler = server._workflows["test"].run()
        await server._run_workflow_handler(handler_id, "test", handler)

        # Wait for workflow to become idle
        async def check_idle() -> None:
            wrapper = server._handlers.get(handler_id)
            if wrapper is None or wrapper.idle_since is None:
                raise ValueError("Not idle yet")

        await wait_for_passing(check_idle)

        wrapper = server._handlers[handler_id]
        # Timer should not be set since idle_release_timeout is None
        assert wrapper._idle_release_timer is None
