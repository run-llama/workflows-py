# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
"""Tests for idle workflow release and reload functionality."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest
import time_machine
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


@pytest.mark.asyncio
async def test_bug_idle_release_leaves_workflow_runtime_alive(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """BUG: Idle release should stop the workflow runtime, but it doesn't.

    When a workflow is released from memory, the underlying WorkflowHandler
    and its context/broker should be cancelled. Currently, _release_handler
    only cancels the server's stream task and removes from _handlers, but
    the actual workflow runtime keeps running in the background.

    This test captures a reference to the run_handler before release, then
    verifies it should be done/cancelled after release.
    """
    server = WorkflowServer(
        workflow_store=memory_store,
        idle_release_timeout=timedelta(milliseconds=50),
    )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )

    async with server.contextmanager():
        # Start a workflow
        handler_id = "runtime-leak-test"
        handler = server._workflows["test"].run()
        await server._run_workflow_handler(handler_id, "test", handler)

        # Wait for workflow to become idle
        async def check_idle() -> None:
            wrapper = server._handlers.get(handler_id)
            if wrapper is None or wrapper.idle_since is None:
                raise ValueError("Not idle yet")

        await wait_for_passing(check_idle)

        # Capture reference to the run_handler before release
        wrapper = server._handlers[handler_id]
        run_handler = wrapper.run_handler

        # Wait for release
        await asyncio.sleep(0.15)
        assert handler_id not in server._handlers, "Handler should be released"

        # BUG: The run_handler should be done/cancelled after release,
        # but it's still running because _release_handler doesn't stop it
        assert run_handler.done(), (
            "BUG: run_handler is still running after release - "
            "workflow runtime was not stopped, causing memory leak"
        )


@pytest.mark.asyncio
async def test_bug_consumer_mutex_blocks_release_without_reschedule(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """BUG: If consumer_mutex is locked when timer fires, release is skipped forever.

    The idle release timer checks if consumer_mutex is locked and skips release
    if so. However, it does NOT reschedule another timer attempt. This means
    if a client briefly holds a stream during the timeout window and then
    disconnects, the handler will never be released.

    This test holds the mutex during the timer, releases it, then verifies
    the handler should eventually be released (but won't be due to the bug).
    """
    server = WorkflowServer(
        workflow_store=memory_store,
        # Short timeout so test runs quickly
        idle_release_timeout=timedelta(milliseconds=50),
    )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )

    async with server.contextmanager():
        # Start workflow
        handler_id = "mutex-reschedule-test"
        handler = server._workflows["test"].run()
        await server._run_workflow_handler(handler_id, "test", handler)

        # Wait for workflow to become idle
        async def check_idle() -> None:
            wrapper = server._handlers.get(handler_id)
            if wrapper is None or wrapper.idle_since is None:
                raise ValueError("Not idle yet")

        await wait_for_passing(check_idle)

        wrapper = server._handlers[handler_id]

        # Grab the mutex before timer fires, then hold it past the timeout
        async with wrapper.consumer_mutex:
            # Wait for the timer to fire and be blocked by mutex
            await asyncio.sleep(0.1)
            # Handler should still be present (timer was blocked by mutex)
            assert handler_id in server._handlers, "Handler released while mutex held"

        # Now mutex is released - handler should eventually be released
        # because the timer reschedules when mutex was locked.
        # Wait for the rescheduled timer to fire.
        await asyncio.sleep(0.15)

        # After fix: Handler should be released because timer was rescheduled
        assert handler_id not in server._handlers, (
            "BUG: Handler was never released after mutex was unlocked - "
            "timer should have been rescheduled but wasn't"
        )


@pytest.mark.asyncio
async def test_bug_race_between_event_post_and_idle_timer(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """BUG: Race condition where event is posted but timer releases handler.

    When an event is posted to wake a workflow, the idle_since is only cleared
    when StepStateChanged(RUNNING) is observed in _stream_events. However,
    the idle timer can fire in the window between event posting and the
    running event being processed, causing the handler to be released while
    the workflow is actively processing.

    DESIRED IMPLEMENTATION:
    We want to support short/immediate idle timeouts (e.g., 0 timeout to release
    workflows immediately when idle). However, "revived" workflows (ones that
    just received an event but haven't processed it yet) need protection. The
    fix should use a minimum grace period for revived workflows - when an event
    is sent to an idle workflow, the timer should be rescheduled with at least
    a minimum timeout (e.g., 1 second) regardless of the configured timeout.
    This prevents immediate release while still allowing fast release of truly
    idle workflows.

    The fix implemented: The server's _post_event endpoint calls mark_active()
    before sending an event, which clears idle_since and cancels the timer.
    This test verifies that mark_active() properly protects the handler.
    """
    server = WorkflowServer(
        workflow_store=memory_store,
        # Short timeout so test runs quickly
        idle_release_timeout=timedelta(milliseconds=50),
    )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )

    async with server.contextmanager():
        handler_id = "race-test"
        handler = server._workflows["test"].run()
        await server._run_workflow_handler(handler_id, "test", handler)

        # Wait for workflow to become idle
        async def check_idle() -> None:
            wrapper = server._handlers.get(handler_id)
            if wrapper is None or wrapper.idle_since is None:
                raise ValueError("Not idle yet")

        await wait_for_passing(check_idle)

        wrapper = server._handlers[handler_id]
        assert wrapper.idle_since is not None, "Should be idle"

        # Mark active before sending event (this is what _post_event does)
        # This clears idle_since and cancels the timer
        wrapper.mark_active()

        # Send event to wake the workflow
        ctx = wrapper.run_handler.ctx
        assert ctx is not None
        ctx.send_event(WaitableExternalEvent(response="wake-up"))

        # Wait past the original idle timeout
        # The workflow should be protected from release because mark_active was called
        await asyncio.sleep(0.15)

        # After the fix: Handler should NOT be released because mark_active()
        # cleared idle_since and cancelled the timer before the event was sent
        assert handler_id in server._handlers, (
            "BUG: Handler was released even though mark_active() was called. "
            "mark_active() should clear idle_since and cancel the timer to "
            "protect the handler from release."
        )


@pytest.mark.asyncio
async def test_bug_concurrent_reloads_create_duplicate_handlers(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """BUG: Concurrent reload requests can create multiple handlers.

    _try_reload_handler has no locking mechanism, so two parallel requests
    can both reload and start the same workflow. This creates split-brain
    scenarios where multiple workflow instances process events independently.

    This test fires multiple reload requests in parallel and checks that
    only one succeeds in creating a handler.
    """
    server = WorkflowServer(
        workflow_store=memory_store,
        idle_release_timeout=timedelta(minutes=5),
    )
    server.add_workflow(
        "test", waiting_workflow, additional_events=[WaitableExternalEvent]
    )

    # Seed a persisted idle handler
    idle_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = Context(waiting_workflow).to_dict()
    handler_id = "concurrent-reload-test"
    await memory_store.update(
        PersistentHandler(
            handler_id=handler_id,
            workflow_name="test",
            status="running",
            idle_since=idle_time,
            ctx=ctx,
        )
    )

    async with server.contextmanager():
        assert handler_id not in server._handlers

        # Track how many workflow instances were created
        instances_created: list[object] = []

        async def reload_and_track() -> None:
            wrapper = await server._try_reload_handler(handler_id)
            if wrapper is not None:
                # Track the actual run_handler object identity
                instances_created.append(id(wrapper.run_handler))

        # Fire 5 concurrent reload requests
        await asyncio.gather(*[reload_and_track() for _ in range(5)])

        # We should have exactly one handler in memory
        assert handler_id in server._handlers

        # BUG: Multiple unique workflow instances may have been created
        # (even though only one ends up in _handlers, the others are leaked)
        unique_instances = set(instances_created)
        assert len(unique_instances) == 1, (
            f"BUG: {len(unique_instances)} different workflow instances were created "
            f"by concurrent reloads. Only one should be created. "
            f"The extra instances are leaked and may process events independently."
        )


# =============================================================================
# REVIEWER-FLAGGED BUG TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_bug_release_handler_leaks_runtime_if_checkpoint_fails(
    memory_store: MemoryWorkflowStore,
    waiting_workflow: Workflow,
) -> None:
    """BUG: _release_handler removes handler from _handlers before checkpoint().

    If checkpoint() raises after the handler is already removed from _handlers,
    the runtime cancel path never runs, leaking a live workflow with no
    in-memory reference.

    The current _release_handler structure is:
    1. Pop from _handlers (handler removed from memory)
    2. Call checkpoint() (if this fails, exception propagates)
    3. Cancel runtime (never reached if step 2 fails)

    This test triggers a realistic failure by storing a non-serializable object
    (a lambda) in the context state. When the idle release timer fires and
    tries to checkpoint, serialization fails and the runtime leaks.

    The fix should ensure that if ANY part of release fails, the workflow
    runtime is still properly cancelled (e.g., use try/finally).
    """
    # Use a moderate timeout - long enough to set up test conditions,
    # short enough to not slow down tests significantly
    idle_timeout = timedelta(milliseconds=100)

    # time_machine with tick=True gives us deterministic timestamps while
    # allowing real async operations to proceed normally
    with time_machine.travel(1000.0, tick=True):
        server = WorkflowServer(
            workflow_store=memory_store,
            idle_release_timeout=idle_timeout,
            persistence_backoff=[],  # No retries - fail immediately
        )
        server.add_workflow(
            "test", waiting_workflow, additional_events=[WaitableExternalEvent]
        )

        await server.start()
        run_handler = None
        try:
            handler_id = "checkpoint-fail-test"
            handler = waiting_workflow.run()
            await server._run_workflow_handler(handler_id, "test", handler)

            # Wait for workflow to become idle
            async def check_idle() -> None:
                wrapper = server._handlers.get(handler_id)
                if wrapper is None or wrapper.idle_since is None:
                    raise ValueError("Not idle yet")

            await wait_for_passing(check_idle)

            # Capture reference to verify runtime is stopped
            wrapper = server._handlers[handler_id]
            run_handler = wrapper.run_handler
            ctx = run_handler.ctx
            assert ctx is not None

            # Store a non-serializable object in context store
            # This will cause checkpoint() to fail when it tries to serialize
            await ctx.store.set("bad_data", lambda x: x)  # lambdas can't be serialized

            # Wait for the idle release timer to fire (100ms + buffer)
            await asyncio.sleep(0.25)

            # Verify handler was removed from _handlers (release started)
            assert handler_id not in server._handlers, "Handler should be removed"

            # BUG: The run_handler should be done/cancelled even if checkpoint failed
            # but it's still running because the cancel code was never reached
            assert run_handler.done(), (
                "BUG: run_handler is still running after release attempt failed. "
                "_release_handler pops from _handlers before checkpoint(), so if "
                "checkpoint() raises (e.g., due to non-serializable state), the "
                "cancel path is never reached and the workflow runtime leaks."
            )
        finally:
            # Clean up the leaked runtime manually (since the bug prevents normal cleanup)
            if run_handler is not None and not run_handler.done():
                run_handler.cancel()
                try:
                    await run_handler.cancel_run()
                except Exception:
                    pass
            await server.stop()


@pytest.mark.asyncio
async def test_failed_send_event_preserves_idle_state(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Verify that failed send_event() doesn't clear idle state.

    When send_event() fails (bad step name, runtime errors, etc.), the idle
    state should be preserved so the idle release timer can still fire.

    This test verifies the fix for the bug where mark_active() was called
    BEFORE send_event(), causing idle state to be cleared even on failure.

    The fix: mark_active() is now called AFTER send_event() succeeds.
    """
    from httpx import ASGITransport, AsyncClient

    # Use a moderate timeout for the test
    idle_timeout = timedelta(milliseconds=100)

    # time_machine with tick=True gives us deterministic timestamps while
    # allowing real async operations to proceed normally
    with time_machine.travel(1000.0, tick=True):
        server = WorkflowServer(
            workflow_store=memory_store,
            idle_release_timeout=idle_timeout,
        )
        server.add_workflow(
            "test", waiting_workflow, additional_events=[WaitableExternalEvent]
        )

        async with server.contextmanager():
            transport = ASGITransport(app=server.app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Start a workflow via HTTP
                response = await client.post("/workflows/test/run-nowait", json={})
                assert response.status_code == 200
                handler_id = response.json()["handler_id"]

                # Wait for workflow to become idle
                async def check_idle() -> None:
                    wrapper = server._handlers.get(handler_id)
                    if wrapper is None or wrapper.idle_since is None:
                        raise ValueError("Not idle yet")

                await wait_for_passing(check_idle)

                wrapper = server._handlers[handler_id]
                original_idle_since = wrapper.idle_since
                assert original_idle_since is not None

                # Post an event with a bad step name via HTTP - this will fail
                response = await client.post(
                    f"/events/{handler_id}",
                    json={
                        "event": {
                            "type": "WaitableExternalEvent",
                            "value": {"response": "test"},
                        },
                        "step": "nonexistent_step",  # This step doesn't exist
                    },
                )
                # The endpoint returns 400 for bad step
                assert response.status_code == 400

                # FIX: mark_active() is now called AFTER send_event() succeeds.
                # Since send_event() failed, idle state should be preserved.
                assert wrapper.idle_since == original_idle_since, (
                    "idle_since should be preserved when send_event() fails"
                )
                assert wrapper._idle_release_timer is not None, (
                    "timer should still be running"
                )

                # Wait for the timeout period (100ms + buffer)
                await asyncio.sleep(0.25)

                # Handler should be released by the timer since idle state was preserved
                assert handler_id not in server._handlers, (
                    "Handler should be released after idle timeout"
                )


@pytest.mark.asyncio
async def test_bug_release_and_reload_race_condition(
    memory_store: MemoryWorkflowStore,
    waiting_workflow: Workflow,
) -> None:
    """BUG: Release/reload isn't coordinated.

    _try_reload_handler uses KeyedLock, but _release_handler does not.
    This means release can overlap with reload, and the release checkpoint()
    can overwrite newer state from the reloaded handler.

    This test demonstrates the race by simulating what happens when release
    checkpoint is delayed and executes after reload has updated the state:
    1. Start handler, wait for idle, capture old state
    2. Reload handler and update to new state (idle_since=None)
    3. Simulate delayed release checkpoint writing old state
    4. Verify old state overwrites new state (the bug)

    The fix should have _release_handler use the same KeyedLock.
    """
    # time_machine with tick=True gives us deterministic timestamps
    with time_machine.travel(1000.0, tick=True):
        server = WorkflowServer(
            workflow_store=memory_store,
            idle_release_timeout=None,  # Disable auto-release for manual control
        )
        server.add_workflow(
            "test", waiting_workflow, additional_events=[WaitableExternalEvent]
        )

        async with server.contextmanager():
            # Start workflow and wait for it to become idle
            handler = waiting_workflow.run()
            handler_id = "race-test-handler"
            await server._run_workflow_handler(handler_id, "test", handler)

            # Wait for workflow to become idle
            async def check_idle() -> None:
                wrapper = server._handlers.get(handler_id)
                if wrapper is None or wrapper.idle_since is None:
                    raise ValueError("Not idle yet")

            await wait_for_passing(check_idle)

            wrapper = server._handlers[handler_id]
            original_idle_since = wrapper.idle_since
            assert original_idle_since is not None

            # Capture the "old" persistent state (what release would checkpoint)
            old_persistent_state = wrapper._as_persistent()
            assert old_persistent_state.idle_since == original_idle_since

            # Simulate release removing handler from memory
            server._handlers.pop(handler_id, None)

            # Now simulate the race: reload happens while release checkpoint is pending
            # 1. Reload the handler
            reloaded = await server._try_reload_handler(handler_id)
            assert reloaded is not None

            # 2. The reloaded handler receives an event and becomes active
            reloaded.mark_active()
            assert reloaded.idle_since is None

            # 3. Checkpoint the new state
            await reloaded.checkpoint()

            # Verify the store has the correct (new) state
            persisted = await memory_store.query(
                HandlerQuery(handler_id_in=[handler_id])
            )
            assert persisted[0].idle_since is None, "Store should have new state"

            # 4. NOW the delayed release checkpoint writes OLD state
            # This is what happens when release and reload race without coordination
            await memory_store.update(old_persistent_state)

            # Check what's in the store now
            persisted = await memory_store.query(
                HandlerQuery(handler_id_in=[handler_id])
            )
            assert len(persisted) == 1

            # BUG: The persisted state was overwritten by the stale release checkpoint.
            # Without proper locking, the release checkpoint can happen after reload
            # has updated the state, causing data loss.
            assert persisted[0].idle_since is None, (
                f"BUG: Stale release checkpoint overwrote newer state from reload. "
                f"Expected idle_since=None (from reload), "
                f"but got idle_since={persisted[0].idle_since}. "
                f"_release_handler should use KeyedLock to coordinate with "
                f"_try_reload_handler."
            )
