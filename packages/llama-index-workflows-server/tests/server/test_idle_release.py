# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
"""Tests for idle workflow release and reload functionality."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest
import time_machine
from httpx import ASGITransport, AsyncClient
from workflows import Context, Workflow, step
from workflows.events import HumanResponseEvent, StartEvent, StopEvent
from workflows.server.abstract_workflow_store import (
    HandlerQuery,
    PersistentHandler,
    Status,
)
from workflows.server.memory_workflow_store import MemoryWorkflowStore
from workflows.server.server import WorkflowServer, _WorkflowHandler

from .util import async_yield, wait_for_passing


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


def get_handler_in_memory(server: WorkflowServer, handler_id: str) -> _WorkflowHandler:
    wrapper = server._handlers.get(handler_id)
    assert wrapper is not None, f"Handler {handler_id} not found in memory"
    return wrapper


def assert_handler_in_memory(server: WorkflowServer, handler_id: str) -> None:
    get_handler_in_memory(server, handler_id)


def assert_handler_not_in_memory(server: WorkflowServer, handler_id: str) -> None:
    assert handler_id not in server._handlers, f"Handler {handler_id} still in memory"


def make_server(
    memory_store: MemoryWorkflowStore,
    waiting_workflow: Workflow,
    idle_release_timeout: Optional[timedelta],
    *,
    persistence_backoff: Optional[list[float]] = None,
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
) -> _WorkflowHandler:
    handler = server._workflows["test"].run()
    await server._run_workflow_handler(handler_id, "test", handler)
    await async_yield(20)
    wrapper = get_handler_in_memory(server, handler_id)
    assert wrapper.idle_since is not None
    return wrapper


async def advance_time(traveller: Any, delta: timedelta, iterations: int = 10) -> None:
    traveller.shift(delta)
    await async_yield(iterations)


async def seed_persistent_handler(
    store: MemoryWorkflowStore,
    handler_id: str,
    *,
    idle_since: Optional[datetime],
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


@pytest.mark.asyncio
async def test_is_idle_query_filter_memory_store() -> None:
    """Test that is_idle filter works in MemoryWorkflowStore."""
    store = MemoryWorkflowStore()

    now = datetime.now(timezone.utc)

    # Handler that is idle (has idle_since set)
    await seed_persistent_handler(
        store,
        "idle-1",
        idle_since=now - timedelta(minutes=5),
        ctx={},
    )

    # Handler that is not idle (no idle_since)
    await seed_persistent_handler(store, "active-1", idle_since=None, ctx={})

    # Another idle handler
    await seed_persistent_handler(
        store,
        "idle-2",
        idle_since=now - timedelta(seconds=10),
        ctx={},
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
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that a workflow becomes idle after a step and can be released."""
    idle_timeout = timedelta(milliseconds=50)

    with time_machine.travel("2026-01-07T12:00:00Z", tick=False) as traveller:
        server = make_server(memory_store, waiting_workflow, idle_timeout)

        async with server.contextmanager():
            # Start a workflow
            handler_id = "release-test-1"
            await start_waiting_handler(server, handler_id)

            # Advance time past the idle timeout to trigger the timer
            await advance_time(traveller, timedelta(milliseconds=100))

            # The handler should be released from memory
            assert_handler_not_in_memory(server, handler_id)

            # But should still exist in the store with status "running"
            persisted = await memory_store.query(
                HandlerQuery(handler_id_in=[handler_id])
            )
            assert len(persisted) == 1
            assert persisted[0].status == "running"
            assert persisted[0].idle_since is not None


@pytest.mark.asyncio
async def test_released_workflow_is_reloaded_on_event(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that a released workflow is reloaded when an event is sent."""
    idle_timeout = timedelta(milliseconds=50)

    with time_machine.travel("2026-01-07T12:00:00Z", tick=False) as traveller:
        server = make_server(memory_store, waiting_workflow, idle_timeout)

        async with server.contextmanager():
            # Start a workflow
            handler_id = "reload-test-1"
            await start_waiting_handler(server, handler_id)

            # Advance time past the idle timeout to trigger the timer
            await advance_time(traveller, timedelta(milliseconds=100))

            # Handler should be released
            assert_handler_not_in_memory(server, handler_id)

            # Now reload by using _try_reload_handler
            reloaded, persisted = await server._try_reload_handler(handler_id)
            assert reloaded is not None
            assert_handler_in_memory(server, handler_id)

            assert persisted is not None
            assert persisted.status == "running"

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
    server = make_server(
        memory_store,
        waiting_workflow,
        timedelta(minutes=5),  # Long timeout
    )

    # Seed a persisted handler with idle_since
    idle_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = Context(waiting_workflow).to_dict()
    await seed_persistent_handler(
        memory_store,
        "idle-restore-1",
        idle_since=idle_time,
        ctx=ctx,
    )

    async with server.contextmanager():
        # Idle handler should NOT be in memory on startup
        assert_handler_not_in_memory(server, "idle-restore-1")

        # Reload it (simulating an event arriving)
        wrapper, persisted = await server._try_reload_handler("idle-restore-1")
        assert wrapper is not None
        assert wrapper.idle_since == idle_time
        assert persisted is not None
        assert persisted.status == "running"


@pytest.mark.asyncio
async def test_reloaded_idle_workflow_is_released_again(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that a reloaded workflow that stays idle gets released again."""
    # Use very short timeout - this test needs real time for timers
    idle_timeout = timedelta(milliseconds=1)

    server = make_server(memory_store, waiting_workflow, idle_timeout)

    async with server.contextmanager():
        # Start a workflow
        handler_id = "reload-idle-test-1"
        handler = server._workflows["test"].run()
        wrapper = await server._run_workflow_handler(handler_id, "test", handler)

        async def wrapper_is_idle() -> None:
            assert wrapper.idle_since is not None

        await wait_for_passing(wrapper_is_idle, interval=0.01, max_duration=1.5)

        # Reload the workflow (simulating an event arriving) once the handler
        # is released from memory.
        async def reload_from_store() -> tuple[_WorkflowHandler, PersistentHandler]:
            reloaded, persisted = await server._try_reload_handler(handler_id)
            assert reloaded is not None
            assert persisted is not None
            assert reloaded is not wrapper
            return reloaded, persisted

        reloaded, persisted = await wait_for_passing(
            reload_from_store, interval=0.01, max_duration=1.5
        )

        assert persisted is not None
        assert persisted.status == "running"

        # The reloaded handler should have idle_since restored
        assert reloaded.idle_since is not None

        # Wait for the reloaded handler to be released again by observing
        # that a subsequent reload returns a new handler instance.
        async def reload_after_release() -> _WorkflowHandler:
            reloaded_again, persisted_again = await server._try_reload_handler(
                handler_id
            )
            assert reloaded_again is not None
            assert persisted_again is not None
            assert reloaded_again is not reloaded
            return reloaded_again

        reloaded_again = await wait_for_passing(
            reload_after_release, interval=0.01, max_duration=1.5
        )
        await server._close_handler(reloaded_again)


@pytest.mark.asyncio
async def test_idle_handlers_not_resumed_on_server_start(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Test that idle handlers are not loaded into memory on server start."""
    # Seed the store with an idle handler
    idle_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = Context(waiting_workflow).to_dict()
    await seed_persistent_handler(
        memory_store,
        "idle-on-start-1",
        idle_since=idle_time,
        ctx=ctx,
    )

    # Also seed an active (non-idle) handler
    await seed_persistent_handler(
        memory_store,
        "active-on-start-1",
        idle_since=None,  # Not idle
        ctx=ctx,
    )

    server = make_server(
        memory_store,
        waiting_workflow,
        timedelta(minutes=5),
    )

    async with server.contextmanager():
        # The idle handler should NOT be in memory
        assert_handler_not_in_memory(server, "idle-on-start-1")

        # The active handler SHOULD be in memory
        assert_handler_in_memory(server, "active-on-start-1")

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
    server = make_server(
        memory_store,
        waiting_workflow,
        None,  # Disabled
    )

    async with server.contextmanager():
        # Start a workflow
        handler_id = "no-release-test"
        wrapper = await start_waiting_handler(server, handler_id)

        # Timer should not be set since idle_release_timeout is None
        assert wrapper._idle_release_timer is None


@pytest.mark.asyncio
async def test_idle_release_cancels_runtime(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Idle release should stop the workflow runtime.

    When a workflow is released from memory, the underlying WorkflowHandler
    and its context/broker should be cancelled. Currently, _release_handler
    only cancels the server's stream task and removes from _handlers, but
    the actual workflow runtime keeps running in the background.

    This test captures a reference to the run_handler before release, then
    verifies it should be done/cancelled after release.
    """
    idle_timeout = timedelta(milliseconds=50)

    with time_machine.travel("2026-01-07T12:00:00Z", tick=False) as traveller:
        server = make_server(memory_store, waiting_workflow, idle_timeout)

        async with server.contextmanager():
            # Start a workflow
            handler_id = "runtime-leak-test"
            wrapper = await start_waiting_handler(server, handler_id)

            # Capture reference to the run_handler before release
            run_handler = wrapper.run_handler

            # Advance time past the idle timeout to trigger the timer
            await advance_time(traveller, timedelta(milliseconds=100))

            # Handler should be released

            assert_handler_not_in_memory(server, handler_id)

            assert run_handler.done(), (
                "run_handler is still running after release workflow runtime was not stopped, causing memory leak"
            )


@pytest.mark.asyncio
async def test_idle_release_waits_for_stream_consumer_then_releases(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Idle release should wait for active consumers and then reschedule.

    The idle release timer checks if consumer_mutex is locked and skips release
    if so. However, it does NOT reschedule another timer attempt. This means
    if a client briefly holds a stream during the timeout window and then
    disconnects, the handler will never be released.

    This test holds the mutex during the timer, releases it, then verifies
    the handler should eventually be released (but won't be due to the issue).
    """
    idle_timeout = timedelta(milliseconds=50)

    with time_machine.travel("2026-01-07T12:00:00Z", tick=False) as traveller:
        server = make_server(memory_store, waiting_workflow, idle_timeout)

        async with server.contextmanager():
            # Start workflow
            handler_id = "mutex-reschedule-test"
            wrapper = await start_waiting_handler(server, handler_id)

            # Grab the mutex before timer fires, then hold it past the timeout
            async with wrapper.consumer_mutex:
                # Advance time past the timeout to trigger the timer
                await advance_time(traveller, timedelta(milliseconds=100))
                # Handler should still be present (timer was blocked by mutex)
                assert_handler_in_memory(server, handler_id)

            # Now mutex is released - handler should eventually be released
            # because the timer reschedules when mutex was locked.
            # Advance time to let the rescheduled timer fire.
            await advance_time(traveller, timedelta(milliseconds=100))

            # Handler should be released because timer was rescheduled
            assert_handler_not_in_memory(server, handler_id)


@pytest.mark.asyncio
async def test_mark_active_prevents_release_during_event_post(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Marking active should protect from release during event post.

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
    idle_timeout = timedelta(milliseconds=50)

    with time_machine.travel("2026-01-07T12:00:00Z", tick=False) as traveller:
        server = make_server(memory_store, waiting_workflow, idle_timeout)

        async with server.contextmanager():
            handler_id = "race-test"
            wrapper = await start_waiting_handler(server, handler_id)

            # Mark active before sending event (this is what _post_event does)
            # This clears idle_since and cancels the timer
            wrapper.mark_active()

            # Send event to wake the workflow
            ctx = wrapper.run_handler.ctx
            assert ctx is not None
            ctx.send_event(WaitableExternalEvent(response="wake-up"))

            # Advance time past the original idle timeout
            # The workflow should be protected from release because mark_active was called
            await advance_time(traveller, timedelta(milliseconds=100))

            # Handler should NOT be released because mark_active()
            # cleared idle_since and cancelled the timer before the event was sent
            assert_handler_in_memory(server, handler_id)


@pytest.mark.asyncio
async def test_try_reload_is_singleton_under_concurrency(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Concurrent reload requests should only create one handler.

    _try_reload_handler has no locking mechanism, so two parallel requests
    can both reload and start the same workflow. This creates split-brain
    scenarios where multiple workflow instances process events independently.

    This test fires multiple reload requests in parallel and checks that
    only one succeeds in creating a handler.
    """
    server = make_server(
        memory_store,
        waiting_workflow,
        timedelta(minutes=5),
    )

    # Seed a persisted idle handler
    idle_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = Context(waiting_workflow).to_dict()
    handler_id = "concurrent-reload-test"
    await seed_persistent_handler(
        memory_store,
        handler_id,
        idle_since=idle_time,
        ctx=ctx,
    )

    async with server.contextmanager():
        assert_handler_not_in_memory(server, handler_id)

        # Track how many workflow instances were created
        instances_created: list[object] = []

        async def reload_and_track() -> None:
            wrapper, _ = await server._try_reload_handler(handler_id)
            if wrapper is not None:
                # Track the actual run_handler object identity
                instances_created.append(id(wrapper.run_handler))

        # Fire 5 concurrent reload requests
        await asyncio.gather(*[reload_and_track() for _ in range(5)])

        # We should have exactly one handler in memory
        assert_handler_in_memory(server, handler_id)

        # Multiple unique workflow instances may have been created
        # (even though only one ends up in _handlers, the others are leaked)
        unique_instances = set(instances_created)
        assert len(unique_instances) == 1, (
            f"{len(unique_instances)} different workflow instances were created "
            f"by concurrent reloads. Only one should be created. "
            f"The extra instances are leaked and may process events independently."
        )


@pytest.mark.asyncio
async def test_release_handler_cancels_runtime_on_checkpoint_failure(
    memory_store: MemoryWorkflowStore,
    waiting_workflow: Workflow,
) -> None:
    """Release should cancel runtime even if checkpoint fails.

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
    idle_timeout = timedelta(milliseconds=100)

    with time_machine.travel("2026-01-07T12:00:00Z", tick=False) as traveller:
        server = make_server(
            memory_store,
            waiting_workflow,
            idle_timeout,
            persistence_backoff=[],  # No retries - fail immediately
        )

        await server.start()
        run_handler = None
        try:
            handler_id = "checkpoint-fail-test"
            wrapper = await start_waiting_handler(server, handler_id)

            # Capture reference to verify runtime is stopped
            run_handler = wrapper.run_handler
            ctx = run_handler.ctx
            assert ctx is not None

            # Store a non-serializable object in context store
            # This will cause checkpoint() to fail when it tries to serialize
            await ctx.store.set("bad_data", lambda x: x)  # lambdas can't be serialized

            # Advance time past the idle timeout to trigger the timer
            await advance_time(traveller, timedelta(milliseconds=200), iterations=20)

            # Verify handler was removed from _handlers (release started)
            assert_handler_not_in_memory(server, handler_id)

            # The run_handler should be done/cancelled even if checkpoint failed
            # but it's still running because the cancel code was never reached
            assert run_handler.done(), (
                "run_handler is still running after release attempt failed. "
                "_release_handler pops from _handlers before checkpoint(), so if "
                "checkpoint() raises (e.g., due to non-serializable state), the "
                "cancel path is never reached and the workflow runtime leaks."
            )
        finally:
            # Clean up the leaked runtime manually (since the issue prevents normal cleanup)
            if run_handler is not None and not run_handler.done():
                run_handler.cancel()
                try:
                    await run_handler.cancel_run()
                except Exception:
                    pass
            await server.stop()


@pytest.mark.asyncio
async def test_send_event_clears_idle_state_before_processing(
    memory_store: MemoryWorkflowStore, waiting_workflow: Workflow
) -> None:
    """Verify that send_event() clears idle state immediately to prevent race conditions.

    mark_active() is called BEFORE processing the event to prevent a race where
    the idle release timer fires while we're still handling the request. This means
    idle_since is cleared even if send_event() subsequently fails.
    """
    idle_timeout = timedelta(milliseconds=100)

    with time_machine.travel("2026-01-07T12:00:00Z", tick=False):
        server = make_server(memory_store, waiting_workflow, idle_timeout)

        async with server.contextmanager():
            transport = ASGITransport(app=server.app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Start a workflow via HTTP
                response = await client.post("/workflows/test/run-nowait", json={})
                assert response.status_code == 200
                handler_id = response.json()["handler_id"]

                # Wait for workflow to become idle (just needs event loop iterations)
                await async_yield(20)
                wrapper = get_handler_in_memory(server, handler_id)
                assert wrapper is not None and wrapper.idle_since is not None

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

                # mark_active() is called BEFORE send_event() to prevent race conditions.
                # Even though send_event() failed, idle state is cleared.
                assert wrapper.idle_since is None, (
                    "idle_since should be cleared before send_event() is attempted"
                )
                assert wrapper._idle_release_timer is None, "timer should be cancelled"


@pytest.mark.asyncio
async def test_release_skips_checkpoint_if_handler_was_reloaded(
    memory_store: MemoryWorkflowStore,
    waiting_workflow: Workflow,
) -> None:
    """Verify that _release_handler skips checkpoint if handler was reloaded.

    When release and reload race, _release_handler should detect if a new
    handler instance is now in _handlers and skip the checkpoint to avoid
    overwriting newer state.

    This test verifies:
    1. Start handler, wait for idle
    2. Reload the handler (simulating it was released earlier)
    3. Update the reloaded handler's state
    4. Call _release_handler with the OLD wrapper
    5. Verify the old release doesn't overwrite the new state
    """
    # time_machine with tick=False for fast test execution
    with time_machine.travel("2026-01-07T12:27:00.000-08:00", tick=False):
        server = WorkflowServer(
            workflow_store=memory_store,
            idle_release_timeout=None,  # Disable auto-release for manual control
        )
        server.add_workflow(
            "test", waiting_workflow, additional_events=[WaitableExternalEvent]
        )

        async with server.contextmanager():
            # Start workflow and wait for it to become idle
            handler_id = "race-test-handler"
            await start_waiting_handler(server, handler_id)

            old_wrapper = get_handler_in_memory(server, handler_id)
            original_idle_since = old_wrapper.idle_since
            assert original_idle_since is not None

            # Checkpoint the old state (simulating what release would have done)
            await old_wrapper.checkpoint()

            # Simulate the scenario where handler was released and then reloaded:
            # Remove old handler from memory
            server._handlers.pop(handler_id, None)

            # Reload the handler (this gets the persisted state)
            reloaded, persisted = await server._try_reload_handler(handler_id)
            assert reloaded is not None
            assert reloaded is not old_wrapper  # Different instance
            assert persisted is not None
            assert persisted.status == "running"

            # The reloaded handler receives an event and becomes active
            reloaded.mark_active()
            assert reloaded.idle_since is None

            # Checkpoint the new state
            await reloaded.checkpoint()

            # Verify store has the new state (idle_since=None)
            stored = await memory_store.query(HandlerQuery(handler_id_in=[handler_id]))
            assert stored[0].idle_since is None, "Store should have new state"

            # NOW call _release_handler with the OLD wrapper
            # This simulates a delayed release that happens after reload
            # _release_handler should detect that a different handler is in _handlers
            # and skip the checkpoint
            await server._release_handler(old_wrapper)

            # Verify the store still has the correct (new) state
            stored = await memory_store.query(HandlerQuery(handler_id_in=[handler_id]))
            assert len(stored) == 1

            # The old release should have skipped the checkpoint
            # because it detected a different handler instance in _handlers
            assert stored[0].idle_since is None, (
                f"Release should have skipped checkpoint since handler was reloaded. "
                f"Expected idle_since=None (from reload), "
                f"but got idle_since={stored[0].idle_since}."
            )
