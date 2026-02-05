# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ServerRuntimeDecorator, ServerInternalAdapter, and ServerExternalAdapter."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

import pytest
from llama_agents.server._server_runtime import (
    ServerExternalAdapter,
    ServerInternalAdapter,
    ServerRuntimeDecorator,
)
from llama_agents.server.abstract_workflow_store import (
    AbstractWorkflowStore,
    PersistentHandler,
)
from llama_agents.server.memory_workflow_store import MemoryWorkflowStore
from workflows.context.state_store import StateStore
from workflows.events import Event, StopEvent
from workflows.runtime.types.named_task import NamedTask
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    RegisteredWorkflow,
    Runtime,
    WaitResult,
    WaitResultTimeout,
)
from workflows.runtime.types.ticks import WorkflowTick

# -- Stubs -----------------------------------------------------------------


class StubInternalAdapter(InternalRunAdapter):
    def __init__(self) -> None:
        self.closed = False
        self.after_step_completed_called = False
        self.wait_for_next_task_calls: list[tuple[list[NamedTask], float | None]] = []

    @property
    def run_id(self) -> str:
        return "r1"

    async def write_to_event_stream(self, event: Event) -> None:
        pass

    async def get_now(self) -> float:
        return 1.0

    async def send_event(self, tick: WorkflowTick) -> None:
        pass

    async def wait_receive(self, timeout_seconds: float | None = None) -> WaitResult:
        return WaitResultTimeout()

    async def close(self) -> None:
        self.closed = True

    def get_state_store(self) -> StateStore[Any] | None:
        return None

    async def after_step_completed(self) -> None:
        self.after_step_completed_called = True

    async def wait_for_next_task(
        self,
        task_set: list[NamedTask],
        timeout: float | None = None,
    ) -> asyncio.Task[Any] | None:
        self.wait_for_next_task_calls.append((task_set, timeout))
        return None


class StubExternalAdapter(ExternalRunAdapter):
    def __init__(self) -> None:
        self.closed = False
        self.cancelled = False

    @property
    def run_id(self) -> str:
        return "r1"

    async def send_event(self, tick: WorkflowTick) -> None:
        pass

    async def stream_published_events(self) -> AsyncGenerator[Event, None]:
        yield StopEvent(result="done")

    async def close(self) -> None:
        self.closed = True

    async def get_result(self) -> StopEvent:
        return StopEvent(result="done")

    def get_state_store(self) -> StateStore[Any] | None:
        return None

    async def cancel(self) -> None:
        self.cancelled = True


class StubRuntime(Runtime):
    def __init__(self) -> None:
        self.launched = False
        self._internal = StubInternalAdapter()
        self._external = StubExternalAdapter()

    def register(self, workflow: Any) -> RegisteredWorkflow:
        return RegisteredWorkflow(
            workflow=workflow, workflow_run_fn=MagicMock(), steps={}
        )

    def run_workflow(
        self,
        run_id: str,
        workflow: Any,
        init_state: Any,
        start_event: Any = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: Any = None,
    ) -> ExternalRunAdapter:
        return self._external

    def get_internal_adapter(self, workflow: Any) -> InternalRunAdapter:
        return self._internal

    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        return self._external

    def launch(self) -> None:
        self.launched = True

    def destroy(self) -> None:
        pass


# -- Fixtures --------------------------------------------------------------


@pytest.fixture
def memory_store() -> MemoryWorkflowStore:
    return MemoryWorkflowStore()


@pytest.fixture
def stub_runtime() -> StubRuntime:
    return StubRuntime()


@pytest.fixture
def runtime_decorator(
    stub_runtime: StubRuntime,
    memory_store: MemoryWorkflowStore,
) -> ServerRuntimeDecorator:
    return ServerRuntimeDecorator(
        stub_runtime,
        store=memory_store,
        persistence_backoff=[0.0, 0.0],
        idle_release_timeout=None,
    )


def _setup_run(
    runtime_decorator: ServerRuntimeDecorator,
    handler_id: str = "h1",
    workflow_name: str = "test_wf",
    ctx_dict: dict[str, Any] | None = None,
) -> ServerExternalAdapter:
    """Helper: run_workflow, register_run, and register_context_provider_strong."""
    ext = runtime_decorator.run_workflow(
        run_id="r1",
        workflow=MagicMock(),
        init_state=MagicMock(),
    )
    assert isinstance(ext, ServerExternalAdapter)
    runtime_decorator.register_run(
        handler_id,
        ext,
        workflow_name=workflow_name,
    )
    if ctx_dict is None:
        ctx_dict = {"globals": {}, "state": {}}
    runtime_decorator.register_context_provider_strong(
        handler_id,
        lambda: ctx_dict,
    )
    return ext


# -- Tests: after_step_completed triggers checkpoint -----------------------


async def test_after_step_completed_triggers_checkpoint(
    runtime_decorator: ServerRuntimeDecorator,
    memory_store: MemoryWorkflowStore,
) -> None:
    """after_step_completed() should persist a PersistentHandler to the store."""
    _setup_run(runtime_decorator)
    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    await internal.after_step_completed()

    assert "h1" in memory_store.handlers
    handler = memory_store.handlers["h1"]
    assert handler.workflow_name == "test_wf"
    assert handler.status == "running"
    assert handler.run_id == "r1"


async def test_after_step_completed_persists_context(
    runtime_decorator: ServerRuntimeDecorator,
    memory_store: MemoryWorkflowStore,
) -> None:
    """The checkpoint should include the context dict from the provider."""
    ctx_data = {"globals": {"key": "value"}, "state": {"counter": 42}}
    _setup_run(runtime_decorator, ctx_dict=ctx_data)
    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    await internal.after_step_completed()

    handler = memory_store.handlers["h1"]
    assert handler.ctx == ctx_data


async def test_after_step_completed_forwards_to_inner(
    runtime_decorator: ServerRuntimeDecorator,
    stub_runtime: StubRuntime,
) -> None:
    """after_step_completed() should still call the inner adapter's method."""
    _setup_run(runtime_decorator)
    internal = runtime_decorator.get_internal_adapter(MagicMock())

    await internal.after_step_completed()

    assert stub_runtime._internal.after_step_completed_called


# -- Tests: context provider missing ---------------------------------------


async def test_after_step_completed_skips_checkpoint_when_no_context_provider(
    runtime_decorator: ServerRuntimeDecorator,
    memory_store: MemoryWorkflowStore,
) -> None:
    """When no context provider is registered, checkpoint should be skipped."""
    ext = runtime_decorator.run_workflow(
        run_id="r1",
        workflow=MagicMock(),
        init_state=MagicMock(),
    )
    assert isinstance(ext, ServerExternalAdapter)
    # Register run but do NOT register a context provider
    runtime_decorator.register_run("h1", ext, workflow_name="test_wf")

    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    # Should not raise and should not write to the store
    await internal.after_step_completed()

    assert "h1" not in memory_store.handlers


async def test_after_step_completed_skips_when_no_matching_run(
    runtime_decorator: ServerRuntimeDecorator,
    memory_store: MemoryWorkflowStore,
) -> None:
    """When no active run matches the internal adapter's run_id, skip checkpoint."""
    # Do not register any run at all
    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    await internal.after_step_completed()

    assert len(memory_store.handlers) == 0


# -- Tests: idle detection via wait_for_next_task --------------------------


async def test_wait_for_next_task_detects_idle_single_task_no_timeout(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """A single task with timeout=None signals idle."""
    ext = _setup_run(runtime_decorator)
    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    dummy_task = asyncio.create_task(asyncio.sleep(10))
    try:
        single_task = [NamedTask.pull(0, dummy_task)]
        await internal.wait_for_next_task(single_task, timeout=None)

        meta = runtime_decorator.get_run_metadata("h1")
        assert meta is not None
        assert meta.idle_since is not None
    finally:
        dummy_task.cancel()
        try:
            await dummy_task
        except asyncio.CancelledError:
            pass


async def test_wait_for_next_task_detects_active_multiple_tasks(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """Multiple tasks means the workflow is active, not idle."""
    ext = _setup_run(runtime_decorator)
    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    task1 = asyncio.create_task(asyncio.sleep(10))
    task2 = asyncio.create_task(asyncio.sleep(10))
    try:
        multi_tasks = [
            NamedTask.pull(0, task1),
            NamedTask.worker("step_a", 0, task2),
        ]
        await internal.wait_for_next_task(multi_tasks, timeout=None)

        meta = runtime_decorator.get_run_metadata("h1")
        assert meta is not None
        assert meta.idle_since is None
    finally:
        task1.cancel()
        task2.cancel()
        for t in [task1, task2]:
            try:
                await t
            except asyncio.CancelledError:
                pass


async def test_wait_for_next_task_clears_idle_on_transition_to_active(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """Going from idle (1 task, no timeout) to active (2 tasks) should clear idle_since."""
    ext = _setup_run(runtime_decorator)
    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    task1 = asyncio.create_task(asyncio.sleep(10))
    task2 = asyncio.create_task(asyncio.sleep(10))
    try:
        # First: become idle
        single_task = [NamedTask.pull(0, task1)]
        await internal.wait_for_next_task(single_task, timeout=None)
        meta = runtime_decorator.get_run_metadata("h1")
        assert meta is not None
        assert meta.idle_since is not None

        # Then: become active
        multi_tasks = [
            NamedTask.pull(0, task1),
            NamedTask.worker("step_a", 0, task2),
        ]
        await internal.wait_for_next_task(multi_tasks, timeout=None)
        assert meta.idle_since is None
    finally:
        task1.cancel()
        task2.cancel()
        for t in [task1, task2]:
            try:
                await t
            except asyncio.CancelledError:
                pass


async def test_wait_for_next_task_single_task_with_timeout_is_not_idle(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """A single task WITH a timeout is not idle (timeout means work is expected)."""
    ext = _setup_run(runtime_decorator)
    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    task = asyncio.create_task(asyncio.sleep(10))
    try:
        single_task = [NamedTask.pull(0, task)]
        await internal.wait_for_next_task(single_task, timeout=5.0)

        meta = runtime_decorator.get_run_metadata("h1")
        assert meta is not None
        assert meta.idle_since is None
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# -- Tests: checkpoint retry/backoff --------------------------------------


class FailThenSucceedStore(AbstractWorkflowStore):
    """Store that fails on the first N update() calls, then succeeds."""

    def __init__(self, fail_count: int = 1) -> None:
        self._fail_count = fail_count
        self._attempts = 0
        self.last_handler: PersistentHandler | None = None

    async def query(self, query: Any) -> list[PersistentHandler]:
        return []

    async def update(self, handler: PersistentHandler) -> None:
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise RuntimeError(f"Simulated store failure (attempt {self._attempts})")
        self.last_handler = handler

    async def delete(self, query: Any) -> int:
        return 0

    @property
    def attempts(self) -> int:
        return self._attempts


async def test_checkpoint_retries_on_first_failure() -> None:
    """Store fails on first attempt, succeeds on second."""
    fail_store = FailThenSucceedStore(fail_count=1)
    inner_runtime = StubRuntime()
    decorator = ServerRuntimeDecorator(
        inner_runtime,
        store=fail_store,
        persistence_backoff=[0.0, 0.0],
        idle_release_timeout=None,
    )
    _setup_run(decorator)
    internal = decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    await internal.after_step_completed()

    assert fail_store.attempts == 2
    assert fail_store.last_handler is not None
    assert fail_store.last_handler.handler_id == "h1"


async def test_checkpoint_raises_after_all_retries_exhausted() -> None:
    """When all retries are exhausted, the checkpoint raises and cancels the run."""
    # 3 failures > 2 retries (backoff list has 2 entries)
    fail_store = FailThenSucceedStore(fail_count=10)
    inner_runtime = StubRuntime()
    decorator = ServerRuntimeDecorator(
        inner_runtime,
        store=fail_store,
        persistence_backoff=[0.0, 0.0],
        idle_release_timeout=None,
    )
    ext = _setup_run(decorator)
    internal = decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    with pytest.raises(RuntimeError, match="Simulated store failure"):
        await internal.after_step_completed()

    # Should have tried 1 initial + 2 retries = 3 total
    assert fail_store.attempts == 3


async def test_checkpoint_succeeds_on_last_retry() -> None:
    """Store fails twice (matching backoff length), succeeds on third attempt."""
    fail_store = FailThenSucceedStore(fail_count=2)
    inner_runtime = StubRuntime()
    decorator = ServerRuntimeDecorator(
        inner_runtime,
        store=fail_store,
        persistence_backoff=[0.0, 0.0],
        idle_release_timeout=None,
    )
    _setup_run(decorator)
    internal = decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    await internal.after_step_completed()

    assert fail_store.attempts == 3
    assert fail_store.last_handler is not None


# -- Tests: ServerRuntimeDecorator registration and lookup -----------------


def test_register_and_unregister_run(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """register_run and unregister_run manage active_runs correctly."""
    ext = _setup_run(runtime_decorator)

    assert runtime_decorator.get_active_run("h1") is ext
    assert runtime_decorator.get_run_metadata("h1") is not None

    runtime_decorator.unregister_run("h1")

    assert runtime_decorator.get_active_run("h1") is None
    assert runtime_decorator.get_run_metadata("h1") is None


def test_context_provider_strong_returns_dict(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """register_context_provider_strong provides context dict correctly."""
    expected = {"globals": {"x": 1}, "state": {}}
    _setup_run(runtime_decorator, ctx_dict=expected)

    result = runtime_decorator.get_context_dict("h1")
    assert result == expected


def test_get_context_dict_returns_none_when_not_registered(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """get_context_dict returns None for unknown run_id."""
    result = runtime_decorator.get_context_dict("nonexistent")
    assert result is None


# -- Tests: ServerExternalAdapter wrapping ---------------------------------


async def test_run_workflow_returns_server_external_adapter(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """run_workflow wraps the inner ExternalRunAdapter in a ServerExternalAdapter."""
    ext = runtime_decorator.run_workflow(
        run_id="r1",
        workflow=MagicMock(),
        init_state=MagicMock(),
    )
    assert isinstance(ext, ServerExternalAdapter)
    assert ext.run_id == "r1"


async def test_get_internal_adapter_returns_server_internal_adapter(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """get_internal_adapter wraps the inner InternalRunAdapter in a ServerInternalAdapter."""
    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)
    assert internal.run_id == "r1"


# -- Tests: ServerExternalAdapter idle notification ------------------------


async def test_external_adapter_is_notified_on_idle(
    runtime_decorator: ServerRuntimeDecorator,
) -> None:
    """ServerExternalAdapter receives idle notification from internal adapter."""
    ext = _setup_run(runtime_decorator)
    internal = runtime_decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    task = asyncio.create_task(asyncio.sleep(10))
    try:
        # Trigger idle
        single_task = [NamedTask.pull(0, task)]
        await internal.wait_for_next_task(single_task, timeout=None)

        # idle_release_timeout is None so no timer is started, but
        # the metadata should reflect idle
        meta = runtime_decorator.get_run_metadata("h1")
        assert meta is not None
        assert meta.idle_since is not None
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# -- Tests: store property -------------------------------------------------


def test_store_property(
    runtime_decorator: ServerRuntimeDecorator,
    memory_store: MemoryWorkflowStore,
) -> None:
    """The store property returns the store passed at construction."""
    assert runtime_decorator.store is memory_store


# -- Helpers for idle-timeout-enabled decorator ----------------------------


def _make_idle_decorator(
    stub_runtime: StubRuntime | None = None,
    memory_store: MemoryWorkflowStore | None = None,
    idle_timeout_ms: int = 50,
) -> tuple[ServerRuntimeDecorator, StubRuntime, MemoryWorkflowStore]:
    """Build a ServerRuntimeDecorator with a very short idle_release_timeout."""
    rt = stub_runtime or StubRuntime()
    store = memory_store or MemoryWorkflowStore()
    decorator = ServerRuntimeDecorator(
        rt,
        store=store,
        persistence_backoff=[0.0, 0.0],
        idle_release_timeout=timedelta(milliseconds=idle_timeout_ms),
    )
    return decorator, rt, store


# -- Tests: Phase 3 - Idle Suspend/Resume ---------------------------------


async def test_idle_timer_triggers_suspend() -> None:
    """When the workflow becomes idle with a short timeout, the adapter is suspended
    and the store is updated with a final checkpoint."""
    decorator, stub_rt, store = _make_idle_decorator(idle_timeout_ms=20)
    ext = _setup_run(decorator, handler_id="h1", workflow_name="test_wf")

    internal = decorator.get_internal_adapter(MagicMock())
    assert isinstance(internal, ServerInternalAdapter)

    # Trigger idle detection: single task, no timeout
    task = asyncio.create_task(asyncio.sleep(10))
    try:
        single_task = [NamedTask.pull(0, task)]
        await internal.wait_for_next_task(single_task, timeout=None)

        # The idle timer should now be running; wait for it to fire
        await asyncio.sleep(0.1)

        assert ext.is_suspended
        # The store should have been updated with the final checkpoint
        assert "h1" in store.handlers
        handler = store.handlers["h1"]
        assert handler.workflow_name == "test_wf"
        assert handler.status == "running"
        # The inner StubExternalAdapter should have been cancelled
        assert stub_rt._external.cancelled
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_send_event_on_suspended_adapter_triggers_resume() -> None:
    """After suspending, calling send_event resumes the adapter by replacing
    the inner run. Verify state transitions: suspended -> not suspended."""
    decorator, stub_rt, store = _make_idle_decorator(idle_timeout_ms=10)
    ctx_data = {"globals": {"key": "val"}, "state": {"counter": 1}}
    ext = _setup_run(
        decorator,
        handler_id="h1",
        workflow_name="test_wf",
        ctx_dict=ctx_data,
    )
    # Store a workflow ref so _resume can access it
    meta = decorator.get_run_metadata("h1")
    assert meta is not None
    meta.workflow = MagicMock()

    # Directly suspend
    await ext._suspend("h1")
    assert ext.is_suspended

    # Pre-populate the store with a "running" handler so _resume can find it
    assert "h1" in store.handlers

    # Create a fresh StubExternalAdapter that run_workflow will return on resume
    new_external = StubExternalAdapter()
    stub_rt._external = new_external

    # Patch Context.from_dict and BrokerState.from_workflow to avoid real workflow
    # machinery. _resume uses inline imports from their source modules.
    import unittest.mock as um

    mock_ctx = MagicMock()
    mock_ctx.to_dict.return_value = ctx_data

    with (
        um.patch(
            "workflows.context.Context.from_dict",
            return_value=mock_ctx,
        ),
        um.patch(
            "workflows.runtime.types.internal_state.BrokerState.from_workflow",
            return_value=MagicMock(),
        ),
    ):
        # send_event should trigger _resume internally
        tick = MagicMock()
        await ext.send_event(tick)

    assert not ext.is_suspended
    # The inner adapter should have been replaced with the new one from run_workflow
    assert ext._inner is new_external


async def test_active_event_cancels_idle_timer() -> None:
    """Calling _notify_active() cancels any pending idle release timer."""
    decorator, _, _ = _make_idle_decorator(idle_timeout_ms=5000)
    ext = _setup_run(decorator, handler_id="h1")

    # Start the idle timer
    ext._notify_idle()
    assert ext._idle_release_timer is not None
    assert not ext._idle_release_timer.done()

    # Notify active -> timer should be cancelled
    ext._notify_active()
    assert ext._idle_release_timer is None


async def test_suspend_checkpoints_before_cancelling() -> None:
    """_suspend writes a final checkpoint to the store before cancelling the inner run."""
    decorator, stub_rt, store = _make_idle_decorator()
    ctx_data = {"globals": {"x": 99}, "state": {"y": 42}}
    ext = _setup_run(decorator, handler_id="h1", workflow_name="wf1", ctx_dict=ctx_data)

    # Ensure inner not yet cancelled and store empty
    assert not stub_rt._external.cancelled
    assert "h1" not in store.handlers

    await ext._suspend("h1")

    # Store should have been updated BEFORE the cancel
    assert "h1" in store.handlers
    handler = store.handlers["h1"]
    assert handler.ctx == ctx_data
    assert handler.status == "running"
    assert handler.workflow_name == "wf1"

    # And the inner was cancelled
    assert stub_rt._external.cancelled
    assert ext.is_suspended


async def test_concurrent_suspend_is_idempotent() -> None:
    """Two concurrent _suspend calls should both succeed without errors, and
    the adapter should end up suspended exactly once."""
    decorator, stub_rt, store = _make_idle_decorator()
    ext = _setup_run(decorator, handler_id="h1", workflow_name="wf1")

    # Run two suspends concurrently
    results = await asyncio.gather(
        ext._suspend("h1"),
        ext._suspend("h1"),
        return_exceptions=True,
    )

    # Neither call should have raised
    for r in results:
        assert r is None, f"Expected None, got {r!r}"

    assert ext.is_suspended
    # Store should have a single entry
    assert "h1" in store.handlers
