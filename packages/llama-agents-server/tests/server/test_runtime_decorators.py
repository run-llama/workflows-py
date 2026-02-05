# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for BaseRuntimeDecorator, BaseInternalRunAdapterDecorator, and BaseExternalRunAdapterDecorator."""

from __future__ import annotations

from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

import pytest
from llama_agents.server.runtime_decorators import (
    BaseExternalRunAdapterDecorator,
    BaseInternalRunAdapterDecorator,
    BaseRuntimeDecorator,
)
from workflows.context.state_store import StateStore
from workflows.events import Event, StopEvent
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    RegisteredWorkflow,
    Runtime,
    WaitResult,
    WaitResultTimeout,
)
from workflows.runtime.types.ticks import TickAddEvent, WorkflowTick

# ---------------------------------------------------------------------------
# Fixtures: concrete stubs of the abstract interfaces
# ---------------------------------------------------------------------------


class StubInternalAdapter(InternalRunAdapter):
    """Minimal concrete InternalRunAdapter for testing."""

    def __init__(self, run_id: str = "internal-run-1") -> None:
        self._run_id = run_id
        self.events_written: list[Event] = []
        self.ticks_sent: list[WorkflowTick] = []
        self.closed = False
        self.finalized = False

    @property
    def run_id(self) -> str:
        return self._run_id

    async def write_to_event_stream(self, event: Event) -> None:
        self.events_written.append(event)

    async def get_now(self) -> float:
        return 1000.0

    async def send_event(self, tick: WorkflowTick) -> None:
        self.ticks_sent.append(tick)

    async def wait_receive(self, timeout_seconds: float | None = None) -> WaitResult:
        return WaitResultTimeout()

    async def close(self) -> None:
        self.closed = True

    async def finalize_step(self) -> None:
        self.finalized = True

    def get_state_store(self) -> StateStore[Any] | None:
        return None


class StubExternalAdapter(ExternalRunAdapter):
    """Minimal concrete ExternalRunAdapter for testing."""

    def __init__(self, run_id: str = "external-run-1") -> None:
        self._run_id = run_id
        self.ticks_sent: list[WorkflowTick] = []
        self.closed = False
        self.cancelled = False

    @property
    def run_id(self) -> str:
        return self._run_id

    async def send_event(self, tick: WorkflowTick) -> None:
        self.ticks_sent.append(tick)

    async def stream_published_events(self) -> AsyncGenerator[Event, None]:
        yield StopEvent(result="done")

    async def close(self) -> None:
        self.closed = True

    async def get_result(self) -> StopEvent:
        return StopEvent(result="done")

    async def cancel(self) -> None:
        self.cancelled = True

    def get_state_store(self) -> StateStore[Any] | None:
        return None


class StubRuntime(Runtime):
    """Minimal concrete Runtime for testing."""

    def __init__(self) -> None:
        self.launched = False
        self.destroyed = False
        self.tracked: list[Any] = []

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
        return StubExternalAdapter(run_id)

    def get_internal_adapter(self, workflow: Any) -> InternalRunAdapter:
        return StubInternalAdapter()

    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        return StubExternalAdapter(run_id)

    def launch(self) -> None:
        self.launched = True

    def destroy(self) -> None:
        self.destroyed = True

    def track_workflow(self, workflow: Any) -> None:
        self.tracked.append(workflow)

    def get_registered(self, workflow: Any) -> RegisteredWorkflow | None:
        return None


@pytest.fixture
def stub_runtime() -> StubRuntime:
    return StubRuntime()


@pytest.fixture
def stub_internal() -> StubInternalAdapter:
    return StubInternalAdapter()


@pytest.fixture
def stub_external() -> StubExternalAdapter:
    return StubExternalAdapter()


# ---------------------------------------------------------------------------
# BaseRuntimeDecorator tests
# ---------------------------------------------------------------------------


def test_runtime_decorator_forwards_register(stub_runtime: StubRuntime) -> None:
    decorator = BaseRuntimeDecorator(stub_runtime)
    sentinel = object()
    result = decorator.register(sentinel)  # type: ignore[arg-type]
    assert result.workflow is sentinel


def test_runtime_decorator_forwards_run_workflow(stub_runtime: StubRuntime) -> None:
    decorator = BaseRuntimeDecorator(stub_runtime)
    adapter = decorator.run_workflow("r1", MagicMock(), MagicMock())
    assert isinstance(adapter, StubExternalAdapter)
    assert adapter.run_id == "r1"


def test_runtime_decorator_forwards_get_internal_adapter(
    stub_runtime: StubRuntime,
) -> None:
    decorator = BaseRuntimeDecorator(stub_runtime)
    adapter = decorator.get_internal_adapter(MagicMock())
    assert isinstance(adapter, StubInternalAdapter)


def test_runtime_decorator_forwards_get_external_adapter(
    stub_runtime: StubRuntime,
) -> None:
    decorator = BaseRuntimeDecorator(stub_runtime)
    adapter = decorator.get_external_adapter("r2")
    assert isinstance(adapter, StubExternalAdapter)
    assert adapter.run_id == "r2"


def test_runtime_decorator_forwards_launch(stub_runtime: StubRuntime) -> None:
    decorator = BaseRuntimeDecorator(stub_runtime)
    decorator.launch()
    assert stub_runtime.launched


def test_runtime_decorator_forwards_destroy(stub_runtime: StubRuntime) -> None:
    decorator = BaseRuntimeDecorator(stub_runtime)
    decorator.destroy()
    assert stub_runtime.destroyed


def test_runtime_decorator_forwards_track_workflow(stub_runtime: StubRuntime) -> None:
    decorator = BaseRuntimeDecorator(stub_runtime)
    wf = object()
    decorator.track_workflow(wf)  # type: ignore[arg-type]
    assert wf in stub_runtime.tracked


def test_runtime_decorator_forwards_get_registered(stub_runtime: StubRuntime) -> None:
    decorator = BaseRuntimeDecorator(stub_runtime)
    assert decorator.get_registered(MagicMock()) is None


# ---------------------------------------------------------------------------
# BaseInternalRunAdapterDecorator tests
# ---------------------------------------------------------------------------


def test_internal_decorator_forwards_run_id(stub_internal: StubInternalAdapter) -> None:
    decorator = BaseInternalRunAdapterDecorator(stub_internal)
    assert decorator.run_id == "internal-run-1"


async def test_internal_decorator_forwards_write_to_event_stream(
    stub_internal: StubInternalAdapter,
) -> None:
    decorator = BaseInternalRunAdapterDecorator(stub_internal)
    event = StopEvent(result="test")
    await decorator.write_to_event_stream(event)
    assert stub_internal.events_written == [event]


async def test_internal_decorator_forwards_get_now(
    stub_internal: StubInternalAdapter,
) -> None:
    decorator = BaseInternalRunAdapterDecorator(stub_internal)
    assert await decorator.get_now() == 1000.0


async def test_internal_decorator_forwards_send_event(
    stub_internal: StubInternalAdapter,
) -> None:
    decorator = BaseInternalRunAdapterDecorator(stub_internal)
    tick = TickAddEvent(event=StopEvent(result="x"))
    await decorator.send_event(tick)
    assert stub_internal.ticks_sent == [tick]


async def test_internal_decorator_forwards_wait_receive(
    stub_internal: StubInternalAdapter,
) -> None:
    decorator = BaseInternalRunAdapterDecorator(stub_internal)
    result = await decorator.wait_receive(timeout_seconds=1.0)
    assert isinstance(result, WaitResultTimeout)


async def test_internal_decorator_forwards_close(
    stub_internal: StubInternalAdapter,
) -> None:
    decorator = BaseInternalRunAdapterDecorator(stub_internal)
    await decorator.close()
    assert stub_internal.closed


async def test_internal_decorator_forwards_finalize_step(
    stub_internal: StubInternalAdapter,
) -> None:
    decorator = BaseInternalRunAdapterDecorator(stub_internal)
    await decorator.finalize_step()
    assert stub_internal.finalized


def test_internal_decorator_forwards_get_state_store(
    stub_internal: StubInternalAdapter,
) -> None:
    decorator = BaseInternalRunAdapterDecorator(stub_internal)
    assert decorator.get_state_store() is None


# ---------------------------------------------------------------------------
# BaseExternalRunAdapterDecorator tests
# ---------------------------------------------------------------------------


def test_external_decorator_forwards_run_id(
    stub_external: StubExternalAdapter,
) -> None:
    decorator = BaseExternalRunAdapterDecorator(stub_external)
    assert decorator.run_id == "external-run-1"


async def test_external_decorator_forwards_send_event(
    stub_external: StubExternalAdapter,
) -> None:
    decorator = BaseExternalRunAdapterDecorator(stub_external)
    tick = TickAddEvent(event=StopEvent(result="x"))
    await decorator.send_event(tick)
    assert stub_external.ticks_sent == [tick]


async def test_external_decorator_forwards_stream_published_events(
    stub_external: StubExternalAdapter,
) -> None:
    decorator = BaseExternalRunAdapterDecorator(stub_external)
    events = [ev async for ev in decorator.stream_published_events()]
    assert len(events) == 1
    assert isinstance(events[0], StopEvent)


async def test_external_decorator_forwards_close(
    stub_external: StubExternalAdapter,
) -> None:
    decorator = BaseExternalRunAdapterDecorator(stub_external)
    await decorator.close()
    assert stub_external.closed


async def test_external_decorator_forwards_get_result(
    stub_external: StubExternalAdapter,
) -> None:
    decorator = BaseExternalRunAdapterDecorator(stub_external)
    result = await decorator.get_result()
    assert isinstance(result, StopEvent)
    assert result.result == "done"


async def test_external_decorator_forwards_cancel(
    stub_external: StubExternalAdapter,
) -> None:
    decorator = BaseExternalRunAdapterDecorator(stub_external)
    await decorator.cancel()
    assert stub_external.cancelled


def test_external_decorator_forwards_get_state_store(
    stub_external: StubExternalAdapter,
) -> None:
    decorator = BaseExternalRunAdapterDecorator(stub_external)
    assert decorator.get_state_store() is None


# ---------------------------------------------------------------------------
# Subclass override tests — verify that subclasses can selectively override
# ---------------------------------------------------------------------------


async def test_internal_subclass_can_override_single_method(
    stub_internal: StubInternalAdapter,
) -> None:
    """A subclass that overrides get_now() still forwards everything else."""

    class CustomInternal(BaseInternalRunAdapterDecorator):
        async def get_now(self) -> float:
            return 42.0

    decorator = CustomInternal(stub_internal)
    assert await decorator.get_now() == 42.0
    # Other methods still forward
    assert decorator.run_id == "internal-run-1"
    await decorator.close()
    assert stub_internal.closed


async def test_external_subclass_can_override_single_method(
    stub_external: StubExternalAdapter,
) -> None:
    """A subclass that overrides get_result() still forwards everything else."""

    class CustomExternal(BaseExternalRunAdapterDecorator):
        async def get_result(self) -> StopEvent:
            return StopEvent(result="custom")

    decorator = CustomExternal(stub_external)
    result = await decorator.get_result()
    assert result.result == "custom"
    # Other methods still forward
    assert decorator.run_id == "external-run-1"
    await decorator.close()
    assert stub_external.closed


def test_runtime_subclass_can_override_single_method(
    stub_runtime: StubRuntime,
) -> None:
    """A subclass that overrides launch() still forwards everything else."""

    class CustomRuntime(BaseRuntimeDecorator):
        def __init__(self, inner: Runtime) -> None:
            super().__init__(inner)
            self.custom_launched = False

        def launch(self) -> None:
            self.custom_launched = True
            self._inner.launch()

    decorator = CustomRuntime(stub_runtime)
    decorator.launch()
    assert decorator.custom_launched
    assert stub_runtime.launched
    # Other methods still forward
    assert decorator.get_registered(MagicMock()) is None
