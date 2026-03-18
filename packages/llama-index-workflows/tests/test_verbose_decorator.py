# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for VerboseDecorator and _VerboseInternalRunAdapter."""

from __future__ import annotations

import logging
from typing import Any

from workflows import Workflow, step
from workflows.events import Event, StartEvent, StepState, StepStateChanged, StopEvent
from workflows.runtime.types.plugin import InternalRunAdapter, WaitResult, WorkflowTick
from workflows.runtime.types.ticks import (
    TickAddEvent,
    TickCancelRun,
    TickIdleCheck,
    TickIdleRelease,
    TickPublishEvent,
    TickStepResult,
    TickTimeout,
    TickWaiterTimeout,
)
from workflows.runtime.verbose import _VerboseInternalRunAdapter
from workflows.testing import WorkflowTestRunner


class FakeInternalRunAdapter(InternalRunAdapter):
    """Minimal fake adapter that records events written to the stream."""

    def __init__(self) -> None:
        self.written_events: list[Event] = []

    @property
    def run_id(self) -> str:
        return "fake-run-id"

    async def write_to_event_stream(self, event: Event) -> None:
        self.written_events.append(event)

    async def get_now(self) -> float:
        raise NotImplementedError

    async def send_event(self, tick: WorkflowTick) -> None:
        raise NotImplementedError

    async def wait_receive(
        self,
        timeout_seconds: float | None = None,
    ) -> WaitResult:
        raise NotImplementedError

    async def sleep(self, seconds: float) -> None:
        raise NotImplementedError


def _make_step_state_changed(
    name: str = "my_step",
    step_state: StepState = StepState.RUNNING,
    worker_id: str = "0",
    input_event_name: str = "StartEvent",
    output_event_name: str | None = None,
) -> StepStateChanged:
    return StepStateChanged(
        name=name,
        step_state=step_state,
        worker_id=worker_id,
        input_event_name=input_event_name,
        output_event_name=output_event_name,
    )


# -- write_to_event_stream tests (step state changes) --


async def test_verbose_print_step_running(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    event = _make_step_state_changed(
        name="my_step", step_state=StepState.RUNNING, worker_id="0"
    )
    await adapter.write_to_event_stream(event)

    captured = capsys.readouterr()
    assert "Running step my_step (worker 0)" in captured.out


async def test_verbose_print_step_completed_with_event(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    event = _make_step_state_changed(
        name="my_step",
        step_state=StepState.NOT_RUNNING,
        output_event_name="MyEvent",
        worker_id="2",
    )
    await adapter.write_to_event_stream(event)

    captured = capsys.readouterr()
    assert "Step my_step produced event MyEvent (worker 2)" in captured.out


async def test_verbose_print_step_completed_no_event(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    event = _make_step_state_changed(
        name="my_step",
        step_state=StepState.NOT_RUNNING,
        output_event_name=None,
        worker_id="1",
    )
    await adapter.write_to_event_stream(event)

    captured = capsys.readouterr()
    assert "Step my_step produced no event (worker 1)" in captured.out


async def test_verbose_print_step_preparing(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    event = _make_step_state_changed(
        name="my_step",
        step_state=StepState.PREPARING,
        worker_id="<enqueued>",
    )
    await adapter.write_to_event_stream(event)

    captured = capsys.readouterr()
    assert "Step my_step enqueued (waiting for capacity)" in captured.out


async def test_verbose_auto_detects_logger_when_info_enabled(caplog: Any) -> None:
    logger = logging.getLogger("workflows.verbose")
    old_level = logger.level
    try:
        logger.setLevel(logging.INFO)
        from workflows.runtime.verbose import _resolve_output

        output = _resolve_output()
        fake = FakeInternalRunAdapter()
        adapter = _VerboseInternalRunAdapter(fake, output=output)

        event = _make_step_state_changed(name="my_step", step_state=StepState.RUNNING)
        with caplog.at_level(logging.INFO, logger="workflows.verbose"):
            await adapter.write_to_event_stream(event)

        assert "Running step my_step (worker 0)" in caplog.text
    finally:
        logger.setLevel(old_level)


async def test_verbose_falls_back_to_print_by_default(capsys: Any) -> None:
    logger = logging.getLogger("workflows.verbose")
    old_level = logger.level
    try:
        logger.setLevel(logging.NOTSET)
        from workflows.runtime.verbose import _resolve_output

        output = _resolve_output()
        assert output is print
    finally:
        logger.setLevel(old_level)


async def test_verbose_forwards_events() -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    event = _make_step_state_changed(name="my_step", step_state=StepState.RUNNING)
    await adapter.write_to_event_stream(event)

    assert len(fake.written_events) == 1
    assert fake.written_events[0] is event


# -- on_tick tests (tick-level logging) --


async def test_verbose_tick_add_event(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    tick = TickAddEvent(event=StartEvent())
    await adapter.on_tick(tick)

    captured = capsys.readouterr()
    assert "Event added: StartEvent" in captured.out


async def test_verbose_tick_add_event_targeted(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    tick = TickAddEvent(event=StartEvent(), step_name="retrieve")
    await adapter.on_tick(tick)

    captured = capsys.readouterr()
    assert "Event added: StartEvent -> retrieve" in captured.out


async def test_verbose_tick_publish_event(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    tick = TickPublishEvent(event=StopEvent(result="done"))
    await adapter.on_tick(tick)

    captured = capsys.readouterr()
    assert "Publish: StopEvent" in captured.out


async def test_verbose_tick_timeout(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    tick = TickTimeout(timeout=30.0)
    await adapter.on_tick(tick)

    captured = capsys.readouterr()
    assert "Timeout: 30.0s elapsed" in captured.out


async def test_verbose_tick_waiter_timeout(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    tick = TickWaiterTimeout(step_name="my_step", waiter_id="w-123")
    await adapter.on_tick(tick)

    captured = capsys.readouterr()
    assert "Waiter timeout: step my_step waiter w-123" in captured.out


async def test_verbose_tick_cancel_run(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    tick = TickCancelRun()
    await adapter.on_tick(tick)

    captured = capsys.readouterr()
    assert "Cancelled" in captured.out


async def test_verbose_tick_idle_check(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    tick = TickIdleCheck()
    await adapter.on_tick(tick)

    captured = capsys.readouterr()
    assert "Idle check" in captured.out


async def test_verbose_tick_idle_release(capsys: Any) -> None:
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    tick = TickIdleRelease()
    await adapter.on_tick(tick)

    captured = capsys.readouterr()
    assert "Idle release" in captured.out


async def test_verbose_tick_step_result_silent(capsys: Any) -> None:
    """TickStepResult should not produce on_tick output (covered by StepStateChanged)."""
    fake = FakeInternalRunAdapter()
    adapter = _VerboseInternalRunAdapter(fake, output=print)

    from workflows.runtime.types.results import StepWorkerResult

    tick = TickStepResult(
        step_name="my_step",
        worker_id=0,
        event=StartEvent(),
        result=[StepWorkerResult(result=StopEvent(result="done"))],
    )
    await adapter.on_tick(tick)

    captured = capsys.readouterr()
    assert captured.out == ""


# -- Integration test --


class TwoStepWorkflow(Workflow):
    @step
    async def first(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


async def test_workflow_verbose_integration(capsys: Any) -> None:
    wf = TwoStepWorkflow(verbose=True)
    result = await WorkflowTestRunner(wf).run()
    assert result.result == "done"

    captured = capsys.readouterr()
    assert "Running step first (worker 0)" in captured.out
    assert "Step first produced event" in captured.out
