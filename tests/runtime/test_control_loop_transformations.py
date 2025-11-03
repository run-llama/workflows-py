# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

"""
Unit tests for control loop transformation functions.

These tests focus on the pure transformation functions in the control loop,
testing them in isolation without running the full async control loop.
"""

from __future__ import annotations
from typing import Any, Union

import pytest

from workflows.events import (
    Event,
    InputRequiredEvent,
    StartEvent,
    StepState,
    StepStateChanged,
    StopEvent,
)
from workflows.errors import WorkflowTimeoutError
from workflows.retry_policy import ConstantDelayRetryPolicy
from workflows.runtime.control_loop import (
    _add_or_enqueue_event,
    _process_add_event_tick,
    _process_cancel_run_tick,
    _process_publish_event_tick,
    _process_step_result_tick,
    _process_timeout_tick,
    rewind_in_progress,
)
from workflows.runtime.types.commands import (
    CommandCompleteRun,
    CommandFailWorkflow,
    CommandHalt,
    CommandPublishEvent,
    CommandQueueEvent,
    CommandRunWorker,
)
from workflows.runtime.types.internal_state import (
    BrokerConfig,
    BrokerState,
    EventAttempt,
    InProgressState,
    InternalStepConfig,
    InternalStepWorkerState,
)
from workflows.runtime.types.results import (
    AddCollectedEvent,
    AddWaiter,
    DeleteCollectedEvent,
    DeleteWaiter,
    StepWorkerFailed,
    StepWorkerResult,
    StepWorkerState,
    StepWorkerWaiter,
)
from workflows.runtime.types.ticks import (
    TickAddEvent,
    TickCancelRun,
    TickPublishEvent,
    TickStepResult,
    TickTimeout,
)
from workflows.decorators import StepConfig


class MyTestEvent(Event):
    value: int


class OtherEvent(Event):
    data: str


@pytest.fixture
def base_state() -> BrokerState:
    """Create a minimal BrokerState for testing."""
    step_config = StepConfig(
        accepted_events=[MyTestEvent, StartEvent],
        event_name="ev",
        return_types=[StopEvent, OtherEvent, type(None)],
        context_parameter="ctx",
        retry_policy=None,
        num_workers=1,
        resources=[],
    )
    return BrokerState(
        is_running=True,
        config=BrokerConfig(
            steps={
                "test_step": InternalStepConfig(
                    accepted_events=[MyTestEvent, StartEvent],
                    retry_policy=None,
                    num_workers=1,
                )
            },
            timeout=None,
        ),
        workers={
            "test_step": InternalStepWorkerState(
                queue=[],
                config=step_config,
                in_progress=[],
                collected_events={},
                collected_waiters=[],
            )
        },
    )


def add_worker(state: BrokerState, event: Event, worker_id: int = 0) -> None:
    """Helper to add an in-progress worker to state."""
    state.workers["test_step"].in_progress.append(
        InProgressState(
            event=event,
            worker_id=worker_id,
            shared_state=StepWorkerState(
                step_name="test_step",
                collected_events={},
                collected_waiters=[],
            ),
            attempts=0,
            first_attempt_at=100.0,
        )
    )


@pytest.mark.parametrize(
    "result,expected_commands",
    [
        (StopEvent(result="done"), [StepStateChanged, StopEvent, CommandCompleteRun]),
        (OtherEvent(data="next"), [StepStateChanged, CommandQueueEvent]),
        (
            InputRequiredEvent(),
            [StepStateChanged, InputRequiredEvent, CommandQueueEvent],
        ),
        (None, [StepStateChanged]),
    ],
)
def test_step_worker_results(
    base_state: BrokerState, result: Union[Event, None], expected_commands: list
) -> None:
    """Test different step worker result types."""
    event = MyTestEvent(value=42)
    add_worker(base_state, event)

    tick = TickStepResult(
        step_name="test_step",
        worker_id=0,
        event=event,
        result=[StepWorkerResult(result=result)],
    )

    new_state, commands = _process_step_result_tick(tick, base_state, now_seconds=110.0)

    # Check expected command types
    for i, expected_type in enumerate(expected_commands):
        if isinstance(expected_type, type) and issubclass(expected_type, Event):
            command = commands[i]
            assert isinstance(command, CommandPublishEvent)
            assert isinstance(command.event, expected_type)
        else:
            assert isinstance(commands[i], expected_type)

    # Worker should be removed from in_progress
    assert len(new_state.workers["test_step"].in_progress) == 0


def test_step_worker_failed_with_retry(base_state: BrokerState) -> None:
    """Test that failures with retry policy queue a retry."""
    base_state.workers["test_step"].config.retry_policy = ConstantDelayRetryPolicy(
        maximum_attempts=3, delay=1.0
    )
    event = MyTestEvent(value=42)
    add_worker(base_state, event)

    tick: TickStepResult[Any] = TickStepResult(
        step_name="test_step",
        worker_id=0,
        event=event,
        result=[StepWorkerFailed(exception=ValueError("test"), failed_at=110.0)],
    )

    _, commands = _process_step_result_tick(tick, base_state, now_seconds=110.0)

    # Should queue retry
    queue_cmds = [c for c in commands if isinstance(c, CommandQueueEvent)]
    assert len(queue_cmds) == 1
    assert queue_cmds[0].attempts == 1

    # First command should be NOT_RUNNING state change before re-queue
    assert isinstance(commands[0], CommandPublishEvent)
    assert isinstance(commands[0].event, StepStateChanged)
    assert commands[0].event.step_state == StepState.NOT_RUNNING


def test_step_worker_failed_without_retry(base_state: BrokerState) -> None:
    """Test that failures without retry fail the workflow."""
    event = MyTestEvent(value=42)
    add_worker(base_state, event)

    tick: TickStepResult[Any] = TickStepResult(
        step_name="test_step",
        worker_id=0,
        event=event,
        result=[StepWorkerFailed(exception=ValueError("test"), failed_at=110.0)],
    )

    new_state, commands = _process_step_result_tick(tick, base_state, now_seconds=110.0)

    assert new_state.is_running is False
    assert any(isinstance(c, CommandFailWorkflow) for c in commands)


def test_collected_events(base_state: BrokerState) -> None:
    """Test AddCollectedEvent and DeleteCollectedEvent."""
    event = MyTestEvent(value=42)
    add_worker(base_state, event)

    # Add event
    tick: TickStepResult[Any] = TickStepResult(
        step_name="test_step",
        worker_id=0,
        event=event,
        result=[AddCollectedEvent(event_id="buf1", event=OtherEvent(data="e1"))],
    )
    new_state, _ = _process_step_result_tick(tick, base_state, now_seconds=110.0)
    assert "buf1" in new_state.workers["test_step"].collected_events

    # Delete event
    add_worker(new_state, event)
    tick = TickStepResult(
        step_name="test_step",
        worker_id=0,
        event=event,
        result=[
            StepWorkerResult(result=StopEvent()),
            DeleteCollectedEvent(event_id="buf1"),
        ],
    )
    new_state, _ = _process_step_result_tick(tick, new_state, now_seconds=110.0)
    assert "buf1" not in new_state.workers["test_step"].collected_events


def test_waiters(base_state: BrokerState) -> None:
    """Test AddWaiter and DeleteWaiter."""
    event = MyTestEvent(value=42)
    add_worker(base_state, event)

    # Add waiter
    tick: TickStepResult[Any] = TickStepResult(
        step_name="test_step",
        worker_id=0,
        event=event,
        result=[
            AddWaiter(
                waiter_id="w1",
                waiter_event=InputRequiredEvent(),
                requirements={},
                timeout=None,
                event_type=OtherEvent,
            )
        ],
    )
    new_state, _ = _process_step_result_tick(tick, base_state, now_seconds=110.0)
    assert len(new_state.workers["test_step"].collected_waiters) == 1

    # Delete waiter
    add_worker(new_state, event)
    tick = TickStepResult(
        step_name="test_step",
        worker_id=0,
        event=event,
        result=[
            StepWorkerResult(result=StopEvent()),
            DeleteWaiter(waiter_id="w1"),
        ],
    )
    new_state, _ = _process_step_result_tick(tick, new_state, now_seconds=110.0)
    assert len(new_state.workers["test_step"].collected_waiters) == 0


def test_start_event_sets_running(base_state: BrokerState) -> None:
    """Test that StartEvent sets is_running to True."""
    base_state.is_running = False
    tick = TickAddEvent(event=StartEvent())
    new_state, _ = _process_add_event_tick(tick, base_state, now_seconds=100.0)
    assert new_state.is_running is True


def test_event_routing(base_state: BrokerState) -> None:
    """Test that events are routed to accepting steps."""
    tick = TickAddEvent(event=MyTestEvent(value=42))
    new_state, commands = _process_add_event_tick(tick, base_state, now_seconds=100.0)

    run_cmds = [c for c in commands if isinstance(c, CommandRunWorker)]
    assert len(run_cmds) == 1
    assert run_cmds[0].step_name == "test_step"


def test_per_step_explicit_routing_accepts_only_matching_types(
    base_state: BrokerState,
) -> None:
    """Explicit routing with step_name must still satisfy accepted event types."""
    # base_state only has test_step that accepts MyTestEvent and StartEvent
    # Explicitly target test_step with MyTestEvent → should run
    tick_ok = TickAddEvent(event=MyTestEvent(value=1), step_name="test_step")
    _, cmds_ok = _process_add_event_tick(tick_ok, base_state, now_seconds=100.0)
    assert any(isinstance(c, CommandRunWorker) for c in cmds_ok)

    # Explicitly target an unknown step → should not run anything
    tick_bad = TickAddEvent(event=MyTestEvent(value=1), step_name="unknown")
    _, cmds_bad = _process_add_event_tick(tick_bad, base_state, now_seconds=100.0)
    assert not any(isinstance(c, CommandRunWorker) for c in cmds_bad)


def test_explicit_routing_requires_acceptance(base_state: BrokerState) -> None:
    """Explicit step routing should still require accepted event types."""
    # Add a second step that does NOT accept MyTestEvent
    other_step_cfg = StepConfig(
        accepted_events=[StartEvent],
        event_name="ev",
        return_types=[StopEvent, OtherEvent, type(None)],
        context_parameter="ctx",
        retry_policy=None,
        num_workers=1,
        resources=[],
    )
    base_state.config.steps["other_step"] = InternalStepConfig(
        accepted_events=[StartEvent], retry_policy=None, num_workers=1
    )
    base_state.workers["other_step"] = InternalStepWorkerState(
        queue=[],
        config=other_step_cfg,
        in_progress=[],
        collected_events={},
        collected_waiters=[],
    )

    # Try to route MyTestEvent explicitly to non-accepting step → should not start
    tick = TickAddEvent(event=MyTestEvent(value=1), step_name="other_step")
    _, commands = _process_add_event_tick(tick, base_state, now_seconds=100.0)
    assert not any(
        isinstance(c, CommandRunWorker) and c.step_name == "other_step"
        for c in commands
    )

    # Explicitly route to accepting step → should start
    tick_ok = TickAddEvent(event=MyTestEvent(value=2), step_name="test_step")
    _, commands_ok = _process_add_event_tick(tick_ok, base_state, now_seconds=100.0)
    assert any(
        isinstance(c, CommandRunWorker) and c.step_name == "test_step"
        for c in commands_ok
    )


def test_waiter_resolution(base_state: BrokerState) -> None:
    """Test that events matching waiters trigger step re-execution."""
    original_event = MyTestEvent(value=1)
    waiter = StepWorkerWaiter(
        waiter_id="w1",
        event=original_event,
        waiting_for_event=OtherEvent,
        requirements={"data": "expected"},
        has_requirements=True,
        resolved_event=None,
    )
    base_state.workers["test_step"].collected_waiters.append(waiter)

    tick = TickAddEvent(event=OtherEvent(data="expected"))
    new_state, commands = _process_add_event_tick(tick, base_state, now_seconds=100.0)

    assert (
        new_state.workers["test_step"].collected_waiters[0].resolved_event is not None
    )
    run_cmds = [c for c in commands if isinstance(c, CommandRunWorker)]
    assert any(c.event == original_event for c in run_cmds)


def test_step_state_changed_names(base_state: BrokerState) -> None:
    """Verify input/output event names on StepStateChanged use actual event types."""
    input_ev = MyTestEvent(value=7)
    add_worker(base_state, input_ev)

    # Return a regular Event → output_event_name should be its type, and input_event_name should be str(type(input))
    tick = TickStepResult(
        step_name="test_step",
        worker_id=0,
        event=input_ev,
        result=[StepWorkerResult(result=OtherEvent(data="x"))],
    )
    _, commands = _process_step_result_tick(tick, base_state, now_seconds=110.0)
    assert isinstance(commands[0], CommandPublishEvent)
    assert isinstance(commands[0].event, StepStateChanged)
    ev = commands[0].event
    assert ev.input_event_name == str(type(input_ev))
    assert ev.output_event_name == str(type(OtherEvent(data="x")))

    # Return StopEvent → output_event_name should be None
    add_worker(base_state, input_ev)
    tick2 = TickStepResult(
        step_name="test_step",
        worker_id=0,
        event=input_ev,
        result=[StepWorkerResult(result=StopEvent(result="done"))],
    )
    _, commands2 = _process_step_result_tick(tick2, base_state, now_seconds=110.0)
    assert isinstance(commands2[0], CommandPublishEvent)
    assert isinstance(commands2[0].event, StepStateChanged)
    ev2 = commands2[0].event
    assert ev2.input_event_name == str(type(input_ev))
    assert ev2.output_event_name == "<class 'workflows.events.StopEvent'>"


def test_cancel_run(base_state: BrokerState) -> None:
    """Test that cancel sets not running and halts."""
    tick = TickCancelRun()
    new_state, commands = _process_cancel_run_tick(tick, base_state)

    assert new_state.is_running is False
    assert len(commands) == 2
    assert isinstance(commands[0], CommandPublishEvent)
    assert isinstance(commands[1], CommandHalt)


def test_publish_event(base_state: BrokerState) -> None:
    """Test that publish events pass through without state changes."""
    event = MyTestEvent(value=42)
    tick = TickPublishEvent(event=event)
    new_state, commands = _process_publish_event_tick(tick, base_state)

    assert new_state is base_state
    assert len(commands) == 1
    assert isinstance(commands[0], CommandPublishEvent)


def test_timeout(base_state: BrokerState) -> None:
    """Test that timeout sets not running and halts with error."""
    tick = TickTimeout(timeout=10.0)
    new_state, commands = _process_timeout_tick(tick, base_state)

    assert new_state.is_running is False
    assert isinstance(commands[1], CommandHalt)
    assert isinstance(commands[1].exception, WorkflowTimeoutError)


def test_add_when_capacity_available(base_state: BrokerState) -> None:
    """Test that events start immediately when capacity available."""
    event = MyTestEvent(value=42)
    commands = _add_or_enqueue_event(
        EventAttempt(event=event),
        "test_step",
        base_state.workers["test_step"],
        now_seconds=100.0,
    )

    assert len(base_state.workers["test_step"].in_progress) == 1
    assert any(isinstance(c, CommandRunWorker) for c in commands)
    assert any(
        isinstance(c, CommandPublishEvent)
        and isinstance(c.event, StepStateChanged)
        and c.event.step_state == StepState.RUNNING
        for c in commands
    )


def test_enqueue_when_no_capacity(base_state: BrokerState) -> None:
    """Test that events queue when no capacity available."""
    # Fill capacity
    add_worker(base_state, MyTestEvent(value=1))

    # Try to add another
    event = MyTestEvent(value=42)
    commands = _add_or_enqueue_event(
        EventAttempt(event=event),
        "test_step",
        base_state.workers["test_step"],
        now_seconds=100.0,
    )

    assert len(base_state.workers["test_step"].queue) == 1
    # PREPARING should be published when we enqueue
    assert isinstance(commands[0], CommandPublishEvent)
    assert isinstance(commands[0].event, StepStateChanged)
    assert commands[0].event.step_state == StepState.PREPARING


def test_rewind_restarts_workers(base_state: BrokerState) -> None:
    """Test that in_progress workers are restarted."""
    base_state.workers["test_step"].config.num_workers = 2
    base_state.config.steps["test_step"].num_workers = 2

    add_worker(base_state, MyTestEvent(value=1), worker_id=0)
    add_worker(base_state, MyTestEvent(value=2), worker_id=1)

    new_state, commands = rewind_in_progress(base_state, now_seconds=120.0)

    # Both should be restarted
    run_cmds = [c for c in commands if isinstance(c, CommandRunWorker)]
    assert len(run_cmds) == 2
    assert len(new_state.workers["test_step"].in_progress) == 2
