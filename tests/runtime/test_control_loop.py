# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

"""
Tests for the control_loop function in the runtime module.

The control loop is the core event processing engine that:
- Processes workflow ticks (events, step results, timeouts, cancellations)
- Manages step worker state and execution
- Coordinates event routing between steps
- Handles retries, timeouts, and failures
"""

import asyncio
import uuid
from typing import Coroutine, Optional, Union

import pytest

from workflows.decorators import step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
    StepStateChanged,
)
from workflows.errors import WorkflowCancelledByUser, WorkflowTimeoutError
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.step_function import as_step_worker_function
from workflows.workflow import Workflow
from workflows.context import Context
from workflows.runtime.control_loop import control_loop
from workflows.runtime.workflow_registry import workflow_registry
from workflows.runtime.types.ticks import TickAddEvent, TickCancelRun
from workflows.retry_policy import ConstantDelayRetryPolicy

from .conftest import MockRuntimePlugin


class IntermediateEvent(Event):
    """Test event passed between workflow steps."""

    value: int


class FinalEvent(Event):
    """Test event indicating workflow completion."""

    final_value: str


class SimpleWorkflow(Workflow):
    """
    A simple three-step workflow for testing the happy path.

    Flow:
        StartEvent -> IntermediateEvent -> FinalEvent -> StopEvent
    """

    @step
    async def start_step(self, ev: StartEvent) -> IntermediateEvent:
        """First step: receives start event and produces intermediate event."""
        return IntermediateEvent(value=42)

    @step
    async def middle_step(self, ev: IntermediateEvent) -> FinalEvent:
        """Second step: processes intermediate event and produces final event."""
        return FinalEvent(final_value=f"processed_{ev.value}")

    @step
    async def end_step(self, ev: FinalEvent) -> StopEvent:
        """Final step: receives final event and returns stop event."""
        return StopEvent(result=ev.final_value)


class CollectEv(Event):
    i: int


class CollectEv2(Event):
    j: int


class CollectMultipleEventTypesWorkflow(Workflow):
    @step
    async def accept_start(
        self, ev: StartEvent, context: Context
    ) -> Optional[CollectEv]:
        for i in range(2):
            context.send_event(CollectEv(i=i + 1))
        return None

    @step
    async def accept_collect1(self, ev: CollectEv, context: Context) -> CollectEv2:
        return CollectEv2(j=ev.i * 10)

    @step
    async def collector(
        self, ev: Union[CollectEv, CollectEv2], context: Context
    ) -> Optional[StopEvent]:
        events = context.collect_events(ev, [CollectEv, CollectEv2] * 2)
        if events is None:
            return None
        assert [type(x) for x in events] == [
            CollectEv,
            CollectEv2,
        ] * 2  # same order as expected
        events = sum(
            [
                e.i
                if isinstance(e, CollectEv)
                else e.j
                if isinstance(e, CollectEv2)
                else 0
                for e in events
            ]
        )
        return StopEvent(result=f"sum_{events}")


class CollectWorkflow(Workflow):
    @step
    async def accept_start(
        self, ev: StartEvent, context: Context
    ) -> Optional[CollectEv]:
        for i in range(4):
            context.send_event(CollectEv(i=i + 1))
        return None

    @step
    async def collector(self, ev: CollectEv, context: Context) -> Optional[StopEvent]:
        events = context.collect_events(ev, [CollectEv] * 4)
        if events is None:
            return None
        events = sum([e.i for e in events])
        return StopEvent(result=f"sum_{events}")


def run_control_loop(
    workflow: Workflow, start_event: Optional[StartEvent], plugin: MockRuntimePlugin
) -> Coroutine[None, None, StopEvent]:
    step_workers = {}
    for name, step_func in workflow._get_steps().items():
        unbound = getattr(step_func, "__func__", step_func)
        step_workers[name] = as_step_worker_function(unbound)
    ctx = Context(workflow=workflow)
    ctx._broker_run = ctx._init_broker(workflow, plugin=plugin)
    run_id = str(uuid.uuid4())
    workflow_registry.register_run(
        run_id=run_id,
        workflow=workflow,
        plugin=plugin,
        context=ctx,
        steps=step_workers,
    )

    async def _run() -> StopEvent:
        try:
            return await control_loop(
                start_event=start_event,
                init_state=BrokerState.from_workflow(workflow),
                run_id=run_id,
            )
        finally:
            workflow_registry.delete_run(run_id)

    return _run()


async def wait_for_stop_event(
    plugin: MockRuntimePlugin, timeout: float = 1.0
) -> Optional[StopEvent]:
    """
    Helper to wait for a StopEvent in the event stream.

    Args:
        plugin: The MockRuntimePlugin to read events from
        timeout: Maximum time to wait for StopEvent (default: 1.0 seconds)

    Returns:
        The StopEvent if found, None if timeout occurs
    """
    try:
        while True:
            try:
                ev = await asyncio.wait_for(
                    plugin.get_stream_event(timeout=timeout), timeout=timeout
                )
                if isinstance(ev, StopEvent):
                    return ev
            except asyncio.TimeoutError:
                return None
    except Exception:
        return None


@pytest.mark.asyncio
async def test_control_loop_happy_path(test_plugin: MockRuntimePlugin) -> None:
    """
    Test the happy path through the control loop.

    This test validates that:
    1. The control loop properly initializes with workflow state
    2. Events flow through the workflow steps in order
    3. Each step executes and produces the correct output event
    4. The workflow completes with the expected StopEvent result
    5. Step state changes are published to the event stream
    """

    result = await run_control_loop(
        workflow=SimpleWorkflow(timeout=1.0),
        start_event=StartEvent(),
        plugin=test_plugin,
    )

    # Verify the workflow completed with expected result
    assert isinstance(result, StopEvent)
    assert result.result == "processed_42"


@pytest.mark.asyncio
async def test_control_loop_with_external_event(test_plugin: MockRuntimePlugin) -> None:
    """
    Test that external events can be sent to a running workflow.

    This validates that the control loop can receive events from outside
    during execution, useful for human-in-the-loop or webhook scenarios.

    The workflow starts with no initial event, and we inject a StartEvent
    externally using the plugin's send_event method.
    """

    class ExternalTriggerWorkflow(Workflow):
        """Workflow that waits for an external event."""

        @step
        async def start_step(self, ev: StartEvent) -> StopEvent:
            """Step that processes the externally sent start event."""
            return StopEvent(result="received_external_event")

    # Setup
    workflow = ExternalTriggerWorkflow(timeout=1.0)

    result_task = asyncio.create_task(
        run_control_loop(
            workflow=workflow,
            start_event=None,
            plugin=test_plugin,
        )
    )

    # Now send an external event to trigger the workflow
    await test_plugin.send_event(TickAddEvent(event=StartEvent()))

    # Wait for completion
    result = await asyncio.wait_for(result_task, timeout=5.0)

    # Verify
    assert isinstance(result, StopEvent)
    assert result.result == "received_external_event"


@pytest.mark.asyncio
async def test_control_loop_timeout(test_plugin: MockRuntimePlugin) -> None:
    """
    Test that workflow timeout raises WorkflowTimeoutError and publishes StopEvent.

    When a workflow times out, an empty StopEvent should be published to the stream
    to signal stream closure before the exception is raised.
    """

    class SlowWorkflow(Workflow):
        @step
        async def slow(self, ev: StartEvent) -> StopEvent:
            await asyncio.sleep(0.5)
            return StopEvent(result="never")

    wf = SlowWorkflow(timeout=0.01)

    task = asyncio.create_task(
        run_control_loop(
            workflow=wf,
            start_event=StartEvent(),
            plugin=test_plugin,
        )
    )

    # Wait for the StopEvent to be published
    stop_event = await wait_for_stop_event(test_plugin)

    # Verify that the timeout exception is raised
    with pytest.raises(WorkflowTimeoutError):
        await asyncio.wait_for(task, timeout=1.0)

    # Verify an empty StopEvent was published to the stream
    assert stop_event is not None, (
        "Timeout should publish empty StopEvent to stream before raising exception"
    )
    assert stop_event.result is None, "Timeout StopEvent should have None result"


@pytest.mark.asyncio
async def test_control_loop_retry_policy(test_plugin: MockRuntimePlugin) -> None:
    """
    Test that retry policy works correctly when a step fails initially but succeeds on retry.
    """

    class RetryWorkflow(Workflow):
        def __init__(self) -> None:
            super().__init__(timeout=1.0)
            self.attempts = 0

        @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=2, delay=0))
        async def flaky(self, ev: StartEvent) -> StopEvent:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("fail once")
            return StopEvent(result=f"ok_{self.attempts}")

    wf = RetryWorkflow()

    result = await run_control_loop(
        workflow=wf,
        start_event=StartEvent(),
        plugin=test_plugin,
    )

    assert isinstance(result, StopEvent)
    assert result.result == "ok_2"


@pytest.mark.asyncio
async def test_control_loop_step_failure_publishes_stop_event(
    test_plugin: MockRuntimePlugin,
) -> None:
    """
    Test that when a step fails permanently (retries exhausted),
    an empty StopEvent is published to the stream before raising the exception.

    This allows external consumers to know the workflow stream has ended.
    """

    class FailingWorkflow(Workflow):
        @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=1, delay=0))
        async def always_fails(self, ev: StartEvent) -> StopEvent:
            raise ValueError("intentional failure")

    wf = FailingWorkflow(timeout=1.0)
    task = asyncio.create_task(
        run_control_loop(
            workflow=wf,
            start_event=StartEvent(),
            plugin=test_plugin,
        )
    )

    # Wait for the StopEvent to be published
    stop_event = await wait_for_stop_event(test_plugin)

    # Now verify the workflow raised an exception
    with pytest.raises(ValueError, match="intentional failure"):
        await asyncio.wait_for(task, timeout=1.0)

    # Verify that an empty StopEvent was published before the exception
    assert stop_event is not None, (
        "Empty StopEvent should be published to stream when step fails permanently"
    )
    assert stop_event.result is None, "Failure StopEvent should have None result"


@pytest.mark.asyncio
async def test_control_loop_waiter_resolution(test_plugin: MockRuntimePlugin) -> None:
    class Awaited(Event):
        tag: str

    class WaiterWorkflow(Workflow):
        @step
        async def start(self, ev: StartEvent, ctx: Context) -> StopEvent:
            print("waiting for event")
            awaited = await ctx.wait_for_event(
                Awaited,
                waiter_event=InputRequiredEvent(),
                requirements={"tag": "go"},
            )
            return StopEvent(result=f"got_{awaited.tag}")

    wf = WaiterWorkflow(timeout=1.0)
    task = asyncio.create_task(
        run_control_loop(
            workflow=wf,
            start_event=StartEvent(),
            plugin=test_plugin,
        )
    )

    # Let first run add the waiter
    async def wait_input_required() -> InputRequiredEvent:
        async for event in test_plugin.stream_published_events():
            if isinstance(event, InputRequiredEvent):
                return event
        raise TimeoutError("InputRequiredEvent not found")

    await asyncio.wait_for(wait_input_required(), timeout=1.0)

    # Send the awaited event that satisfies requirements
    await test_plugin.send_event(TickAddEvent(event=Awaited(tag="go")))

    result = await asyncio.wait_for(task, timeout=2.0)
    assert isinstance(result, StopEvent)
    assert result.result == "got_go"


@pytest.mark.asyncio
async def test_control_loop_input_required_published_to_stream(
    test_plugin: MockRuntimePlugin,
) -> None:
    """
    Test that InputRequiredEvent is automatically published to the outward stream.

    When a workflow step calls wait_for_event, an InputRequiredEvent should be
    automatically published to the event stream so that external consumers
    (like UIs or monitoring systems) can be notified that the workflow is
    waiting for input.
    """

    class AwaitedEvent(Event):
        value: str

    class WaitingWorkflow(Workflow):
        @step
        async def waiter(self, ev: StartEvent, ctx: Context) -> StopEvent:
            # This should cause an InputRequiredEvent to be published
            awaited = await ctx.wait_for_event(
                AwaitedEvent,
                waiter_event=InputRequiredEvent(),
            )
            return StopEvent(result=f"received_{awaited.value}")

    wf = WaitingWorkflow(timeout=2.0)
    task = asyncio.create_task(
        run_control_loop(
            workflow=wf,
            start_event=StartEvent(),
            plugin=test_plugin,
        )
    )

    # Wait for the InputRequiredEvent to appear in the stream
    input_required_found = False
    while True:
        ev = await test_plugin.get_stream_event(timeout=1.0)
        if isinstance(ev, InputRequiredEvent):
            input_required_found = True
            break
        # Skip StepStateChanged events
        if isinstance(ev, StopEvent):
            break

    assert input_required_found, "InputRequiredEvent should be published to stream"

    # Now send the awaited event to complete the workflow
    await test_plugin.send_event(TickAddEvent(event=AwaitedEvent(value="test_data")))

    # Verify workflow completes successfully
    result = await asyncio.wait_for(task, timeout=1.0)
    assert isinstance(result, StopEvent)
    assert result.result == "received_test_data"


@pytest.mark.asyncio
async def test_control_loop_collect_events_same_type(
    test_plugin: MockRuntimePlugin,
) -> None:
    wf = CollectWorkflow(timeout=1.0)
    result = await asyncio.create_task(
        run_control_loop(
            workflow=wf,
            start_event=StartEvent(),
            plugin=test_plugin,
        )
    )

    assert isinstance(result, StopEvent)
    assert result.result == "sum_10"


@pytest.mark.asyncio
async def test_control_loop_collect_events_multiple_types(
    test_plugin: MockRuntimePlugin,
) -> None:
    wf = CollectMultipleEventTypesWorkflow(timeout=1.0)
    result = await run_control_loop(
        workflow=wf,
        start_event=StartEvent(),
        plugin=test_plugin,
    )
    assert isinstance(result, StopEvent)
    assert result.result == "sum_33"


@pytest.mark.asyncio
async def test_control_loop_stream_events(test_plugin: MockRuntimePlugin) -> None:
    wf = SimpleWorkflow(timeout=5.0)
    task = asyncio.create_task(
        run_control_loop(
            workflow=wf,
            start_event=StartEvent(),
            plugin=test_plugin,
        )
    )

    # Expect at least one StepStateChanged event while running
    stream_events: list[Event] = []
    while True:
        ev = await test_plugin.get_stream_event(timeout=1.0)
        stream_events.append(ev)
        if isinstance(ev, StopEvent):
            break

    result = await asyncio.wait_for(task, timeout=2.0)
    assert isinstance(result, StopEvent)
    # Ensure at least one StepStateChanged observed
    assert len(stream_events) == 7
    assert [type(x) for x in stream_events] == [StepStateChanged] * 6 + [StopEvent]
    change_events = [x for x in stream_events if isinstance(x, StepStateChanged)]
    assert [x.step_state.name + " - " + x.name for x in change_events] == [
        "RUNNING - start_step",
        "NOT_RUNNING - start_step",
        "RUNNING - middle_step",
        "NOT_RUNNING - middle_step",
        "RUNNING - end_step",
        "NOT_RUNNING - end_step",
    ]


class SomeEvent(HumanResponseEvent):
    pass


@pytest.mark.asyncio
async def test_control_loop_per_step_routing(test_plugin: MockRuntimePlugin) -> None:
    class RouteWorkflow(Workflow):
        @step
        async def starter(self, ev: StartEvent) -> Optional[StopEvent]:
            return None

        @step
        async def first(self, ev: SomeEvent) -> StopEvent:
            return StopEvent(result="first")

        @step
        async def second(self, ev: SomeEvent) -> StopEvent:
            return StopEvent(result="second")

    wf = RouteWorkflow(timeout=1.0)
    task = asyncio.create_task(
        run_control_loop(
            workflow=wf,
            start_event=StartEvent(),
            plugin=test_plugin,
        )
    )

    # Route explicitly to the 'second' step with an accepted event type
    await test_plugin.send_event(TickAddEvent(event=SomeEvent(), step_name="second"))

    result = await asyncio.wait_for(task, timeout=2.0)
    assert isinstance(result, StopEvent)
    assert result.result == "second"


@pytest.mark.asyncio
async def test_control_loop_concurrency_queueing(
    test_plugin: MockRuntimePlugin,
) -> None:
    class LimitedWorkflow(Workflow):
        @step(num_workers=1)
        async def only_one(self, ev: StartEvent) -> StopEvent:
            # Hold to simulate long work
            await asyncio.sleep(0.01)
            return StopEvent(result="done")

    wf = LimitedWorkflow(timeout=5.0)

    task = asyncio.create_task(
        run_control_loop(
            workflow=wf,
            plugin=test_plugin,
            start_event=None,
        )
    )

    await asyncio.sleep(0)
    # Send two events quickly; with num_workers=1, second should queue (PREPARING)
    await asyncio.gather(
        *[test_plugin.send_event(TickAddEvent(event=StartEvent())) for _ in range(10)]
    )

    # Observe stream for PREPARING signal
    saw_preparing = False
    for _ in range(5):
        ev = await test_plugin.get_stream_event(timeout=1.0)
        if isinstance(ev, StepStateChanged) and ev.step_state.name == "PREPARING":
            saw_preparing = True
            break

    # Drain to completion
    result = await asyncio.wait_for(task, timeout=2.0)
    assert isinstance(result, StopEvent)
    assert saw_preparing


@pytest.mark.asyncio
async def test_control_loop_user_cancellation(test_plugin: MockRuntimePlugin) -> None:
    """
    Test that user cancellation raises WorkflowCancelledByUser and publishes StopEvent.

    When a workflow is cancelled, an empty StopEvent should be published to the stream
    to signal stream closure before the exception is raised.
    """

    class CancelWorkflow(Workflow):
        @step
        async def slow(self, ev: StartEvent) -> StopEvent:
            await asyncio.sleep(1.0)
            return StopEvent(result="never")

    wf = CancelWorkflow(timeout=5.0)

    task = asyncio.create_task(
        run_control_loop(
            workflow=wf,
            plugin=test_plugin,
            start_event=StartEvent(),
        )
    )

    # Cancel the run externally
    await asyncio.sleep(0)
    await test_plugin.send_event(TickCancelRun())

    # Wait for the StopEvent to be published
    stop_event = await wait_for_stop_event(test_plugin)

    # Verify that the cancellation exception is raised
    with pytest.raises(WorkflowCancelledByUser):
        await asyncio.wait_for(task, timeout=1.0)

    # Verify an empty StopEvent was published to the stream
    assert stop_event is not None, (
        "Cancellation should publish empty StopEvent to stream before raising exception"
    )
    assert stop_event.result is None, "Cancellation StopEvent should have None result"
