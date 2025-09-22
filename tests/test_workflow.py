# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import asyncio
import logging
import sys
import time
from typing import Any, Type, Union
from unittest import mock

import pytest

from workflows.context import Context, PickleSerializer
from workflows.decorators import step
from workflows.errors import (
    WorkflowConfigurationError,
    WorkflowRuntimeError,
    WorkflowTimeoutError,
    WorkflowValidationError,
)
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.handler import WorkflowHandler
from workflows.workflow import Workflow
from workflows.testing import WorkflowTestRunner

from .conftest import (
    AnotherTestEvent,
    DummyWorkflow,
    LastEvent,
    OneTestEvent,
)


class EventWithName(Event):
    name: str


class MyStart(StartEvent):
    query: str


class MyStop(StopEvent):
    outcome: str


def test_fn() -> None:
    print("test_fn")


@pytest.mark.asyncio
async def test_workflow_initialization(workflow: Workflow) -> None:
    assert workflow._timeout == 45
    assert not workflow._disable_validation
    assert not workflow._verbose


@pytest.mark.asyncio
async def test_workflow_run(workflow: Workflow) -> None:
    r = await WorkflowTestRunner(workflow).run()
    assert r.result == "Workflow completed"


@pytest.mark.asyncio
async def test_workflow_timeout() -> None:
    class SlowWorkflow(Workflow):
        @step
        async def slow_step(self, ev: StartEvent) -> StopEvent:
            await asyncio.sleep(2.0)
            return StopEvent(result="Done")

    with pytest.raises(WorkflowTimeoutError):
        await WorkflowTestRunner(SlowWorkflow(timeout=1)).run()


@pytest.mark.asyncio
async def test_workflow_validation_unproduced_events() -> None:
    class InvalidWorkflow(Workflow):
        @step
        async def invalid_step(self, ev: StartEvent) -> None:
            pass

    with pytest.raises(
        WorkflowConfigurationError,
        match="At least one Event of type StopEvent must be returned by any step.",
    ):
        InvalidWorkflow()


@pytest.mark.asyncio
async def test_workflow_validation_unconsumed_events() -> None:
    class InvalidWorkflow(Workflow):
        @step
        async def invalid_step(self, ev: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step
        async def a_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    with pytest.raises(
        WorkflowValidationError,
        match="The following events are produced but never consumed: OneTestEvent",
    ):
        await WorkflowTestRunner(InvalidWorkflow()).run()


@pytest.mark.asyncio
async def test_workflow_validation_start_event_not_consumed() -> None:
    class InvalidWorkflow(Workflow):
        @step
        async def a_step(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent()

        @step
        async def another_step(self, ev: OneTestEvent) -> OneTestEvent:
            return OneTestEvent()

    with pytest.raises(
        WorkflowConfigurationError,
        match="At least one Event of type StartEvent must be received by any step.",
    ):
        InvalidWorkflow()


@pytest.mark.asyncio
async def test_workflow_event_propagation() -> None:
    events = []

    class EventTrackingWorkflow(Workflow):
        @step
        async def step1(self, ev: StartEvent) -> OneTestEvent:
            events.append("step1")
            return OneTestEvent()

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            events.append("step2")
            return StopEvent(result="Done")

    await WorkflowTestRunner(EventTrackingWorkflow()).run()
    assert events == ["step1", "step2"]


@pytest.mark.asyncio
async def test_workflow_sync_async_steps() -> None:
    class SyncAsyncWorkflow(Workflow):
        @step
        async def async_step(self, ev: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step
        def sync_step(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="Done")

    await WorkflowTestRunner(SyncAsyncWorkflow()).run()


@pytest.mark.asyncio
async def test_workflow_sync_steps_only() -> None:
    class SyncWorkflow(Workflow):
        @step
        def step_one(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            ctx.collect_events(ev, [StartEvent])
            return OneTestEvent()

        @step
        def step_two(self, ctx: Context, ev: OneTestEvent) -> StopEvent:
            # ctx.collect_events(ev, [OneTestEvent])
            return StopEvent()

    await WorkflowTestRunner(SyncWorkflow()).run()


@pytest.mark.asyncio
async def test_workflow_num_workers() -> None:
    class NumWorkersWorkflow(Workflow):
        @step
        async def original_step(
            self, ctx: Context, ev: StartEvent
        ) -> Union[OneTestEvent, LastEvent]:
            await ctx.store.set("num_to_collect", 3)
            ctx.send_event(OneTestEvent(test_param="test1"))
            ctx.send_event(OneTestEvent(test_param="test2"))
            ctx.send_event(OneTestEvent(test_param="test3"))

            # send one extra event
            ctx.send_event(AnotherTestEvent(another_test_param="test4"))

            return LastEvent()

        @step(num_workers=3)
        async def test_step(self, ev: OneTestEvent) -> AnotherTestEvent:
            await asyncio.sleep(1.0)
            return AnotherTestEvent(another_test_param=ev.test_param)

        @step
        async def final_step(
            self, ctx: Context, ev: Union[AnotherTestEvent, LastEvent]
        ) -> StopEvent:
            n = await ctx.store.get("num_to_collect")
            events = ctx.collect_events(ev, [AnotherTestEvent] * n)
            if events is None:
                return None  # type: ignore
            return StopEvent(result=[ev.another_test_param for ev in events])

    workflow = NumWorkersWorkflow()
    start_time = time.time()
    r = await WorkflowTestRunner(workflow).run()
    end_time = time.time()

    assert set(r.result) == {"test1", "test2", "test4"}

    # ctx should have 1 extra event
    ctx = workflow._contexts.pop()
    assert ctx
    assert "final_step" in ctx._event_buffers
    event_buffer = ctx._event_buffers["final_step"]
    assert len(event_buffer["tests.conftest.AnotherTestEvent"]) == 1

    # ensure ctx is serializable
    ctx.to_dict()

    # Check if the execution time is close to 1 second (with some tolerance)
    execution_time = end_time - start_time
    assert 1.0 <= execution_time < 1.1, (
        f"Execution time was {execution_time:.2f} seconds"
    )


@pytest.mark.asyncio
async def test_workflow_step_send_event() -> None:
    class StepSendEventWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            ctx.send_event(OneTestEvent(), step="step2")
            return None  # type: ignore

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="step2")

        @step
        async def step3(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="step3")

    workflow = StepSendEventWorkflow()
    r = await WorkflowTestRunner(workflow).run()
    assert r.result == "step2"
    ctx = workflow._contexts.pop()
    assert ("step2", "OneTestEvent") in ctx._accepted_events
    assert ("step3", "OneTestEvent") not in ctx._accepted_events


@pytest.mark.asyncio
async def test_workflow_step_send_event_to_None() -> None:
    class StepSendEventToNoneWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            ctx.send_event(OneTestEvent(), step=None)
            return  # type:ignore

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="step2")

    workflow = StepSendEventToNoneWorkflow(verbose=True)
    await WorkflowTestRunner(workflow).run()
    assert ("step2", "OneTestEvent") in workflow._contexts.pop()._accepted_events


@pytest.mark.asyncio
async def test_workflow_step_returning_bogus() -> None:
    class TestWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            return "foo"  # type:ignore

        @step
        async def step2(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step
        async def step3(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="step2")

    with pytest.raises(
        WorkflowRuntimeError,
        match="Step function step1 returned str instead of an Event instance.",
    ):
        await WorkflowTestRunner(TestWorkflow()).run()


@pytest.mark.asyncio
async def test_workflow_multiple_runs() -> None:
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result=ev.number * 2)

    runner = WorkflowTestRunner(DummyWorkflow())
    results = await asyncio.gather(
        runner.run(StartEvent(number=3)),  # type: ignore
        runner.run(StartEvent(number=42)),  # type: ignore
        runner.run(StartEvent(number=-99)),  # type: ignore
    )
    assert set([r.result for r in results]) == {6, 84, -198}


def test_add_step() -> None:
    class TestWorkflow(Workflow):
        @step
        def foo_step(self, ev: StartEvent) -> None:
            pass

    with pytest.raises(
        WorkflowValidationError,
        match="A step foo_step is already part of this workflow, please choose another name.",
    ):

        @step(workflow=TestWorkflow)
        def foo_step(ev: StartEvent) -> None:
            pass


def test_add_step_not_a_step() -> None:
    class TestWorkflow(Workflow):
        @step
        def a_ste(self, ev: StartEvent) -> None:
            pass

    def another_step(ev: StartEvent) -> None:
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step function another_step is missing the `@step` decorator.",
    ):
        TestWorkflow.add_step(another_step)


@pytest.mark.asyncio
async def test_workflow_task_raises() -> None:
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ev: StartEvent) -> StopEvent:
            raise ValueError("The step raised an error!")

    with pytest.raises(ValueError, match="The step raised an error!"):
        await WorkflowTestRunner(DummyWorkflow()).run()


def test_workflow_disable_validation() -> None:
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ev: StartEvent) -> StopEvent:
            raise ValueError("The step raised an error!")

    w = DummyWorkflow(disable_validation=True)
    w._get_steps = mock.MagicMock()  # type:ignore
    w._validate()
    w._get_steps.assert_not_called()


@pytest.mark.asyncio
async def test_workflow_continue_context() -> None:
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            cur_number = await ctx.store.get("number", default=0)
            await ctx.store.set("number", cur_number + 1)
            return StopEvent(result="Done")

    wf = DummyWorkflow()

    # first run
    r = await WorkflowTestRunner(wf).run()
    assert r.result == "Done"
    ctx = wf._contexts.pop()
    assert ctx
    assert await ctx.store.get("number") == 1

    # second run -- independent from the first
    r = await WorkflowTestRunner(wf).run()
    assert r.result == "Done"
    ctx = wf._contexts.pop()
    assert ctx
    assert await ctx.store.get("number") == 1

    # third run -- continue from the second run
    handler = wf.run(ctx=ctx)
    result = await handler
    assert handler.ctx
    assert result == "Done"
    assert await handler.ctx.store.get("number") == 2


@pytest.mark.asyncio
async def test_workflow_pickle() -> None:
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            cur_step = await ctx.store.get("step", default=0)
            await ctx.store.set("step", cur_step + 1)
            await ctx.store.set("test_fn", test_fn)
            return StopEvent(result="Done")

    wf = DummyWorkflow()
    await WorkflowTestRunner(wf).run()
    ctx = wf._contexts.copy().pop()
    assert ctx

    # by default, we can't pickle the LLM/embedding object
    with pytest.raises(ValueError):
        state_dict = ctx.to_dict()

    # if we allow pickle, then we can pickle the LLM/embedding object
    state_dict = ctx.to_dict(serializer=PickleSerializer())
    new_handler = WorkflowHandler(
        ctx=Context.from_dict(wf, state_dict, serializer=PickleSerializer())
    )
    assert new_handler.ctx

    # check that the step count is the same
    cur_step = await ctx.store.get("step")
    new_step = await new_handler.ctx.store.get("step")
    assert new_step == cur_step

    await WorkflowTestRunner(wf).run(ctx=new_handler.ctx)

    # check that the step count is incremented
    assert await new_handler.ctx.store.get("step") == cur_step + 1


@pytest.mark.asyncio
async def test_workflow_context_to_dict(workflow: Workflow) -> None:
    handler = workflow.run()
    ctx = handler.ctx

    ctx.send_event(EventWithName(name="test"))  # type:ignore

    # get the context dict
    data = ctx.to_dict()  # type:ignore

    # finish workflow
    await handler

    new_ctx = Context.from_dict(workflow, data)

    print(new_ctx._queues)
    assert new_ctx._queues["start_step"].get_nowait().name == "test"


class HumanInTheLoopWorkflow(Workflow):
    @step
    async def step1(self, ctx: Context, ev: StartEvent) -> InputRequiredEvent:
        cur_runs = await ctx.store.get("step1_runs", default=0)
        await ctx.store.set("step1_runs", cur_runs + 1)
        return InputRequiredEvent(prefix="Enter a number: ")  # type:ignore

    @step
    async def step2(self, ctx: Context, ev: HumanResponseEvent) -> StopEvent:
        cur_runs = await ctx.store.get("step2_runs", default=0)
        await ctx.store.set("step2_runs", cur_runs + 1)
        return StopEvent(result=ev.response)


@pytest.mark.asyncio
async def test_human_in_the_loop() -> None:
    # workflow should raise a timeout error because hitl only works with streaming
    with pytest.raises(WorkflowTimeoutError):
        await WorkflowTestRunner(HumanInTheLoopWorkflow(timeout=1)).run()

    # workflow should work with streaming
    workflow = HumanInTheLoopWorkflow()

    handler = workflow.run()
    assert handler.ctx
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            assert event.prefix == "Enter a number: "
            handler.ctx.send_event(HumanResponseEvent(response="42"))  # type:ignore

    final_result = await handler
    assert final_result == "42"


@pytest.mark.asyncio
async def test_human_in_the_loop_with_resume() -> None:
    # workflow should work with streaming
    workflow = HumanInTheLoopWorkflow()

    handler: WorkflowHandler = workflow.run()
    assert handler.ctx

    ctx_dict = None
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            ctx_dict = handler.ctx.to_dict()
            await handler.cancel_run()
            await asyncio.sleep(0.01)
            break

    assert handler.exception()

    new_handler = workflow.run(ctx=Context.from_dict(workflow, ctx_dict))  # type:ignore
    new_handler.ctx.send_event(HumanResponseEvent(response="42"))  # type:ignore

    final_result = await new_handler
    assert final_result == "42"

    # ensure the workflow ran each step once
    step1_runs = await new_handler.ctx.store.get("step1_runs")  # type:ignore
    step2_runs = await new_handler.ctx.store.get("step2_runs")  # type:ignore
    assert step1_runs == 1
    assert step2_runs == 1


class DummyWorkflowForConcurrentRunsTest(Workflow):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = asyncio.Lock()
        self.num_active_runs = 0

    @step
    async def step_one(self, ev: StartEvent) -> StopEvent:
        run_num = ev.get("run_num")
        async with self._lock:
            self.num_active_runs += 1
        await asyncio.sleep(0.01)
        return StopEvent(result=f"Run {run_num}: Done")

    @step
    async def _done(self, ctx: Context, ev: StopEvent) -> None:
        async with self._lock:
            self.num_active_runs -= 1
        await super()._done(ctx, ev)

    async def get_active_runs(self) -> Any:
        async with self._lock:
            return self.num_active_runs


class NumConcurrentRunsException(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "workflow",
        "desired_max_concurrent_runs",
        "expected_exception",
    ),
    [
        (
            DummyWorkflowForConcurrentRunsTest(num_concurrent_runs=1),
            1,
            type(None),
        ),
        # This workflow is not protected, and so NumConcurrentRunsException is raised
        (
            DummyWorkflowForConcurrentRunsTest(),
            1,
            NumConcurrentRunsException,
        ),
    ],
)
async def test_workflow_run_num_concurrent(
    workflow: DummyWorkflowForConcurrentRunsTest,
    desired_max_concurrent_runs: int,
    expected_exception: Type,
) -> None:
    # skip test if python version is 3.9 or lower
    if sys.version_info < (3, 10):
        pytest.skip("Skipping test for Python 3.9 or lower")

    async def _poll_workflow(
        wf: DummyWorkflowForConcurrentRunsTest, desired_max_concurrent_runs: int
    ) -> None:
        """Check that number of concurrent runs is less than desired max amount."""
        for _ in range(100):
            num_active_runs = await wf.get_active_runs()
            if num_active_runs > desired_max_concurrent_runs:
                raise NumConcurrentRunsException
            await asyncio.sleep(0.01)

    poll_task = asyncio.create_task(
        _poll_workflow(
            wf=workflow, desired_max_concurrent_runs=desired_max_concurrent_runs
        ),
    )

    tasks = []
    for ix in range(1, 5):
        tasks.append(workflow.run(run_num=ix))

    results = await asyncio.gather(*tasks)

    if not poll_task.done():
        await poll_task
    e = poll_task.exception()

    assert type(e) is expected_exception
    assert results == [f"Run {ix}: Done" for ix in range(1, 5)]


@pytest.mark.asyncio
async def test_custom_stop_event() -> None:
    class CustomEventsWorkflow(Workflow):
        @step
        async def start_step(self, ev: MyStart) -> OneTestEvent:
            return OneTestEvent()

        @step
        async def middle_step(self, ev: OneTestEvent) -> LastEvent:
            return LastEvent()

        @step
        async def end_step(self, ev: LastEvent) -> MyStop:
            return MyStop(outcome="Workflow completed")

    wf = CustomEventsWorkflow()
    assert wf._start_event_class == MyStart
    assert wf.start_event_class == wf._start_event_class
    assert wf._stop_event_class == MyStop
    assert wf.stop_event_class == wf._stop_event_class
    result: MyStop = await wf.run(query="foo")
    assert result.outcome == "Workflow completed"

    # Ensure the event types can be inferred when not passed to the init
    wf = CustomEventsWorkflow()
    assert wf._start_event_class == MyStart
    assert wf._stop_event_class == MyStop
    result = await wf.run(query="foo")
    assert result.outcome == "Workflow completed"

    # ensure that streaming exits
    r = await WorkflowTestRunner(wf).run(MyStart(query="foo"))
    assert len(r.collected) > 0


@pytest.mark.asyncio
async def test_workflow_stream_events_exits() -> None:
    class CustomEventsWorkflow(Workflow):
        @step
        async def start_step(self, ev: MyStart) -> OneTestEvent:
            return OneTestEvent()

        @step
        async def middle_step(self, ev: OneTestEvent) -> LastEvent:
            return LastEvent()

        @step
        async def end_step(self, ev: LastEvent) -> MyStop:
            return MyStop(outcome="Workflow completed")

    wf = CustomEventsWorkflow()
    handler = wf.run(query="foo")

    async def _stream_events() -> Any:
        async for event in handler.stream_events():
            continue

        return await handler

    stream_task = asyncio.create_task(_stream_events())

    result = await asyncio.wait_for(
        stream_task,
        timeout=1,
    )
    assert result.outcome == "Workflow completed"


def test_wrong_event_types() -> None:
    class RandomEvent(Event):
        pass

    class InvalidStopWorkflow(Workflow):
        @step
        async def a_step(self, ev: MyStart) -> RandomEvent:
            return RandomEvent()

    with pytest.raises(
        WorkflowConfigurationError,
        match="At least one Event of type StopEvent must be returned by any step.",
    ):
        InvalidStopWorkflow()

    class InvalidStartWorkflow(Workflow):
        @step
        async def a_step(self, ev: RandomEvent) -> StopEvent:
            return StopEvent()

    with pytest.raises(
        WorkflowConfigurationError,
        match="At least one Event of type StartEvent must be received by any step.",
    ):
        InvalidStartWorkflow()


def test__get_start_event_instance(caplog: Any) -> None:
    class CustomEvent(StartEvent):
        field: str

    e = CustomEvent(field="test")
    d = DummyWorkflow()
    d._start_event_class = CustomEvent

    # Invoke run() passing a legit start event but with additional kwargs
    with caplog.at_level(logging.WARN):
        assert d._get_start_event_instance(e, this_will_be_ignored=True) == e
        assert (
            "Keyword arguments are not supported when 'run()' is invoked with the 'start_event' parameter."
            in caplog.text
        )

    # Old style kwargs passed to the designed StartEvent
    assert type(d._get_start_event_instance(None, field="test")) is CustomEvent

    # Old style but wrong kwargs passed to the designed StartEvent
    err = "Failed creating a start event of type 'CustomEvent' with the keyword arguments: {'wrong_field': 'test'}"
    with pytest.raises(WorkflowRuntimeError, match=err):
        d._get_start_event_instance(None, wrong_field="test")


def test__ensure_start_event_class_multiple_types() -> None:
    class DummyWorkflow(Workflow):
        @step
        def one(self, ev: MyStart) -> None:
            pass

        @step
        def two(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    with pytest.raises(
        WorkflowConfigurationError,
        match="Only one type of StartEvent is allowed per workflow, found 2",
    ):
        DummyWorkflow()


def test__ensure_stop_event_class_multiple_types() -> None:
    class DummyWorkflow(Workflow):
        @step
        def one(self, ev: MyStart) -> MyStop:
            return MyStop(outcome="nope")

        @step
        def two(self, ev: MyStart) -> StopEvent:
            return StopEvent()

    with pytest.raises(
        WorkflowConfigurationError,
        match="Only one type of StopEvent is allowed per workflow, found 2",
    ):
        DummyWorkflow()


@pytest.mark.asyncio
async def test_workflow_validation_steps_cannot_accept_stop_event() -> None:
    # Test single step that incorrectly accepts StopEvent
    class InvalidWorkflowSingleStep(Workflow):
        @step
        async def start_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

        @step
        async def bad_step(self, ev: StopEvent) -> StopEvent:
            return StopEvent()

    with pytest.raises(
        WorkflowValidationError,
        match="Step 'bad_step' cannot accept StopEvent. StopEvent signals the end of the workflow. Use a different Event type instead.",
    ):
        await WorkflowTestRunner(InvalidWorkflowSingleStep()).run()
