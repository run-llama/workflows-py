# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import gc
import logging
import pickle
import threading
import weakref
from typing import Any, Callable, Union
from unittest import mock

from pydantic import PrivateAttr
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
from workflows.runtime.types.ticks import TickAddEvent
from workflows.testing import WorkflowTestRunner
from workflows.workflow import Workflow

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
        await WorkflowTestRunner(SlowWorkflow(timeout=0.1)).run()


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
    signal = asyncio.Event()
    lock = asyncio.Lock()
    counter = 0

    async def await_count(count: int) -> None:
        nonlocal counter
        async with lock:
            counter += 1
            if counter == count:
                signal.set()
                return
        await signal.wait()

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
            await await_count(3)  # wait for all 3 to be waiting

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

    workflow = NumWorkersWorkflow(timeout=1)
    r = await WorkflowTestRunner(workflow).run()

    assert "test4" in set(r.result)
    assert len({"test1", "test2", "test3"} - set(r.result)) == 1

    # Ensure ctx is serializable
    ctx = r.ctx
    assert ctx
    assert ctx._broker_run is not None
    ctx.to_dict()


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
    ctx = r.ctx
    replay = ctx._broker_run._runtime.replay()  # type:ignore
    assert TickAddEvent(OneTestEvent(), step_name="step2") in replay


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
    result = await WorkflowTestRunner(workflow).run()
    assert result.ctx._broker_run is not None
    replay = result.ctx._broker_run._runtime.replay()  # type:ignore
    assert TickAddEvent(OneTestEvent()) in replay


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
        TestWorkflow.add_step(another_step)  # type: ignore


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
    ctx = r.ctx
    assert ctx
    assert await ctx.store.get("number") == 1

    # second run -- independent from the first
    r = await WorkflowTestRunner(wf).run()
    assert r.result == "Done"
    ctx = r.ctx
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

    wf = DummyWorkflow(timeout=1)
    r = await WorkflowTestRunner(wf).run()
    ctx = r.ctx
    assert ctx

    # by default, we can't pickle the LLM/embedding object
    with pytest.raises(ValueError):
        state_dict = ctx.to_dict()

    # if we allow pickle, then we can pickle the LLM/embedding object
    state_dict = ctx.to_dict(serializer=PickleSerializer())
    new_ctx = Context.from_dict(wf, state_dict, serializer=PickleSerializer())
    assert await new_ctx.store.get("step") == 1
    new_handler = WorkflowHandler(ctx=new_ctx)
    assert new_handler.ctx
    assert await new_handler.ctx.store.get("step") == 1

    # check that the step count is the same
    cur_step = await ctx.store.get("step")
    new_step = await new_handler.ctx.store.get("step")
    assert new_step == cur_step
    await WorkflowTestRunner(wf).run(ctx=new_handler.ctx)

    # check that the step count is incremented
    assert await new_handler.ctx.store.get("step") == cur_step + 1


@pytest.mark.asyncio
async def test_workflow_context_to_dict() -> None:
    ctx: Union[Context, None] = None
    new_ctx: Union[Context, None] = None
    signal_continue = asyncio.Event()
    signal_ready = asyncio.Event()
    run_count = 0

    class StallableWorkflow(Workflow):
        @step
        async def step1(self, ev: StartEvent) -> EventWithName:
            nonlocal run_count
            run_count += 1
            if run_count > 1:
                raise ValueError("Start ran more than once")
            return EventWithName(name="test")

        @step
        async def step2(self, ev: EventWithName) -> StopEvent:
            signal_ready.set()
            await signal_continue.wait()
            return StopEvent(result="Done")

    workflow = StallableWorkflow()
    try:
        handler = workflow.run()
        ctx = handler.ctx
        await signal_ready.wait()
        # get the context dict
        data = ctx.to_dict()  # type:ignore

        await handler.cancel_run()

        new_ctx = Context.from_dict(workflow, data)
        handler2 = workflow.run(ctx=new_ctx)
        signal_continue.set()
        await handler2
    finally:
        if ctx is not None and ctx._broker_run is not None:
            await ctx._broker_run.shutdown()
        if new_ctx is not None and new_ctx._broker_run is not None:
            await new_ctx._broker_run.shutdown()


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
        await WorkflowTestRunner(HumanInTheLoopWorkflow(timeout=0.01)).run()

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
        self.num_active_runs_history: list[int] = []

    @step
    async def step_one(self, ev: StartEvent) -> StopEvent:
        run_num = ev.get("run_num")
        async with self._lock:
            self.num_active_runs += 1
            self.num_active_runs_history.append(self.num_active_runs)
        await asyncio.sleep(0.01)
        async with self._lock:
            self.num_active_runs -= 1
        return StopEvent(result=f"Run {run_num}: Done")

    async def get_active_runs(self) -> list[int]:
        return self.num_active_runs_history


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "workflow_factory",
        "validate_max_concurrent_runs",
    ),
    [
        (
            lambda: DummyWorkflowForConcurrentRunsTest(num_concurrent_runs=1),
            lambda actual_max_concurrent_runs: actual_max_concurrent_runs == 1,
        ),
        # This workflow is not protected, and so NumConcurrentRunsException is raised
        (
            lambda: DummyWorkflowForConcurrentRunsTest(),
            lambda actual_max_concurrent_runs: actual_max_concurrent_runs > 1,
        ),
    ],
)
async def test_workflow_run_num_concurrent(
    workflow_factory: Callable[[], DummyWorkflowForConcurrentRunsTest],
    validate_max_concurrent_runs: Callable[[int], bool],
) -> None:
    workflow = workflow_factory()
    results = await asyncio.gather(*[workflow.run(run_num=ix) for ix in range(1, 5)])
    max_concurrent_runs = max(workflow.num_active_runs_history)
    assert validate_max_concurrent_runs(max_concurrent_runs)
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


class RandomEvent(Event):
    pass


class InvalidStopWorkflow(Workflow):
    @step
    async def a_step(self, ev: MyStart) -> RandomEvent:
        return RandomEvent()


class InvalidStartWorkflow(Workflow):
    @step
    async def a_step(self, ev: RandomEvent) -> StopEvent:
        return StopEvent()


def test_wrong_event_types() -> None:
    with pytest.raises(
        WorkflowConfigurationError,
        match="At least one Event of type StopEvent must be returned by any step.",
    ):
        InvalidStopWorkflow()

    with pytest.raises(
        WorkflowConfigurationError,
        match="At least one Event of type StartEvent must be received by any step.",
    ):
        InvalidStartWorkflow()


class LockEvent(Event):
    key: Any


class LockResponseEvent(Event):
    key: Any


class NonSerializableRequirement:
    def __init__(self, be_serializable: bool = False) -> None:
        if be_serializable:
            self.lock = None
        else:
            self.lock = threading.Lock()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, NonSerializableRequirement)


class NonSerializableRequirementsWorkflow(Workflow):
    @step
    async def wait_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        # Create a waiter with a non-picklable requirement value
        await ctx.wait_for_event(
            LockResponseEvent,
            waiter_event=InputRequiredEvent(),
            waiter_id="lock_wait",
            requirements={"key": NonSerializableRequirement()},
            timeout=5,
        )
        return StopEvent(result="Done")


@pytest.mark.asyncio
async def test_human_in_the_loop_waiter_with_nonserializable_requirements_pickle_resume() -> (
    None
):
    with pytest.raises(TypeError, match="cannot pickle '_thread.lock'"):
        pickle.dumps(NonSerializableRequirement())

    workflow = NonSerializableRequirementsWorkflow()

    handler: WorkflowHandler = workflow.run()
    assert handler.ctx

    ctx_dict: dict[str, Any] | None = None

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            ctx_dict = handler.ctx.to_dict(serializer=PickleSerializer())
            await handler.cancel_run()
            break

    # Restore and resume
    new_ctx = Context.from_dict(workflow, ctx_dict or {}, serializer=PickleSerializer())
    new_handler = workflow.run(ctx=new_ctx)
    new_handler.ctx.send_event(
        LockResponseEvent(key=NonSerializableRequirement(be_serializable=True))
    )
    final_result = await new_handler
    assert final_result == "Done"


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


def test_get_workflow_events() -> None:
    class DummyWorkflow(Workflow):
        @step
        def one(self, ev: MyStart) -> MyStop:
            return MyStop(outcome="nope")

    events = DummyWorkflow().events
    assert len(events) == 2
    event_names = [e.__name__ for e in events]
    assert "MyStop" in event_names
    assert "MyStart" in event_names


@pytest.mark.asyncio
async def test_workflow_instances_garbage_collected_after_completion() -> None:
    # test for memory leaks
    class TinyWorkflow(Workflow):
        @step
        async def only(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

    refs: list[weakref.ReferenceType[Workflow]] = []

    for _ in range(10):
        wf = TinyWorkflow()
        refs.append(weakref.ref(wf))
        await WorkflowTestRunner(wf).run()
        # Drop strong reference before next iteration
        del wf

    # Force GC to clear weakly-referenced registry entries
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0)

    # All weakrefs should be cleared
    assert all([r() is None for r in refs])


class SomeEvent(StartEvent):
    _not_serializable: threading.Lock | None = PrivateAttr(default=None)


@pytest.mark.asyncio
async def test_workflow_non_picklable_event() -> None:
    step_started = asyncio.Event()
    step_continued = asyncio.Event()

    class DummyWorkflow(Workflow):
        @step
        async def step(self, ev: SomeEvent) -> StopEvent:
            step_started.set()
            await step_continued.wait()
            return StopEvent()

    start_event = SomeEvent()
    start_event._not_serializable = threading.Lock()
    wf = DummyWorkflow()
    handler = wf.run(start_event=start_event)
    await step_started.wait()
    handler.ctx.to_dict()
    step_continued.set()
    await handler
