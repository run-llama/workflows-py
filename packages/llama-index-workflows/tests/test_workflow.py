# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import gc
import json
import logging
import pickle
import threading
import weakref
from typing import Any, Callable, Optional, cast
from unittest import mock

import pytest
from llama_index_instrumentation.dispatcher import (
    active_instrument_tags,
    instrument_tags,
)
from pydantic import PrivateAttr
from workflows.context import Context, PickleSerializer
from workflows.decorators import step
from workflows.errors import (
    WorkflowConfigurationError,
    WorkflowRuntimeError,
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

from .conftest import (  # type: ignore[import]
    DummyWorkflow,
    OneTestEvent,
)


class EventWithName(Event):
    name: str


class MyStart(StartEvent):
    query: str


class MyStop(StopEvent):
    outcome: str


class ResumeStartEvent(StartEvent):
    topic: str


def test_fn() -> None:
    print("test_fn")


@pytest.mark.asyncio
async def test_workflow_initialization(workflow: Workflow) -> None:
    assert workflow._timeout == 45
    assert not workflow._disable_validation
    assert not workflow._verbose


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
    from workflows.context.external_context import ExternalContext

    assert isinstance(result.ctx._face, ExternalContext)
    replay = result.ctx._face._tick_log
    assert TickAddEvent(event=OneTestEvent()) in replay


@pytest.mark.asyncio
async def test_workflow_step_returning_bogus() -> None:
    class TestWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> StopEvent:
            return "foo"  # type:ignore

    with pytest.raises(
        WorkflowRuntimeError,
        match="Step function step1 returned str instead of an Event instance.",
    ):
        await WorkflowTestRunner(TestWorkflow()).run()


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


def test_workflow_disable_validation() -> None:
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ev: StartEvent) -> StopEvent:
            raise ValueError("The step raised an error!")

    w = DummyWorkflow(disable_validation=True)
    w._get_steps = mock.MagicMock()  # type: ignore[assignment]
    mock_get_steps = cast(mock.MagicMock, w._get_steps)
    w._validate()
    mock_get_steps.assert_not_called()


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
        ctx.to_dict()

    # if we allow pickle, then we can pickle the LLM/embedding object
    state_dict = ctx.to_dict(serializer=PickleSerializer())
    # Verify step count via serialized dict (integers are JSON serialized)
    assert json.loads(state_dict["state"]["state_data"]["_data"]["step"]) == 1

    # Restore and run again to verify deserialization works
    new_ctx = Context.from_dict(wf, state_dict, serializer=PickleSerializer())
    r2 = await WorkflowTestRunner(wf).run(ctx=new_ctx)
    ctx2 = r2.ctx
    assert ctx2

    # check that the step count is incremented after second run
    state_dict2 = ctx2.to_dict(serializer=PickleSerializer())
    assert json.loads(state_dict2["state"]["state_data"]["_data"]["step"]) == 2


@pytest.mark.asyncio
async def test_workflow_context_to_dict() -> None:
    ctx: Context | None = None
    new_ctx: Context | None = None
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

    from workflows.context.external_context import ExternalContext

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
        if ctx is not None and isinstance(ctx._face, ExternalContext):
            await ctx._face.shutdown()
        if new_ctx is not None and isinstance(new_ctx._face, ExternalContext):
            await new_ctx._face.shutdown()


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
    assert new_handler.ctx is not None
    ctx_dict = new_handler.ctx.to_dict()
    assert json.loads(ctx_dict["state"]["state_data"]["_data"]["step1_runs"]) == 1  # type: ignore[index]
    assert json.loads(ctx_dict["state"]["state_data"]["_data"]["step2_runs"]) == 1  # type: ignore[index]


@pytest.mark.asyncio
async def test_human_in_the_loop_resume_custom_start_event_inactive_ctx() -> None:
    class CustomHumanWorkflow(Workflow):
        @step
        async def ask(self, ctx: Context, ev: ResumeStartEvent) -> InputRequiredEvent:
            runs = await ctx.store.get("ask_runs", default=0)
            await ctx.store.set("ask_runs", runs + 1)
            return InputRequiredEvent(prefix=ev.topic)  # type: ignore[arg-type]

        @step
        async def complete(self, ctx: Context, ev: HumanResponseEvent) -> StopEvent:
            runs = await ctx.store.get("complete_runs", default=0)
            await ctx.store.set("complete_runs", runs + 1)
            return StopEvent(result=ev.response)

    workflow = CustomHumanWorkflow()
    handler: WorkflowHandler = workflow.run(topic="pizza")
    assert handler.ctx

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            break

    await handler.cancel_run()
    ctx_dict = handler.ctx.to_dict()
    assert ctx_dict["is_running"]

    resumed_ctx = Context.from_dict(workflow, ctx_dict)
    resumed_handler = workflow.run(ctx=resumed_ctx)
    resumed_handler.ctx.send_event(HumanResponseEvent(response="42"))  # type: ignore[arg-type]

    events = []
    async for event in resumed_handler.stream_events():
        events.append(event)

    assert events == [StopEvent(result="42")]

    final_result = await resumed_handler
    assert final_result == "42"

    ctx_dict = resumed_handler.ctx.to_dict()  # type: ignore[union-attr]
    assert json.loads(ctx_dict["state"]["state_data"]["_data"]["ask_runs"]) == 1
    assert json.loads(ctx_dict["state"]["state_data"]["_data"]["complete_runs"]) == 1


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


def test_run_with_invalid_start_event_raises() -> None:
    """Passing wrong type to start_event argument should raise ValueError."""

    class SimpleWorkflow(Workflow):
        @step
        async def only(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

    wf = SimpleWorkflow()

    # Pass a non-StartEvent object
    with pytest.raises(ValueError, match="must be an instance of 'StartEvent'"):
        wf.run(start_event="not a StartEvent")  # type: ignore[arg-type]

    # Also test with other invalid types
    with pytest.raises(ValueError, match="must be an instance of 'StartEvent'"):
        wf.run(start_event=42)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be an instance of 'StartEvent'"):
        wf.run(start_event={"topic": "test"})  # type: ignore[arg-type]


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
        wf_ref: weakref.ReferenceType[Workflow] = cast(
            weakref.ReferenceType[Workflow], weakref.ref(wf)
        )
        refs.append(wf_ref)
        await WorkflowTestRunner(wf).run()
        # Drop strong reference before next iteration
        del wf

    # Force GC to clear weakly-referenced registry entries
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0)

    # All weakrefs should be cleared
    assert all([r() is None for r in refs])


def test_workflow_error_no_steps_configured_message() -> None:
    class Dummy(Workflow):
        @step
        def ok(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    wf = Dummy()

    # simulate a workflow with no steps at validation time
    wf._get_steps = lambda: {}  # type: ignore[method-assign]

    with pytest.raises(
        WorkflowConfigurationError,
        match=r"Workflow 'Dummy' has no configured steps",
    ):
        wf._validate()


def test_missing_start_event_error_includes_class_name() -> None:
    class Dummy(Workflow):
        @step
        def ok(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

        @step
        def bad(self, ev: Event) -> StopEvent:
            return StopEvent()

    wf = Dummy()
    # validate only the 'bad' step to simulate missing StartEvent config
    only_bad = {"bad": wf._get_steps()["bad"]}
    wf._get_steps = lambda: only_bad  # type: ignore[method-assign]

    with pytest.raises(
        WorkflowConfigurationError,
        match=r"Workflow 'Dummy' has no @step that accepts StartEvent",
    ):
        wf._validate()


def test_missing_stop_event_error_includes_class_name() -> None:
    class Dummy(Workflow):
        @step
        def ok(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

        @step
        def bad(self, ev: StartEvent) -> None:
            return None  # type: ignore[return-value]

    wf = Dummy()
    # validate only the 'bad' step to simulate missing StopEvent config
    only_bad = {"bad": wf._get_steps()["bad"]}
    wf._get_steps = lambda: only_bad  # type: ignore[method-assign]

    with pytest.raises(
        WorkflowConfigurationError,
        match=r"Workflow 'Dummy' has no @step that returns StopEvent",
    ):
        wf._validate()


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


@pytest.mark.asyncio
async def test_inner_step_can_access_run_id_and_others_from_instrument_tags() -> None:
    # container to mutate rather than deal with nonlocal
    run_id: dict[str, str | None] = {"run_id": None, "foo": None}

    class TagReadingWorkflow(Workflow):
        @step
        async def read_tags(self, ctx: Context, ev: StartEvent) -> StopEvent:
            tags = active_instrument_tags.get()
            run_id["run_id"] = tags.get("run_id")
            run_id["foo"] = tags.get("foo")
            return StopEvent()

    wf = TagReadingWorkflow()
    with instrument_tags({"foo": "bar"}):
        handler = wf.run()
    await handler

    assert handler.run_id is not None
    assert run_id["run_id"] is not None
    assert run_id["run_id"] == handler.run_id
    assert run_id["foo"] is not None
    assert run_id["foo"] == "bar"


class Par(Event):
    id: int


class ParDone(Event):
    id: int


@pytest.mark.asyncio
async def test_workflow_parallel_resume() -> None:
    """Test that workflows with parallel workers can be serialized and resumed.

    This test verifies that:
    1. Events sent via ctx.send_event() are properly captured during serialization
    2. In-progress workers are properly restored on resume
    3. The workflow can complete after multiple serialize/resume cycles
    """
    resume_event = asyncio.Event()
    all_workers_started = asyncio.Event()
    worker_count = 0

    class ParallelResumeWorkflow(Workflow):
        @step
        async def step1(self, ev: StartEvent, ctx: Context) -> Optional[Par]:  # noqa - python 3.9 struggles here with | None
            for i in range(4):
                ctx.send_event(Par(id=i))
            return None

        @step(num_workers=4)
        async def par(self, ev: Par) -> ParDone:
            nonlocal worker_count
            # Track when workers start waiting (or complete for id=0)
            worker_count += 1
            if worker_count >= 4:
                all_workers_started.set()
            if ev.id == 0:
                return ParDone(id=ev.id)
            # Others wait for resume signal
            await resume_event.wait()
            return ParDone(id=ev.id)

        @step
        async def step3(self, ev: ParDone, ctx: Context) -> Optional[StopEvent]:  # noqa - python 3.9 struggles here with | None
            if ctx.collect_events(ev, [ParDone] * 4) is None:
                return None
            return StopEvent(result="Done")

    wf = ParallelResumeWorkflow(timeout=10)

    # First run: wait until all par workers have started, then serialize
    handler = wf.run()
    await all_workers_started.wait()
    # Small delay to ensure all workers are in BrokerState
    await asyncio.sleep(0.01)
    serialized_ctx = handler.ctx.to_dict()
    try:
        handler.cancel()
        await handler.cancel_run()
    except Exception:
        pass

    # Second run: resume and complete
    worker_count = 0
    all_workers_started.clear()
    new_handler = wf.run(ctx=Context.from_dict(wf, serialized_ctx))
    resume_event.set()
    result = await new_handler
    assert result == "Done"
