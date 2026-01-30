"""Runtime matrix tests - testing workflows against both BasicRuntime and DBOSRuntime.

All workflow classes are defined at module level so they can be registered with
DBOS once at module initialization time, avoiding repeated init/destroy cycles.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional, Union

import pytest
from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.errors import WorkflowTimeoutError
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.plugins.basic import BasicRuntime
from workflows.runtime.types.plugin import Runtime
from workflows.testing import WorkflowTestRunner
from workflows.workflow import Workflow

# -- Fixtures --


@pytest.fixture(
    params=[
        pytest.param("basic", id="basic"),
    ]
)
async def runtime(
    request: pytest.FixtureRequest,
) -> AsyncGenerator[Runtime, None]:
    """Yield an unlaunched runtime.

    For DBOS, returns the module-scoped runtime (already created, not yet launched).
    Each test must call runtime.launch() after creating workflows.
    """
    if request.param == "basic":
        rt = BasicRuntime()
        try:
            yield rt
        finally:
            rt.destroy()


# -- Shared event types --


class OneTestEvent(Event):
    test_param: str = Field(default="test")


class AnotherTestEvent(Event):
    another_test_param: str = Field(default="another_test")


class LastEvent(Event):
    pass


class MyStart(StartEvent):
    query: str


class MyStop(StopEvent):
    outcome: str


# -- Workflow definitions (module level for DBOS registration) --


class SimpleWorkflow(Workflow):
    @step
    async def start_step(self, ev: StartEvent) -> OneTestEvent:
        return OneTestEvent()

    @step
    async def middle_step(self, ev: OneTestEvent) -> LastEvent:
        return LastEvent()

    @step
    async def end_step(self, ev: LastEvent) -> StopEvent:
        return StopEvent(result="Workflow completed")


class SlowWorkflow(Workflow):
    @step
    async def slow_step(self, ev: StartEvent) -> StopEvent:
        await asyncio.sleep(2.0)
        return StopEvent(result="Done")


class EventTrackingWorkflow(Workflow):
    """Workflow that tracks events in an external list."""

    tracked_events: list[str] = []

    @step
    async def step1(self, ev: StartEvent) -> OneTestEvent:
        self.tracked_events.append("step1")
        return OneTestEvent()

    @step
    async def step2(self, ev: OneTestEvent) -> StopEvent:
        self.tracked_events.append("step2")
        return StopEvent(result="Done")


class SyncAsyncWorkflow(Workflow):
    @step
    async def async_step(self, ev: StartEvent) -> OneTestEvent:
        return OneTestEvent()

    @step
    def sync_step(self, ev: OneTestEvent) -> StopEvent:
        return StopEvent(result="Done")


class SyncWorkflow(Workflow):
    @step
    def step_one(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
        ctx.collect_events(ev, [StartEvent])
        return OneTestEvent()

    @step
    def step_two(self, ctx: Context, ev: OneTestEvent) -> StopEvent:
        return StopEvent()


class MultiRunWorkflow(Workflow):
    @step
    async def step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=ev.number * 2)  # type: ignore


class ErrorWorkflow(Workflow):
    @step
    async def step(self, ev: StartEvent) -> StopEvent:
        raise ValueError("The step raised an error!")


class CounterWorkflow(Workflow):
    @step
    async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        cur_number = await ctx.store.get("number", default=0)
        new_number = cur_number + 1
        await ctx.store.set("number", new_number)
        return StopEvent(result=new_number)


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


class NumWorkersWorkflow(Workflow):
    @step
    async def original_step(
        self, ctx: Context, ev: StartEvent
    ) -> Union[OneTestEvent, LastEvent]:
        await ctx.store.set("num_to_collect", 3)
        ctx.send_event(OneTestEvent(test_param="test1"))
        ctx.send_event(OneTestEvent(test_param="test2"))
        ctx.send_event(OneTestEvent(test_param="test3"))
        ctx.send_event(AnotherTestEvent(another_test_param="test4"))
        return LastEvent()

    @step(num_workers=3)
    async def test_step(self, ev: OneTestEvent) -> AnotherTestEvent:
        # Note: await_count logic needs to be injected per-test
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


class HITLWorkflow(Workflow):
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


class StreamWorkflow(Workflow):
    @step
    async def chat(self, ctx: Context, ev: StartEvent) -> StopEvent:
        async def stream_messages() -> AsyncGenerator[str, None]:
            resp = "Paul Graham is a British-American computer scientist, entrepreneur, vc, and writer."
            for word in resp.split():
                yield word

        async for w in stream_messages():
            ctx.write_event_to_stream(Event(msg=w))

        return StopEvent(result=None)


class ErrorStreamingWorkflow(Workflow):
    @step
    async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(OneTestEvent(test_param="foo"))
        raise ValueError("The step raised an error!")


class TimeoutStreamingWorkflow(Workflow):
    @step
    async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(OneTestEvent(test_param="foo"))
        await asyncio.sleep(2)
        return StopEvent()


# -- Tests --


@pytest.mark.asyncio
async def test_workflow_run(runtime: Runtime) -> None:
    workflow = SimpleWorkflow(runtime=runtime)
    runtime.launch()
    r = await WorkflowTestRunner(workflow).run()
    assert r.result == "Workflow completed"


@pytest.mark.asyncio
async def test_workflow_timeout(runtime: Runtime) -> None:
    wf = SlowWorkflow(timeout=0.1, runtime=runtime)
    runtime.launch()
    with pytest.raises(WorkflowTimeoutError):
        await WorkflowTestRunner(wf).run()


@pytest.mark.asyncio
async def test_workflow_event_propagation(runtime: Runtime) -> None:
    events: list[str] = []

    class LocalEventTrackingWorkflow(Workflow):
        @step
        async def step1(self, ev: StartEvent) -> OneTestEvent:
            events.append("step1")
            return OneTestEvent()

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            events.append("step2")
            return StopEvent(result="Done")

    wf = LocalEventTrackingWorkflow(runtime=runtime)
    runtime.launch()
    await WorkflowTestRunner(wf).run()
    assert events == ["step1", "step2"]


@pytest.mark.asyncio
async def test_workflow_sync_async_steps(runtime: Runtime) -> None:
    wf = SyncAsyncWorkflow(runtime=runtime)
    runtime.launch()
    await WorkflowTestRunner(wf).run()


@pytest.mark.asyncio
async def test_workflow_sync_steps_only(runtime: Runtime) -> None:
    wf = SyncWorkflow(runtime=runtime)
    runtime.launch()
    await WorkflowTestRunner(wf).run()


@pytest.mark.asyncio
async def test_workflow_multiple_runs(runtime: Runtime) -> None:
    wf = MultiRunWorkflow(runtime=runtime)
    runtime.launch()
    runner = WorkflowTestRunner(wf)
    results = await asyncio.gather(
        runner.run(StartEvent(number=3)),  # type: ignore
        runner.run(StartEvent(number=42)),  # type: ignore
        runner.run(StartEvent(number=-99)),  # type: ignore
    )
    assert set([r.result for r in results]) == {6, 84, -198}


@pytest.mark.asyncio
async def test_workflow_task_raises(runtime: Runtime) -> None:
    wf = ErrorWorkflow(runtime=runtime)
    runtime.launch()
    with pytest.raises(ValueError, match="The step raised an error!"):
        await WorkflowTestRunner(wf).run()


@pytest.mark.asyncio
async def test_workflow_step_send_event(runtime: Runtime) -> None:
    workflow = StepSendEventWorkflow(runtime=runtime)
    runtime.launch()
    r = await WorkflowTestRunner(workflow).run()
    assert r.result == "step2"


@pytest.mark.asyncio
async def test_workflow_num_workers(runtime: Runtime) -> None:
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

    class LocalNumWorkersWorkflow(Workflow):
        @step
        async def original_step(
            self, ctx: Context, ev: StartEvent
        ) -> Union[OneTestEvent, LastEvent]:
            await ctx.store.set("num_to_collect", 3)
            # Send test4 first to ensure it's pulled from receive_queue
            # before test_step workers complete. Events are pulled one per
            # iteration, so ordering in receive_queue determines delivery order.
            ctx.send_event(AnotherTestEvent(another_test_param="test4"))
            ctx.send_event(OneTestEvent(test_param="test1"))
            ctx.send_event(OneTestEvent(test_param="test2"))
            ctx.send_event(OneTestEvent(test_param="test3"))
            return LastEvent()

        @step(num_workers=3)
        async def test_step(self, ev: OneTestEvent) -> AnotherTestEvent:
            await await_count(3)
            return AnotherTestEvent(another_test_param=ev.test_param)

        @step
        async def final_step(
            self, ctx: Context, ev: Union[AnotherTestEvent, LastEvent]
        ) -> Optional[StopEvent]:
            n = await ctx.store.get("num_to_collect")
            events = ctx.collect_events(ev, [AnotherTestEvent] * n)
            if events is None:
                return None
            return StopEvent(result=[ev.another_test_param for ev in events])

    workflow = LocalNumWorkersWorkflow(timeout=10, runtime=runtime)
    runtime.launch()
    r = await WorkflowTestRunner(workflow).run()

    assert "test4" in set(r.result)
    assert len({"test1", "test2", "test3"} - set(r.result)) == 1


@pytest.mark.asyncio
async def test_custom_stop_event(runtime: Runtime) -> None:
    wf = CustomEventsWorkflow(runtime=runtime)
    runtime.launch()

    assert wf._start_event_class == MyStart
    assert wf.start_event_class == wf._start_event_class
    assert wf._stop_event_class == MyStop
    assert wf.stop_event_class == wf._stop_event_class
    result: MyStop = await wf.run(query="foo")
    assert result.outcome == "Workflow completed"

    # Run again with the same workflow instance
    assert wf._start_event_class == MyStart
    assert wf._stop_event_class == MyStop
    result = await wf.run(query="foo")
    assert result.outcome == "Workflow completed"

    # ensure that streaming exits
    r = await WorkflowTestRunner(wf).run(MyStart(query="foo"))
    assert len(r.collected) > 0


@pytest.mark.asyncio
async def test_human_in_the_loop(runtime: Runtime) -> None:
    # Create both workflow instances before launch
    timeout_wf = HITLWorkflow(timeout=0.01, runtime=runtime)
    workflow = HITLWorkflow(runtime=runtime)
    runtime.launch()

    # workflow should raise a timeout error because hitl only works with streaming
    with pytest.raises(WorkflowTimeoutError):
        await WorkflowTestRunner(timeout_wf).run()

    # workflow should work with streaming
    handler = workflow.run()
    assert handler.ctx
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            assert event.prefix == "Enter a number: "
            handler.ctx.send_event(HumanResponseEvent(response="42"))  # type:ignore

    final_result = await handler
    assert final_result == "42"


@pytest.mark.asyncio
async def test_workflow_stream_events_exits(runtime: Runtime) -> None:
    wf = CustomEventsWorkflow(runtime=runtime)
    runtime.launch()
    handler = wf.run(query="foo")

    async def _stream_events() -> MyStop:
        async for event in handler.stream_events():
            continue
        return await handler

    stream_task = asyncio.create_task(_stream_events())
    result = await asyncio.wait_for(stream_task, timeout=10)
    assert result.outcome == "Workflow completed"


# -- Streaming tests --


@pytest.mark.asyncio
async def test_streaming_e2e(runtime: Runtime) -> None:
    wf = StreamWorkflow(runtime=runtime)
    runtime.launch()
    test_runner = WorkflowTestRunner(wf)
    r = await test_runner.run(expose_internal=False, exclude_events=[StopEvent])
    assert all("msg" in ev for ev in r.collected)


@pytest.mark.asyncio
async def test_streaming_task_raised(runtime: Runtime) -> None:
    wf = ErrorStreamingWorkflow(runtime=runtime)
    runtime.launch()
    r = wf.run()

    async for ev in r.stream_events():
        if not isinstance(ev, StopEvent):
            assert ev.test_param == "foo"

    with pytest.raises(ValueError, match="The step raised an error!"):
        await r


@pytest.mark.asyncio
async def test_streaming_task_timeout(runtime: Runtime) -> None:
    wf = TimeoutStreamingWorkflow(timeout=0.1, runtime=runtime)
    runtime.launch()
    r = wf.run()

    async for ev in r.stream_events():
        if not isinstance(ev, StopEvent):
            assert ev.test_param == "foo"

    with pytest.raises(WorkflowTimeoutError, match="Operation timed out"):
        await r
