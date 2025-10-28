# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import asyncio

from workflows.runtime.types.ticks import TickAddEvent

try:
    from typing import Union
except ImportError:
    from typing_extensions import Union

from typing import Optional

import pytest
from pydantic import BaseModel

from workflows.context import Context
from workflows.context.state_store import DictState
from workflows.decorators import step
from workflows.errors import WorkflowRuntimeError
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.testing import WorkflowTestRunner
from workflows.workflow import Workflow

from ..conftest import AnotherTestEvent, LastEvent, OneTestEvent


@pytest.mark.asyncio
async def test_collect_events() -> None:
    ev1 = OneTestEvent()
    ev2 = AnotherTestEvent()

    class TestWorkflow(Workflow):
        @step
        async def step1(self, _: StartEvent) -> OneTestEvent:
            return ev1

        @step
        async def step2(self, _: StartEvent) -> AnotherTestEvent:
            return ev2

        @step
        async def step3(
            self, ctx: Context, ev: Union[OneTestEvent, AnotherTestEvent]
        ) -> Optional[StopEvent]:
            events = ctx.collect_events(ev, [OneTestEvent, AnotherTestEvent])
            if events is None:
                return None
            return StopEvent(result=events)

    r = await WorkflowTestRunner(TestWorkflow()).run()
    assert r.result == [ev1, ev2]


@pytest.mark.asyncio
async def test_collect_events_with_extra_event_type() -> None:
    """
    Test that collect_events properly handles when an event of a different type
    arrives first, before the expected events.

    This validates that when collect_events is called with an event that's NOT
    in the expected types list, it returns None and waits for matching events.
    """

    class TestWorkflow(Workflow):
        @step
        async def start_step(
            self, ctx: Context, ev: StartEvent
        ) -> Union[OneTestEvent, AnotherTestEvent, LastEvent]:
            await ctx.store.set("num_to_collect", 2)
            await ctx.store.set("calls", 0)
            # Send a LastEvent first (not in the expected collection types)
            ctx.send_event(LastEvent())
            # Then send the events we want to collect
            ctx.send_event(OneTestEvent(test_param="first"))
            ctx.send_event(AnotherTestEvent(another_test_param="second"))
            return None  # type: ignore

        @step
        async def collector(
            self, ctx: Context, ev: Union[OneTestEvent, AnotherTestEvent, LastEvent]
        ) -> Optional[StopEvent]:
            # Track how many times this step is called
            calls = await ctx.store.get("calls")
            await ctx.store.set("calls", calls + 1)

            # Try to collect OneTestEvent and AnotherTestEvent
            # LastEvent is NOT in this list
            events = ctx.collect_events(ev, [OneTestEvent, AnotherTestEvent])
            if events is None:
                # This happens when we receive LastEvent or haven't received all events yet
                return None

            # Verify we got the right events
            assert len(events) == 2
            assert isinstance(events[0], OneTestEvent)
            assert isinstance(events[1], AnotherTestEvent)
            assert events[0].test_param == "first"
            assert events[1].another_test_param == "second"

            return StopEvent(result="collected")

    r = await WorkflowTestRunner(TestWorkflow()).run()
    assert r.result == "collected"

    # Verify the collector was called multiple times (once for each event)
    ctx = r.ctx
    assert ctx is not None
    calls = await ctx.store.get("calls")
    # Should be called at least 3 times: once for LastEvent (returns None),
    # once for OneTestEvent (returns None), once for AnotherTestEvent (returns result)
    assert calls >= 3


@pytest.mark.asyncio
async def test_get_default(workflow: Workflow) -> None:
    c1: Context[DictState] = Context(workflow)
    assert await c1.store.get("test_key", default=42) == 42


@pytest.mark.asyncio
async def test_get(ctx: Context) -> None:
    await ctx.store.set("foo", 42)
    assert await ctx.store.get("foo") == 42


@pytest.mark.asyncio
async def test_get_not_found(ctx: Context) -> None:
    with pytest.raises(ValueError):
        await ctx.store.get("foo")


@pytest.mark.asyncio
async def test_send_event_step_is_none(workflow: Workflow, ctx: Context) -> None:
    ev = Event(foo="bar")
    ctx._workflow_run(workflow, start_event=StartEvent())
    ctx.send_event(ev)
    await asyncio.sleep(0.01)
    assert ctx._broker_run is not None
    replay = ctx._broker_run._tick_log
    assert TickAddEvent(event=ev, step_name=None) in replay


@pytest.mark.asyncio
async def test_send_event_to_non_existent_step(ctx: Context) -> None:
    with pytest.raises(
        WorkflowRuntimeError, match="Step does_not_exist does not exist"
    ):
        ctx.send_event(Event(), "does_not_exist")


@pytest.mark.asyncio
async def test_send_event_to_wrong_step(ctx: Context) -> None:
    with pytest.raises(
        WorkflowRuntimeError,
        match="Step middle_step does not accept event of type <class 'workflows.events.Event'>",
    ):
        ctx.send_event(Event(), "middle_step")


@pytest.mark.asyncio
async def test_empty_inprogress_when_workflow_done(workflow: Workflow) -> None:
    result = await WorkflowTestRunner(workflow).run()
    ctx = result.ctx

    # there shouldn't be any in progress events
    assert ctx is not None
    assert ctx._broker_run is not None
    # After workflow completion, in_progress should be empty for all steps
    state = ctx._broker_run._state
    for step_name, worker_state in state.workers.items():
        assert len(worker_state.in_progress) == 0, (
            f"Step {step_name} has {len(worker_state.in_progress)} in-progress events"
        )


@pytest.mark.asyncio
async def test_wait_for_event_in_workflow() -> None:
    class TestWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> StopEvent:
            result = await ctx.wait_for_event(
                Event,
                waiter_event=Event(msg="foo"),
                waiter_id="test_id",
            )
            return StopEvent(result=result.msg)

    workflow = TestWorkflow()
    handler = workflow.run()
    assert handler.ctx
    async for ev in handler.stream_events():
        if isinstance(ev, Event) and ev.msg == "foo":
            handler.ctx.send_event(Event(msg="bar"))
            break

    result = await handler
    assert result == "bar"


class CustomState(BaseModel):
    pass


@pytest.mark.asyncio
async def test_wait_for_event_in_workflow_serialization() -> None:
    """Ensure hitl works with serialization and custom state."""

    class TestWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context[CustomState], ev: StartEvent) -> StopEvent:
            result = await ctx.wait_for_event(
                Event,
                waiter_event=Event(msg="foo"),
                waiter_id="test_id",
            )
            return StopEvent(result=result.msg)

    workflow = TestWorkflow()
    handler = workflow.run()
    ctx_dict = None

    assert handler.ctx
    async for ev in handler.stream_events():
        if isinstance(ev, Event) and ev.msg == "foo":
            ctx_dict = handler.ctx.to_dict()
            # Check that at least one worker has waiters
            assert ctx_dict["version"] == 1
            total_waiters = sum(
                len(worker_data["collected_waiters"])
                for worker_data in ctx_dict["workers"].values()
            )
            assert total_waiters == 1
            await handler.cancel_run()
            break

    # Roundtrip the context
    assert ctx_dict is not None
    # verify creating a new context has the correct state
    new_ctx = Context.from_dict(workflow, ctx_dict)
    new_handler = workflow.run(ctx=new_ctx)
    assert new_ctx._broker_run
    # Check that the waiters are properly restored
    state = new_ctx._broker_run._state
    total_waiters = sum(
        len(worker.collected_waiters) for worker in state.workers.values()
    )
    assert total_waiters == 1

    # Continue the workflow
    assert new_handler.ctx
    new_handler.ctx.send_event(Event(msg="bar"))
    result = await new_handler
    assert result == "bar"
    assert new_handler.ctx._broker_run
    # After workflow completion, there should be no more waiters
    state = new_handler.ctx._broker_run._state
    total_waiters = sum(
        len(worker.collected_waiters) for worker in state.workers.values()
    )
    assert total_waiters == 0


class Waiter1(Event):
    msg: str


class Waiter2(Event):
    msg: str


class ResultEvent(Event):
    result: str


class WaitingWorkflow(Workflow):
    @step
    async def spawn_waiters(
        self, ctx: Context, ev: StartEvent
    ) -> Union[Waiter1, Waiter2]:
        ctx.send_event(Waiter1(msg="foo"))
        ctx.send_event(Waiter2(msg="bar"))
        return None  # type: ignore

    @step
    async def waiter_one(self, ctx: Context, ev: Waiter1) -> ResultEvent:
        ctx.write_event_to_stream(InputRequiredEvent(prefix="waiter_one"))  # type: ignore

        new_ev: HumanResponseEvent = await ctx.wait_for_event(
            HumanResponseEvent,
            requirements={"waiter_id": "waiter_one"},
        )
        return ResultEvent(result=new_ev.response)

    @step
    async def waiter_two(self, ctx: Context, ev: Waiter2) -> ResultEvent:
        ctx.write_event_to_stream(InputRequiredEvent(prefix="waiter_two"))  # type: ignore

        new_ev: HumanResponseEvent = await ctx.wait_for_event(
            HumanResponseEvent,
            requirements={"waiter_id": "waiter_two"},
        )
        return ResultEvent(result=new_ev.response)

    @step
    async def collect_waiters(self, ctx: Context, ev: ResultEvent) -> StopEvent:
        events: list[ResultEvent] | None = ctx.collect_events(  # type: ignore
            ev, [ResultEvent, ResultEvent]
        )
        if events is None:
            return None  # type: ignore

        return StopEvent(result=[e.result for e in events])


@pytest.mark.asyncio
async def test_wait_for_multiple_events_in_workflow() -> None:
    workflow = WaitingWorkflow()
    handler = workflow.run()
    assert handler.ctx

    async for ev in handler.stream_events():
        if isinstance(ev, InputRequiredEvent) and ev.prefix == "waiter_one":
            handler.ctx.send_event(
                HumanResponseEvent(response="foo", waiter_id="waiter_one")  # type: ignore
            )
        elif isinstance(ev, InputRequiredEvent) and ev.prefix == "waiter_two":
            handler.ctx.send_event(
                HumanResponseEvent(response="bar", waiter_id="waiter_two")  # type: ignore
            )

    result = await handler
    assert result == ["foo", "bar"]
    assert not handler.ctx.is_running

    # serialize and resume
    ctx_dict = handler.ctx.to_dict()
    ctx = Context.from_dict(workflow, ctx_dict)
    handler = workflow.run(ctx=ctx)
    assert handler.ctx

    async for ev in handler.stream_events():
        if isinstance(ev, InputRequiredEvent) and ev.prefix == "waiter_one":
            handler.ctx.send_event(
                HumanResponseEvent(response="fizz", waiter_id="waiter_one")  # type: ignore
            )
        elif isinstance(ev, InputRequiredEvent) and ev.prefix == "waiter_two":
            handler.ctx.send_event(
                HumanResponseEvent(response="buzz", waiter_id="waiter_two")  # type: ignore
            )

    result = await handler
    assert result == ["fizz", "buzz"]


@pytest.mark.asyncio
async def test_clear(ctx: Context) -> None:
    await ctx.store.set("test_key", 42)
    await ctx.store.clear()
    res = await ctx.store.get("test_key", default=None)
    assert res is None
