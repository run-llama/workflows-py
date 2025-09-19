# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import json
import sys

try:
    from typing import Union
except ImportError:
    from typing_extensions import Union

from typing import Optional
from unittest import mock

import pytest
from pydantic import BaseModel

from workflows.context import Context
from workflows.context.state_store import DictState
from workflows.decorators import StepConfig, step
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

from ..conftest import AnotherTestEvent, OneTestEvent


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


def test_send_event_step_is_none(ctx: Context) -> None:
    ctx._queues = {"step1": mock.MagicMock(), "step2": mock.MagicMock()}
    ev = Event(foo="bar")
    ctx.send_event(ev)
    for q in ctx._queues.values():
        q.put_nowait.assert_called_with(ev)  # type: ignore
    assert ctx._broker_log == [ev]


def test_send_event_to_non_existent_step(ctx: Context) -> None:
    with pytest.raises(
        WorkflowRuntimeError, match="Step does_not_exist does not exist"
    ):
        ctx.send_event(Event(), "does_not_exist")


def test_send_event_to_wrong_step(ctx: Context) -> None:
    ctx._step_configs["step"] = StepConfig(  # type: ignore[attr-defined]
        accepted_events=[],
        event_name="test_event",
        return_types=[],
        context_parameter="",
        num_workers=99,
        retry_policy=None,
        resources=[],
    )

    with pytest.raises(
        WorkflowRuntimeError,
        match="Step step does not accept event of type <class 'workflows.events.Event'>",
    ):
        ctx.send_event(Event(), "step")


def test_send_event_to_step(workflow: Workflow) -> None:
    step2 = mock.MagicMock()
    step2.__step_config.accepted_events = [Event]

    workflow._get_steps = mock.MagicMock(  # type: ignore
        return_value={"step1": mock.MagicMock(), "step2": step2}
    )

    ctx: Context[DictState] = Context(workflow=workflow)
    ctx._queues = {"step1": mock.MagicMock(), "step2": mock.MagicMock()}

    ev = Event(foo="bar")
    ctx.send_event(ev, "step2")

    ctx._queues["step1"].put_nowait.assert_not_called()  # type: ignore
    ctx._queues["step2"].put_nowait.assert_called_with(ev)  # type: ignore


def test_get_result(ctx: Context) -> None:
    ctx._retval = 42
    assert ctx.get_result() == 42


def test_to_dict_with_events_buffer(ctx: Context) -> None:
    ctx.collect_events(OneTestEvent(), [OneTestEvent, AnotherTestEvent])
    assert json.dumps(ctx.to_dict())


@pytest.mark.asyncio
async def test_empty_inprogress_when_workflow_done(workflow: Workflow) -> None:
    await WorkflowTestRunner(workflow).run()
    ctx = workflow._contexts.pop()

    # there shouldn't be any in progress events
    assert ctx is not None
    for inprogress_list in ctx._in_progress.values():
        assert len(inprogress_list) == 0


@pytest.mark.asyncio
async def test_wait_for_event(ctx: Context) -> None:
    # skip test if python version is 3.9 or lower
    if sys.version_info < (3, 10):
        pytest.skip("Skipping test for Python 3.9 or lower")

    wait_job = asyncio.create_task(ctx.wait_for_event(Event))
    await asyncio.sleep(0.01)
    ctx.send_event(Event(msg="foo"))
    ev = await wait_job
    assert ev.msg == "foo"


@pytest.mark.asyncio
async def test_wait_for_event_with_requirements(ctx: Context) -> None:
    # skip test if python version is 3.9 or lower
    if sys.version_info < (3, 10):
        pytest.skip("Skipping test for Python 3.9 or lower")

    wait_job = asyncio.create_task(
        ctx.wait_for_event(Event, requirements={"msg": "foo"})
    )
    await asyncio.sleep(0.01)
    ctx.send_event(Event(msg="bar"))
    ctx.send_event(Event(msg="foo"))
    ev = await wait_job
    assert ev.msg == "foo"


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
            assert len(ctx_dict["waiting_ids"]) == 1
            await handler.cancel_run()
            break

    # Roundtrip the context
    assert ctx_dict is not None
    new_ctx = Context.from_dict(workflow, ctx_dict)
    assert len(new_ctx._waiting_ids) == 1
    new_handler = workflow.run(ctx=new_ctx)

    # Continue the workflow
    assert new_handler.ctx
    new_handler.ctx.send_event(Event(msg="bar"))

    result = await new_handler
    assert result == "bar"
    assert len(new_handler.ctx._waiting_ids) == 0


@pytest.mark.asyncio
async def test_prompt_and_wait(ctx: Context) -> None:
    prompt_id = "test_prompt_and_wait"
    prompt_event = InputRequiredEvent(prefix="test_prompt_and_wait")  # type: ignore
    expected_event = HumanResponseEvent
    requirements = {"waiter_id": "test_prompt_and_wait"}
    timeout = 10

    waiting_task = asyncio.create_task(
        ctx.wait_for_event(
            expected_event,
            waiter_id=prompt_id,
            waiter_event=prompt_event,
            timeout=timeout,
            requirements=requirements,
        )
    )
    await asyncio.sleep(0.01)
    ctx.send_event(HumanResponseEvent(response="foo", waiter_id="test_prompt_and_wait"))  # type: ignore

    result = await waiting_task
    assert result.response == "foo"


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
            {"waiter_id": "waiter_one"},  # type: ignore
        )
        return ResultEvent(result=new_ev.response)

    @step
    async def waiter_two(self, ctx: Context, ev: Waiter2) -> ResultEvent:
        ctx.write_event_to_stream(InputRequiredEvent(prefix="waiter_two"))  # type: ignore

        new_ev: HumanResponseEvent = await ctx.wait_for_event(
            HumanResponseEvent,
            {"waiter_id": "waiter_two"},  # type: ignore
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
