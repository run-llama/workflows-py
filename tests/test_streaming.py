# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import asyncio
from typing import AsyncGenerator

import pytest

from workflows.context import Context
from workflows.testing import WorkflowTestRunner
from workflows.decorators import step
from workflows.errors import WorkflowRuntimeError, WorkflowTimeoutError
from workflows.events import Event, StartEvent, StopEvent
from workflows.workflow import Workflow

from .conftest import OneTestEvent


class StreamingWorkflow(Workflow):
    @step
    async def chat(self, ctx: Context, ev: StartEvent) -> StopEvent:
        async def stream_messages() -> AsyncGenerator[str, None]:
            resp = "Paul Graham is a British-American computer scientist, entrepreneur, vc, and writer."
            for word in resp.split():
                yield word

        async for w in stream_messages():
            ctx.write_event_to_stream(Event(msg=w))

        return StopEvent(result=None)


@pytest.mark.asyncio
async def test_e2e() -> None:
    test_runner = WorkflowTestRunner(StreamingWorkflow())
    r = await test_runner.run(expose_internal=False, exclude_events=[StopEvent])

    assert all("msg" in ev for ev in r.collected)


@pytest.mark.asyncio
async def test_task_raised() -> None:
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            ctx.write_event_to_stream(OneTestEvent(test_param="foo"))
            raise ValueError("The step raised an error!")

    wf = DummyWorkflow()
    r = wf.run()

    # Make sure we don't block indefinitely here because the step raised
    async for ev in r.stream_events():
        if not isinstance(ev, StopEvent):
            assert ev.test_param == "foo"

    # Make sure the await actually caught the exception
    with pytest.raises(ValueError, match="The step raised an error!"):
        await r


@pytest.mark.asyncio
async def test_task_timeout() -> None:
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            ctx.write_event_to_stream(OneTestEvent(test_param="foo"))
            await asyncio.sleep(2)
            return StopEvent()

    wf = DummyWorkflow(timeout=1)
    r = wf.run()

    # Make sure we don't block indefinitely here because the step raised
    async for ev in r.stream_events():
        if not isinstance(ev, StopEvent):
            assert ev.test_param == "foo"

    # Make sure the await actually caught the exception
    with pytest.raises(WorkflowTimeoutError, match="Operation timed out"):
        await r


@pytest.mark.asyncio
async def test_multiple_sequential_streams() -> None:
    test_runner = WorkflowTestRunner(StreamingWorkflow())
    # stream 1
    await test_runner.run(StartEvent())
    # stream 2 -- should not raise an error
    await test_runner.run(StartEvent())


@pytest.mark.asyncio
async def test_consume_only_once() -> None:
    wf = StreamingWorkflow()
    handler = wf.run()

    async for _ in handler.stream_events():
        pass

    with pytest.raises(
        WorkflowRuntimeError,
        match="All the streamed events have already been consumed.",
    ):
        async for _ in handler.stream_events():
            pass

    await handler


@pytest.mark.asyncio
async def test_multiple_ongoing_streams() -> None:
    wf = StreamingWorkflow()
    stream_1 = wf.run()
    stream_2 = wf.run()

    async for ev in stream_1.stream_events():
        if not isinstance(ev, StopEvent):
            assert "msg" in ev

    async for ev in stream_2.stream_events():
        if not isinstance(ev, StopEvent):
            assert "msg" in ev

    await asyncio.gather(stream_1, stream_2)


@pytest.mark.asyncio
async def test_resume_streams() -> None:
    class CounterWorkflow(Workflow):
        @step
        async def count(self, ctx: Context, ev: StartEvent) -> StopEvent:
            ctx.write_event_to_stream(Event(msg="hello!"))

            cur_count = await ctx.store.get("cur_count", default=0)
            await ctx.store.set("cur_count", cur_count + 1)
            return StopEvent(result="done")

    wf = CounterWorkflow()
    await WorkflowTestRunner(wf).run()
    ctx1 = wf._contexts.copy().pop()
    assert ctx1

    await WorkflowTestRunner(wf).run(ctx=ctx1)

    ctx2 = wf._contexts.pop()

    assert ctx2
    assert await ctx2.store.get("cur_count") == 2
