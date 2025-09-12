# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.


import asyncio

import pytest

from workflows import Context, Workflow, step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)


class SimpleTestWorkflow(Workflow):
    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        message = await ctx.store.get("test_param", None)
        if message is None:
            message = getattr(ev, "message", "default")
        return StopEvent(result=f"processed: {message}")


class ErrorWorkflow(Workflow):
    @step
    async def error_step(self, ev: StartEvent) -> StopEvent:
        raise ValueError("Test error")


class StreamEvent(Event):
    message: str
    sequence: int


class StreamingWorkflow(Workflow):
    @step
    async def stream_data(self, ctx: Context, ev: StartEvent) -> StopEvent:
        count = getattr(ev, "count", 3)

        for i in range(count):
            ctx.write_event_to_stream(StreamEvent(message=f"event_{i}", sequence=i))
            await asyncio.sleep(0.01)  # Small delay between events

        return StopEvent(result=f"completed_{count}_events")


class RequestedExternalEvent(InputRequiredEvent):
    message: str


class ExternalEvent(HumanResponseEvent):
    response: str


class InteractiveWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> RequestedExternalEvent:
        # Wait for an external event
        return RequestedExternalEvent(message="ping")

    @step
    async def end(self, ctx: Context, ev: ExternalEvent) -> StopEvent:
        return StopEvent(result=f"received: {ev.response}")


@pytest.fixture
def simple_test_workflow() -> Workflow:
    return SimpleTestWorkflow()


@pytest.fixture
def error_workflow() -> Workflow:
    return ErrorWorkflow()


@pytest.fixture
def streaming_workflow() -> Workflow:
    return StreamingWorkflow()


@pytest.fixture
def interactive_workflow() -> Workflow:
    return InteractiveWorkflow()
