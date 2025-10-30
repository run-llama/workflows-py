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
        if ev.response == "error":
            raise RuntimeError("Error response received")
        return StopEvent(result=f"received: {ev.response}")


class CumulativeWorkflow(Workflow):
    @step
    async def accumulate(self, ctx: Context, ev: StartEvent) -> StopEvent:
        # Get the current count from context store, defaulting to 0
        current_count = await ctx.store.get("count", 0)

        # Get the increment value from the start event, defaulting to 1
        increment = getattr(ev, "increment", 1)

        # Add to the count
        new_count = current_count + increment
        await ctx.store.set("count", new_count)

        # Also track run history
        run_history = await ctx.store.get("run_history", [])
        run_history.append(f"run_{len(run_history) + 1}_increment_{increment}")
        await ctx.store.set("run_history", run_history)

        return StopEvent(result=f"count: {new_count}, runs: {len(run_history)}")


class RequiredStartEvent(StartEvent):
    message: str


class StructuredStartWorkflow(Workflow):
    @step
    async def start(self, ev: RequiredStartEvent) -> StopEvent:
        return StopEvent(result=ev.message)


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


@pytest.fixture
def cumulative_workflow() -> Workflow:
    return CumulativeWorkflow()


@pytest.fixture
def structured_start_workflow() -> Workflow:
    return StructuredStartWorkflow()
