# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.


import asyncio
import socket
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable, TypeVar

import httpx
import pytest
import uvicorn
from llama_agents.server import WorkflowServer
from workflows import Context, Workflow, step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)

T = TypeVar("T")


async def async_yield(iterations: int = 10) -> None:
    """Yield to the event loop multiple times to let async tasks run."""
    for _ in range(iterations):
        await asyncio.sleep(0)


async def wait_for_passing(
    func: Callable[[], Awaitable[T]],
    max_duration: float = 5.0,
    interval: float = 0.05,
) -> T:
    start_time = time.monotonic()
    last_exception = None
    while time.monotonic() - start_time < max_duration:
        remaining_duration = max_duration - (time.monotonic() - start_time)
        try:
            return await asyncio.wait_for(func(), timeout=remaining_duration)
        except Exception as e:
            last_exception = e
            await asyncio.sleep(interval)
    if last_exception:
        raise last_exception
    else:
        func_name = getattr(func, "__name__", repr(func))
        raise TimeoutError(
            f"Function {func_name} timed out after {max_duration} seconds"
        )


@asynccontextmanager
async def live_server(
    server_factory: Callable[[], WorkflowServer],
) -> AsyncGenerator[tuple[str, WorkflowServer], None]:
    """Start a live HTTP server for testing with atomic port acquisition."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        port = sock.getsockname()[1]

        server = server_factory()

        config = uvicorn.Config(
            server.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            loop="asyncio",
        )
        uv_server = uvicorn.Server(config)

        task = asyncio.create_task(uv_server.serve(sockets=[sock]))

        base_url = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(base_url=base_url, timeout=1.0) as client:
            for _ in range(50):
                try:
                    resp = await client.get("/health")
                    if resp.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.01)
            else:
                uv_server.should_exit = True
                await task
                raise RuntimeError("Live server did not start in time")

        try:
            yield base_url, server
        finally:
            uv_server.should_exit = True
            try:
                await task
            finally:
                await server.stop()
    finally:
        try:
            sock.close()
        except Exception:
            pass


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
