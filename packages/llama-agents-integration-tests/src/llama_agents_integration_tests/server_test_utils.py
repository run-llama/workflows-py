# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Shared test utilities for integration tests that run live HTTP servers."""

from __future__ import annotations

import asyncio
import socket
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable, Literal, TypeVar

import httpx
import uvicorn
from llama_agents.client.client import WorkflowClient
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


async def wait_for_passing(
    func: Callable[[], Awaitable[T]],
    max_duration: float = 5.0,
    interval: float = 0.05,
) -> T:
    """Retry an async callable until it stops raising, or time runs out."""
    start_time = time.monotonic()
    last_exception: Exception | None = None
    while time.monotonic() - start_time < max_duration:
        remaining_duration = max_duration - (time.monotonic() - start_time)
        try:
            return await asyncio.wait_for(func(), timeout=remaining_duration)
        except Exception as e:
            last_exception = e
            await asyncio.sleep(interval)
    if last_exception:
        raise last_exception
    raise TimeoutError(f"Timed out after {max_duration}s")


async def wait_for_requested_external_event_stream(
    client: WorkflowClient,
    handler_id: str,
    *,
    label: str,
    max_duration: float = 5.0,
) -> int | Literal["now"]:
    async def wait_for_prompt() -> int | Literal["now"]:
        stream = client.get_workflow_events(handler_id)
        try:
            async for ev in stream:
                event = ev.load_event([RequestedExternalEvent])
                if isinstance(event, RequestedExternalEvent):
                    return stream.last_sequence
        finally:
            await stream.aclose()

        raise AssertionError(
            f"{label}: event stream ended before RequestedExternalEvent "
            f"for handler {handler_id}"
        )

    try:
        return await asyncio.wait_for(wait_for_prompt(), timeout=max_duration)
    except TimeoutError as exc:
        raise AssertionError(
            f"{label}: timed out waiting for RequestedExternalEvent "
            f"for handler {handler_id}"
        ) from exc


# -- Shared workflow definitions --


class SimpleTestWorkflow(Workflow):
    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        message = await ctx.store.get("test_param", None)
        if message is None:
            message = getattr(ev, "message", "default")
        return StopEvent(result=f"processed: {message}")


class StreamEvent(Event):
    message: str
    sequence: int


class StreamingWorkflow(Workflow):
    @step
    async def stream_data(self, ctx: Context, ev: StartEvent) -> StopEvent:
        count = getattr(ev, "count", 3)
        for i in range(count):
            ctx.write_event_to_stream(StreamEvent(message=f"event_{i}", sequence=i))
            await asyncio.sleep(0.01)
        return StopEvent(result=f"completed_{count}_events")


class RequestedExternalEvent(InputRequiredEvent):
    message: str


class ExternalEvent(HumanResponseEvent):
    response: str


class InteractiveWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> RequestedExternalEvent:
        return RequestedExternalEvent(message="ping")

    @step
    async def end(self, ctx: Context, ev: ExternalEvent) -> StopEvent:
        if ev.response == "error":
            raise RuntimeError("Error response received")
        return StopEvent(result=f"received: {ev.response}")


# -- Live server utility --


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
        await server.start()

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
