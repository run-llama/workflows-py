# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Black box tests for idle release behavior over live HTTP."""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager, closing
from datetime import timedelta
from typing import AsyncGenerator

import httpx
import pytest
import uvicorn
from workflows import Context, Workflow, step
from workflows.client.client import WorkflowClient
from workflows.events import Event, StartEvent, StopEvent, WorkflowIdleEvent
from workflows.server import WorkflowServer
from workflows.server.memory_workflow_store import MemoryWorkflowStore

from .util import wait_for_passing  # type: ignore[import]


class WaitableExternalEvent(Event):
    response: str


class WaitingWorkflow(Workflow):
    @step
    async def start_and_wait(self, ctx: Context, ev: StartEvent) -> StopEvent:
        external = await ctx.wait_for_event(WaitableExternalEvent)
        return StopEvent(result=f"received: {external.response}")


def _get_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


@asynccontextmanager
async def live_server(
    idle_release_timeout: timedelta | None,
) -> AsyncGenerator[str, None]:
    port = _get_free_port()
    server = WorkflowServer(
        workflow_store=MemoryWorkflowStore(),
        idle_release_timeout=idle_release_timeout,
    )
    server.add_workflow(
        "waiting",
        WaitingWorkflow(),
        additional_events=[WaitableExternalEvent],
    )

    config = uvicorn.Config(
        server.app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        loop="asyncio",
    )
    uv_server = uvicorn.Server(config)

    task = asyncio.create_task(uv_server.serve())

    base_url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(base_url=base_url, timeout=1.0) as client:
        for _ in range(50):
            try:
                resp = await client.get("/health")
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.001)
        else:
            uv_server.should_exit = True
            await task
            raise RuntimeError("Live server did not start in time")

    try:
        yield base_url
    finally:
        uv_server.should_exit = True
        try:
            await task
        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_fast_idle_timeout_does_not_drop_valid_event() -> None:
    async with live_server(idle_release_timeout=timedelta(milliseconds=1)) as base_url:
        client = WorkflowClient(base_url=base_url)
        started = await client.run_workflow_nowait("waiting")
        handler_id = started.handler_id

        async for env in client.get_workflow_events(
            handler_id, include_internal_events=True
        ):
            event = env.load_event([WorkflowIdleEvent])
            if isinstance(event, WorkflowIdleEvent):
                send = await client.send_event(
                    handler_id, WaitableExternalEvent(response="hello")
                )
                assert send.status == "sent"
                break

        async def handler_completed() -> None:
            data = await client.get_handler(handler_id)
            assert data.status == "completed"
            assert data.result is not None
            assert data.result.value.get("result") == "received: hello"

        await wait_for_passing(handler_completed, max_duration=2.0, interval=0.01)
