# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import contextlib
import socket
from contextlib import closing
from typing import AsyncGenerator

import httpx
import pytest
import uvicorn
from workflows import Workflow
from workflows.client.client import WorkflowClient
from workflows.events import StopEvent
from workflows.server import WorkflowServer

from .conftest import (  # type: ignore[import]
    ExternalEvent,
    RequestedExternalEvent,
)
from .util import wait_for_passing  # type: ignore[import]


def _get_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


@pytest.fixture
async def live_server(
    simple_test_workflow: Workflow,
    streaming_workflow: Workflow,
    interactive_workflow: Workflow,
    error_workflow: Workflow,
) -> AsyncGenerator[tuple[str, WorkflowServer], None]:
    port = _get_free_port()
    server = WorkflowServer()
    server.add_workflow("test", simple_test_workflow)
    server.add_workflow("streaming", streaming_workflow)
    server.add_workflow("interactive", interactive_workflow)
    server.add_workflow("error", error_workflow)

    config = uvicorn.Config(
        server.app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        loop="asyncio",
    )
    uv_server = uvicorn.Server(config)

    # Start server in background task (lifespan will start workflows)
    task = asyncio.create_task(uv_server.serve())

    # Wait until server responds on /health or timeout
    base_url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(base_url=base_url, timeout=1.0) as client:
        for _ in range(50):  # ~0.5s max wait
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
            # Ensure graceful shutdown of workflow server
            await server.stop()


@pytest.mark.asyncio
async def test_streaming_over_real_http(
    live_server: tuple[str, WorkflowServer],
) -> None:
    base_url, _server = live_server

    client = WorkflowClient(base_url=base_url)

    # 1) Start interactive workflow (no-wait)
    started = await client.run_workflow_nowait("interactive")
    handler_id = started.handler_id

    # Stream until we see the RequestedExternalEvent, then respond and stop streaming
    saw_prompt = False
    async for ev in client.get_workflow_events(handler_id):
        event = ev.load_event()
        if isinstance(event, RequestedExternalEvent):
            saw_prompt = True
            sent = await client.send_event(handler_id, ExternalEvent(response="pong"))
            assert sent.status == "sent"
            break
    assert saw_prompt

    data = await client.get_handler(handler_id)
    assert data.status == "completed"
    assert data.result is not None
    assert data.result.value.get("result") == "received: pong"


@pytest.mark.asyncio
async def test_reconnect_stream_and_send_event_succeeds(
    live_server: tuple[str, WorkflowServer],
) -> None:
    base_url, _server = live_server

    client = WorkflowClient(base_url=base_url)

    # Start the interactive workflow (no-wait)
    started = await client.run_workflow_nowait("interactive")
    handler_id = started.handler_id

    # 1) Connect and read until the first RequestedExternalEvent, then disconnect
    saw_prompt = False
    async for ev in client.get_workflow_events(handler_id):
        event = ev.load_event()
        if isinstance(event, RequestedExternalEvent):
            saw_prompt = True
            break  # simulate client disconnect
    assert saw_prompt

    # 2) Reconnect to stream; this should succeed and allow streaming again
    stop_seen = asyncio.Event()

    async def _consume_again() -> None:
        async for ev in client.get_workflow_events(handler_id):
            event = ev.load_event()
            if isinstance(event, StopEvent):
                stop_seen.set()
                break

    consumer_task = asyncio.create_task(_consume_again())
    try:
        # 3) After reconnect, post the human response; this should succeed
        sent = await client.send_event(handler_id, ExternalEvent(response="pong"))
        assert sent.status == "sent"

        # 4) Wait for completion
        async def validate_result_response() -> None:
            data = await client.get_handler(handler_id)
            assert data.status == "completed"

        await wait_for_passing(validate_result_response)

        await asyncio.wait_for(stop_seen.wait(), timeout=2.0)
    finally:
        if not consumer_task.done():
            consumer_task.cancel()
            with contextlib.suppress(Exception):
                await consumer_task
