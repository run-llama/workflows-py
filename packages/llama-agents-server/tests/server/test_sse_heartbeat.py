# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from llama_agents.server import MemoryWorkflowStore, WorkflowServer
from server_test_fixtures import (  # type: ignore[import]
    InteractiveWorkflow,
    StreamingWorkflow,
    live_server,
)


def _server_factory(*, sse_heartbeat_interval: float | None) -> WorkflowServer:
    server = WorkflowServer(
        workflow_store=MemoryWorkflowStore(),
        idle_timeout=0.01,
        sse_heartbeat_interval=sse_heartbeat_interval,
    )
    server.add_workflow("interactive", InteractiveWorkflow())
    server.add_workflow("streaming", StreamingWorkflow())
    return server


@pytest.mark.asyncio
async def test_sse_heartbeat_during_idle() -> None:
    """Heartbeat comments appear when no real events are flowing."""
    async with live_server(lambda: _server_factory(sse_heartbeat_interval=0.05)) as (
        base_url,
        _server,
    ):
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            response = await client.post("/workflows/interactive/run-nowait", json={})
            assert response.status_code == 200
            handler_id = response.json()["handler_id"]

            collected: list[str] = []

            async def read_stream() -> None:
                async with client.stream(
                    "GET",
                    f"/events/{handler_id}?sse=true&after_sequence=-1",
                ) as resp:
                    async for line in resp.aiter_lines():
                        collected.append(line)
                        hb_count = sum(1 for x in collected if x == ": heartbeat")
                        if hb_count >= 2:
                            return

            await asyncio.wait_for(read_stream(), timeout=5.0)

            heartbeat_lines = [x for x in collected if x == ": heartbeat"]
            assert len(heartbeat_lines) >= 2


@pytest.mark.asyncio
async def test_sse_heartbeat_interspersed_with_events() -> None:
    """Real events and heartbeat comments both appear on the same stream."""
    async with live_server(lambda: _server_factory(sse_heartbeat_interval=0.05)) as (
        base_url,
        _server,
    ):
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            response = await client.post("/workflows/interactive/run-nowait", json={})
            assert response.status_code == 200
            handler_id = response.json()["handler_id"]

            collected: list[str] = []

            async def read_stream() -> None:
                async with client.stream(
                    "GET",
                    f"/events/{handler_id}?sse=true&after_sequence=-1",
                ) as resp:
                    async for line in resp.aiter_lines():
                        collected.append(line)
                        has_hb = any(x == ": heartbeat" for x in collected)
                        has_data = any(x.startswith("data: ") for x in collected)
                        if has_hb and has_data:
                            return

            await asyncio.wait_for(read_stream(), timeout=5.0)

            heartbeat_lines = [x for x in collected if x == ": heartbeat"]
            data_lines = [x for x in collected if x.startswith("data: ")]
            assert len(heartbeat_lines) >= 1
            assert len(data_lines) >= 1


@pytest.mark.asyncio
async def test_ndjson_no_heartbeat() -> None:
    """NDJSON mode does not receive heartbeat comments even when configured."""
    async with live_server(lambda: _server_factory(sse_heartbeat_interval=0.05)) as (
        base_url,
        _server,
    ):
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            response = await client.post("/workflows/interactive/run-nowait", json={})
            assert response.status_code == 200
            handler_id = response.json()["handler_id"]

            collected: list[str] = []

            async def read_stream() -> None:
                async with client.stream(
                    "GET",
                    f"/events/{handler_id}?sse=false&after_sequence=-1",
                ) as resp:
                    async for line in resp.aiter_lines():
                        collected.append(line)

            # Let it run briefly — should NOT get any heartbeat
            try:
                await asyncio.wait_for(read_stream(), timeout=0.3)
            except (TimeoutError, asyncio.TimeoutError):
                pass

            heartbeat_lines = [x for x in collected if ": heartbeat" in x]
            assert len(heartbeat_lines) == 0


@pytest.mark.asyncio
async def test_no_heartbeat_when_disabled() -> None:
    """Default None heartbeat produces no heartbeat comments."""
    async with live_server(lambda: _server_factory(sse_heartbeat_interval=None)) as (
        base_url,
        _server,
    ):
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            response = await client.post("/workflows/interactive/run-nowait", json={})
            assert response.status_code == 200
            handler_id = response.json()["handler_id"]

            collected: list[str] = []

            async def read_stream() -> None:
                async with client.stream(
                    "GET",
                    f"/events/{handler_id}?sse=true&after_sequence=-1",
                ) as resp:
                    async for line in resp.aiter_lines():
                        collected.append(line)

            # Let it run briefly — should NOT get any heartbeat
            try:
                await asyncio.wait_for(read_stream(), timeout=0.3)
            except (TimeoutError, asyncio.TimeoutError):
                pass

            heartbeat_lines = [x for x in collected if x == ": heartbeat"]
            assert len(heartbeat_lines) == 0


@pytest.mark.asyncio
async def test_heartbeat_with_completed_workflow() -> None:
    """Heartbeat doesn't prevent stream from ending when workflow completes."""
    async with live_server(lambda: _server_factory(sse_heartbeat_interval=0.05)) as (
        base_url,
        _server,
    ):
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            response = await client.post(
                "/workflows/streaming/run-nowait",
                json={"kwargs": {"count": 2}},
            )
            assert response.status_code == 200
            handler_id = response.json()["handler_id"]

            # Wait for workflow to finish, then consume the full stream
            await asyncio.sleep(0.3)

            response = await client.get(
                f"/events/{handler_id}?sse=true&after_sequence=-1"
            )
            assert response.status_code == 200

            data_events = []
            for line in response.text.splitlines():
                if line.startswith("data: "):
                    data_events.append(json.loads(line.removeprefix("data: ")))

            # Should have streaming events + StopEvent
            assert len(data_events) >= 2
