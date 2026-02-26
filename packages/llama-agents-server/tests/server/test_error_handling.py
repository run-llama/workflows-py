# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import logging
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from llama_agents.server import MemoryWorkflowStore, WorkflowServer
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


class BrokenWorkflow(Workflow):
    @step
    async def explode(self, ev: StartEvent) -> StopEvent:
        raise RuntimeError("something went wrong internally")


@pytest.fixture
def server() -> WorkflowServer:
    server = WorkflowServer(workflow_store=MemoryWorkflowStore())
    server.add_workflow("broken", BrokenWorkflow())
    return server


@pytest_asyncio.fixture
async def client(server: WorkflowServer) -> AsyncGenerator[AsyncClient, None]:
    async with server.contextmanager():
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.mark.asyncio
async def test_malformed_json_returns_400(client: AsyncClient) -> None:
    # Start a workflow to get a handler_id
    response = await client.post(
        "/workflows/broken/run-nowait", json={"start_event": "{}"}
    )
    assert response.status_code == 200
    handler_id = response.json()["handler_id"]

    # Send malformed JSON body
    response = await client.post(
        f"/events/{handler_id}",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400, (
        f"Expected 400, got {response.status_code}: {response.text}"
    )
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_unhandled_exception_returns_json_500(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="llama_agents.server._api"):
        response = await client.post(
            "/workflows/broken/run", json={"start_event": "{}"}
        )
    assert response.status_code == 500, (
        f"Expected 500, got {response.status_code}: {response.text}"
    )
    data = response.json()
    # Workflow failure returns handler data with error field
    assert data["error"] == "something went wrong internally"
    assert any("finished with status=failed" in r.message for r in caplog.records)
