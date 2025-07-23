# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from workflows import Context
from workflows.server import WorkflowServer
from workflows.workflow import Workflow


@pytest.fixture
def server() -> WorkflowServer:
    return WorkflowServer()


@pytest_asyncio.fixture
async def async_client(
    server: WorkflowServer, simple_test_workflow: Workflow, error_workflow: Workflow
) -> AsyncGenerator:
    server.add_workflow("test", simple_test_workflow)
    server.add_workflow("error", error_workflow)
    transport = ASGITransport(app=server.app)
    yield AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_list_workflows(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.get("/workflows")
        assert response.status_code == 200
        data = response.json()
        assert "workflows" in data
        assert set(data["workflows"]) == {"test", "error"}


@pytest.mark.asyncio
async def test_run_workflow_success(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post(
            "/workflows/test/run", json={"kwargs": {"message": "hello"}}
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"] == "processed: hello"


@pytest.mark.asyncio
async def test_run_workflow_no_kwargs(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post("/workflows/test/run", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "processed: default"


@pytest.mark.asyncio
async def test_run_workflow_with_context(
    async_client: AsyncClient, server: WorkflowServer
) -> None:
    async with async_client as client:
        ctx: Context = Context(server._workflows["test"])
        await ctx.store.set("test_param", "message from context")
        ctx_dict = ctx.to_dict()
        response = await client.post("/workflows/test/run", json={"context": ctx_dict})
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "processed: message from context"


@pytest.mark.asyncio
async def test_run_workflow_not_found(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post("/workflows/nonexistent/run", json={})
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_run_workflow_error(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post("/workflows/error/run", json={})
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Test error" in data["error"]


@pytest.mark.asyncio
async def test_run_workflow_invalid_json(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post(
            "/workflows/test/run",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_run_workflow_nowait_success(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post(
            "/workflows/test/run-nowait", json={"kwargs": {"message": "async"}}
        )
        assert response.status_code == 200
        data = response.json()
        assert "handler_id" in data
        assert "status" in data
        assert data["status"] == "started"
        assert len(data["handler_id"]) == 10  # Default nanoid length


@pytest.mark.asyncio
async def test_run_workflow_nowait_not_found(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post("/workflows/nonexistent/run-nowait", json={})
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_run_workflow_nowait_error(async_client: AsyncClient) -> None:
    async with async_client as client:
        # run no-wait
        response = await client.post("/workflows/error/run-nowait", json="wrong_format")
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_get_workflow_result(
    async_client: AsyncClient, server: WorkflowServer
) -> None:
    # Setup a context to test all the code paths
    ctx: Context = Context(server._workflows["test"])
    await ctx.store.set("test_param", "message from context")
    ctx_dict = ctx.to_dict()
    async with async_client as client:
        # run no-wait
        response = await client.post(
            "/workflows/test/run-nowait", json={"context": ctx_dict}
        )
        assert response.status_code == 200
        data = response.json()
        assert "handler_id" in data
        handler_id = data["handler_id"]

        # get result
        response = await client.get(f"/results/{handler_id}")
        assert response.status_code == 200

        # Verify the result content
        result_data = response.json()
        assert "result" in result_data
        assert result_data["result"] == "processed: message from context"

        assert handler_id not in server._handlers


@pytest.mark.asyncio
async def test_get_workflow_result_error(
    async_client: AsyncClient, server: WorkflowServer
) -> None:
    async with async_client as client:
        # run no-wait
        response = await client.post("/workflows/error/run-nowait", json={})
        assert response.status_code == 200
        data = response.json()
        assert "handler_id" in data
        handler_id = data["handler_id"]

        # get result
        response = await client.get(f"/results/{handler_id}")
        assert response.status_code == 500

        # Verify the result content
        result_data = response.json()
        assert "error" in result_data
        assert result_data["error"] == "Error in step 'error_step': Test error"

        assert handler_id not in server._handlers


@pytest.mark.asyncio
async def test_get_workflow_result_not_found(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.get("/results/nonexistent")
        assert response.status_code == 404
