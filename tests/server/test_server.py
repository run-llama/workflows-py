# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.middleware import Middleware
from starlette.testclient import TestClient

from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.server import WorkflowServer


class SimpleTestWorkflow(Workflow):
    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        message = await ctx.store.get("test_param", None)
        if message is None:
            message = getattr(ev, "message", "default")
        return StopEvent(result=f"processed: {message}")


class SlowWorkflow(Workflow):
    @step
    async def slow_process(self, ev: StartEvent) -> StopEvent:
        import asyncio

        await asyncio.sleep(0.1)  # Simulate slow processing
        return StopEvent(result="slow result")


class ErrorWorkflow(Workflow):
    @step
    async def error_step(self, ev: StartEvent) -> StopEvent:
        raise ValueError("Test error")


@pytest.fixture
def server():
    return WorkflowServer()


@pytest.fixture
def client(server):
    """Create a test client for the server."""
    server.add_workflow("test", SimpleTestWorkflow())
    server.add_workflow("slow", SlowWorkflow())
    server.add_workflow("error", ErrorWorkflow())
    return TestClient(server.app)


def test_init():
    server = WorkflowServer()
    assert len(server._middleware) == 1
    assert server._workflows == {}
    assert server._contexts == {}
    assert server._handlers == {}


def test_init_custom_middleware():
    custom_middleware = [Mock(spec=Middleware)]
    server = WorkflowServer(middleware=custom_middleware)  # type: ignore
    assert server._middleware == custom_middleware


def test_add_workflow():
    server = WorkflowServer()
    workflow = SimpleTestWorkflow()
    server.add_workflow("test", workflow)
    assert "test" in server._workflows
    assert server._workflows["test"] == workflow


@pytest.mark.asyncio
@patch("workflows.server.server.uvicorn.Server")
@patch("workflows.server.server.uvicorn.Config")
async def test_serve(mock_config, mock_server, server):
    """Test the serve method."""
    mock_server_instance = AsyncMock()
    mock_server.return_value = mock_server_instance

    await server.serve(host="localhost", port=8000)

    mock_config.assert_called_once_with(server.app, host="localhost", port=8000)
    mock_server_instance.serve.assert_called_once()


@pytest.mark.asyncio
@patch("workflows.server.server.uvicorn.Server")
@patch("workflows.server.server.uvicorn.Config")
async def test_serve_with_uvicorn_config(mock_config, mock_server, server):
    """Test the serve method with custom uvicorn config."""
    mock_server_instance = AsyncMock()
    mock_server.return_value = mock_server_instance

    uvicorn_config = {"log_level": "debug", "reload": True}
    await server.serve(host="localhost", port=8000, uvicorn_config=uvicorn_config)

    mock_config.assert_called_once_with(
        server.app, host="localhost", port=8000, log_level="debug", reload=True
    )


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_list_workflows_empty():
    server = WorkflowServer()
    client = TestClient(server.app)
    response = client.get("/workflows")
    assert response.status_code == 200
    assert response.json() == {"workflows": []}


def test_list_workflows(client):
    response = client.get("/workflows")
    assert response.status_code == 200
    data = response.json()
    assert "workflows" in data
    assert set(data["workflows"]) == {"test", "slow", "error"}


def test_run_workflow_success(client):
    response = client.post("/workflows/test/run", json={"kwargs": {"message": "hello"}})
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == "processed: hello"


def test_run_workflow_no_kwargs(client):
    response = client.post("/workflows/test/run", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "processed: default"


@pytest.mark.asyncio
async def test_run_workflow_with_context(server, client):
    ctx = Context(server._workflows["test"])
    await ctx.store.set("test_param", "message from context")
    ctx_dict = ctx.to_dict()
    response = client.post("/workflows/test/run", json={"context": ctx_dict})
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "processed: message from context"


def test_run_workflow_not_found(client):
    response = client.post("/workflows/nonexistent/run", json={})
    assert response.status_code == 404


def test_run_workflow_error(client):
    response = client.post("/workflows/error/run", json={})
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "Test error" in data["error"]


def test_run_workflow_invalid_json(client):
    response = client.post(
        "/workflows/test/run",
        content="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 500  # Starlette returns 500 for JSON decode errors


def test_run_workflow_nowait_success(client):
    response = client.post(
        "/workflows/test/run-nowait", json={"kwargs": {"message": "async"}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "handler_id" in data
    assert "status" in data
    assert data["status"] == "started"
    assert len(data["handler_id"]) == 10  # Default nanoid length


def test_run_workflow_nowait_not_found(client):
    response = client.post("/workflows/nonexistent/run-nowait", json={})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_run_workflow_nowait_error(server):
    # Add workflows to server
    server.add_workflow("error", ErrorWorkflow())

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # run no-wait
        response = await client.post("/workflows/error/run-nowait", json="wrong_format")
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_get_workflow_result(server):
    # Add workflows to server
    server.add_workflow("test", SimpleTestWorkflow())
    # Setup a context to test all the code paths
    ctx = Context(server._workflows["test"])
    await ctx.store.set("test_param", "message from context")
    ctx_dict = ctx.to_dict()
    # Use httpx as test client
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
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
async def test_get_workflow_result_error(server):
    # Add workflows to server
    server.add_workflow("test", ErrorWorkflow())
    # Use httpx as test client
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # run no-wait
        response = await client.post("/workflows/test/run-nowait", json={})
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


def test_get_workflow_result_not_found(client):
    response = client.get("/results/nonexistent")
    assert response.status_code == 404


def test_extract_workflow_success():
    server = WorkflowServer()
    server.add_workflow("test", SimpleTestWorkflow())

    # Mock request with path params
    mock_request = Mock()
    mock_request.path_params = {"name": "test"}

    workflow = server._extract_workflow(mock_request)
    assert isinstance(workflow, SimpleTestWorkflow)


def test_extract_workflow_missing_name():
    server = WorkflowServer()
    mock_request = Mock()
    mock_request.path_params = {}

    with pytest.raises(Exception) as exc_info:
        server._extract_workflow(mock_request)
    assert exc_info.value.status_code == 400
    assert "name" in exc_info.value.detail


def test_extract_workflow_not_found():
    server = WorkflowServer()
    mock_request = Mock()
    mock_request.path_params = {"name": "nonexistent"}

    with pytest.raises(Exception) as exc_info:
        server._extract_workflow(mock_request)
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail
