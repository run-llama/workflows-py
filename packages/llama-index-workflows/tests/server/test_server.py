# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.middleware import Middleware

from workflows.server import WorkflowServer
from workflows.workflow import Workflow


def test_init() -> None:
    server = WorkflowServer()
    assert len(server._middleware) == 1
    assert server._workflows == {}
    assert server._contexts == {}
    assert server._handlers == {}


def test_init_custom_middleware() -> None:
    custom_middleware = [Mock(spec=Middleware)]
    server = WorkflowServer(middleware=custom_middleware)  # type: ignore
    assert server._middleware == custom_middleware


def test_add_workflow(simple_test_workflow: Workflow) -> None:
    server = WorkflowServer()
    server.add_workflow("test", simple_test_workflow)
    assert "test" in server._workflows
    assert server._workflows["test"] == simple_test_workflow


@pytest.mark.asyncio
@patch("workflows.server.server.uvicorn.Server")
@patch("workflows.server.server.uvicorn.Config")
async def test_serve(mock_config: Any, mock_server: Any) -> None:
    server = WorkflowServer()
    mock_server_instance = AsyncMock()
    mock_server.return_value = mock_server_instance

    await server.serve(host="localhost", port=8000)

    mock_config.assert_called_once_with(server.app, host="localhost", port=8000)
    mock_server_instance.serve.assert_called_once()


@pytest.mark.asyncio
@patch("workflows.server.server.uvicorn.Server")
@patch("workflows.server.server.uvicorn.Config")
async def test_serve_with_uvicorn_config(mock_config: Any, mock_server: Any) -> None:
    server = WorkflowServer()
    mock_server_instance = AsyncMock()
    mock_server.return_value = mock_server_instance

    uvicorn_config = {"log_level": "debug", "reload": True}
    await server.serve(host="localhost", port=8000, uvicorn_config=uvicorn_config)

    mock_config.assert_called_once_with(
        server.app, host="localhost", port=8000, log_level="debug", reload=True
    )


def test_extract_workflow_success(simple_test_workflow: Workflow) -> None:
    server = WorkflowServer()
    server.add_workflow("test", simple_test_workflow)

    # Mocked request with path_params
    mock_request = Mock()
    mock_request.path_params = {"name": "test"}

    assert server._extract_workflow(mock_request).workflow == simple_test_workflow


def test_extract_workflow_missing_name() -> None:
    server = WorkflowServer()
    mock_request = Mock()
    mock_request.path_params = {}

    with pytest.raises(Exception) as exc_info:
        server._extract_workflow(mock_request)
    assert exc_info.value.status_code == 400  # type: ignore
    assert "name" in exc_info.value.detail  # type: ignore


def test_extract_workflow_not_found() -> None:
    server = WorkflowServer()
    mock_request = Mock()
    mock_request.path_params = {"name": "nonexistent"}

    with pytest.raises(Exception) as exc_info:
        server._extract_workflow(mock_request)
    assert exc_info.value.status_code == 404  # type: ignore
    assert "not found" in exc_info.value.detail  # type: ignore
