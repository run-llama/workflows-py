"""Tests for client.py configuration and setup"""

import json
from collections.abc import AsyncIterator
from datetime import datetime

import httpx
import pytest
import pytest_asyncio
import respx
from llama_agents.core.client.manage_client import (
    ControlPlaneClient,
)
from llama_agents.core.client.manage_client import (
    ProjectClient as LlamaDeployClient,
)
from llama_agents.core.schema import LogEvent
from llama_agents.core.schema.deployments import DeploymentCreate


@pytest_asyncio.fixture
async def client() -> AsyncIterator[LlamaDeployClient]:
    """Create a client with mocked config"""
    c = LlamaDeployClient(base_url="http://localhost:8000", project_id="test-project")
    try:
        yield c
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_client_initialization() -> None:
    """Test client initialization with config"""
    client = LlamaDeployClient(
        base_url="http://localhost:8000", project_id="test-project"
    )
    assert client.base_url == "http://localhost:8000"
    assert client.project_id == "test-project"
    await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_server_version_includes_new_fields() -> None:
    async with ControlPlaneClient.ctx(base_url="http://localhost:8000") as client:
        respx.get("http://localhost:8000/api/v1beta1/deployments-public/version").mock(
            return_value=httpx.Response(
                200,
                json={
                    "version": "1.2.3",
                    "requires_auth": False,
                    "min_llamactl_version": "0.3.0a13",
                },
            )
        )

        result = await client.server_version()
        assert result.version == "1.2.3"
        assert result.requires_auth is False
        assert result.min_llamactl_version == "0.3.0a13"


@respx.mock
@pytest.mark.asyncio
async def test_server_version_defaults_when_min_missing() -> None:
    async with ControlPlaneClient.ctx(base_url="http://localhost:8000") as client:
        respx.get("http://localhost:8000/api/v1beta1/deployments-public/version").mock(
            return_value=httpx.Response(
                200,
                json={
                    "version": "1.2.3",
                    "requires_auth": True,
                },
            )
        )

        result = await client.server_version()
        assert result.version == "1.2.3"
        assert result.requires_auth is True
        assert result.min_llamactl_version is None


@respx.mock
@pytest.mark.asyncio
async def test_successful_request(client: LlamaDeployClient) -> None:
    """Test successful HTTP request"""
    respx.get("http://localhost:8000/test").mock(
        return_value=httpx.Response(200, json={"result": "success"})
    )

    result = await client.client.get("/test")

    assert result.status_code == 200
    assert result.json() == {"result": "success"}


@respx.mock
@pytest.mark.asyncio
async def test_request_with_json_data(client: LlamaDeployClient) -> None:
    """Test HTTP request with JSON data"""
    respx.post("http://localhost:8000/test").mock(
        return_value=httpx.Response(200, json={"result": "success"})
    )

    data = {"key": "value"}
    result = await client.client.post("/test", json=data)

    assert result.status_code == 200
    assert result.json() == {"result": "success"}


@respx.mock
@pytest.mark.asyncio
async def test_get_deployments(client: LlamaDeployClient) -> None:
    """Test get_deployments method"""
    respx.get(
        "http://localhost:8000/api/v1beta1/deployments",
        params={"project_id": "test-project"},
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "deployments": [
                    {
                        "id": "test-deploy-1",
                        "name": "Test Deploy 1",
                        "project_id": "test-project",
                        "repo_url": "https://github.com/test/repo1.git",
                        "git_ref": "main",
                        "status": "Running",
                        "has_personal_access_token": False,
                        "secret_names": None,
                        "apiserver_url": None,
                        "deployment_file_path": "deploy.yml",
                    }
                ]
            },
        )
    )

    result = await client.list_deployments()

    assert len(result) == 1
    assert result[0].display_name == "Test Deploy 1"
    assert result[0].git_ref == "main"


@respx.mock
@pytest.mark.asyncio
async def test_get_deployment(client: LlamaDeployClient) -> None:
    """Test get_deployment method"""
    respx.get(
        "http://localhost:8000/api/v1beta1/deployments/test-deploy-1",
        params={"project_id": "test-project"},
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "test-deploy-1",
                "name": "Test Deploy 1",
                "project_id": "test-project",
                "repo_url": "https://github.com/test/repo1.git",
                "git_ref": "main",
                "status": "Running",
                "has_personal_access_token": False,
                "secret_names": None,
                "apiserver_url": None,
                "deployment_file_path": "deploy.yml",
            },
        )
    )

    result = await client.get_deployment("test-deploy-1")

    assert result.display_name == "Test Deploy 1"
    assert result.git_ref == "main"
    assert result.repo_url == "https://github.com/test/repo1.git"


@respx.mock
@pytest.mark.asyncio
async def test_create_deployment_basic(client: LlamaDeployClient) -> None:
    """Test create_deployment without git_ref"""
    route = respx.post(
        "http://localhost:8000/api/v1beta1/deployments",
        params={"project_id": "test-project"},
    ).mock(
        return_value=httpx.Response(
            201,
            json={
                "id": "new-deploy-123",
                "name": "New Deploy",
                "project_id": "test-project",
                "repo_url": "https://github.com/test/repo.git",
                "git_ref": "",
                "deployment_file_path": "deploy.yml",
                "status": "Pending",
                "has_personal_access_token": False,
                "secret_names": None,
                "apiserver_url": None,
            },
        )
    )

    result = await client.create_deployment(
        DeploymentCreate(
            display_name="New Deploy",
            repo_url="https://github.com/test/repo.git",
            deployment_file_path="deploy.yml",
        )
    )

    assert result.display_name == "New Deploy"
    assert result.git_ref == ""

    # Verify request was made with correct data
    assert route.called
    request = route.calls.last.request
    request_data = (request.content or b"").decode()
    parsed_data = json.loads(request_data)
    assert parsed_data["display_name"] == "New Deploy"
    assert parsed_data["name"] == "New Deploy"  # backwards compat with old servers
    assert parsed_data["repo_url"] == "https://github.com/test/repo.git"
    assert parsed_data["deployment_file_path"] == "deploy.yml"
    assert "git_ref" not in parsed_data  # Should be excluded when None


@respx.mock
@pytest.mark.asyncio
async def test_create_deployment_with_git_ref(client: LlamaDeployClient) -> None:
    """Test create_deployment with git_ref parameter"""
    route = respx.post(
        "http://localhost:8000/api/v1beta1/deployments",
        params={"project_id": "test-project"},
    ).mock(
        return_value=httpx.Response(
            201,
            json={
                "id": "new-deploy-456",
                "name": "New Deploy with Ref",
                "project_id": "test-project",
                "repo_url": "https://github.com/test/repo.git",
                "git_ref": "feature-branch",
                "deployment_file_path": "deploy.yml",
                "status": "Pending",
                "has_personal_access_token": False,
                "secret_names": None,
                "apiserver_url": None,
            },
        )
    )

    result = await client.create_deployment(
        DeploymentCreate(
            display_name="New Deploy with Ref",
            repo_url="https://github.com/test/repo.git",
            git_ref="feature-branch",
            deployment_file_path="deploy.yml",
        )
    )

    assert result.display_name == "New Deploy with Ref"
    assert result.git_ref == "feature-branch"

    # Verify request was made with correct data including git_ref
    assert route.called
    request = route.calls.last.request
    request_data = (request.content or b"").decode()
    parsed_data = json.loads(request_data)
    assert parsed_data["display_name"] == "New Deploy with Ref"
    assert parsed_data["repo_url"] == "https://github.com/test/repo.git"
    assert parsed_data["git_ref"] == "feature-branch"
    assert parsed_data["deployment_file_path"] == "deploy.yml"


@respx.mock
@pytest.mark.asyncio
async def test_create_deployment_with_all_params(client: LlamaDeployClient) -> None:
    """Test create_deployment with all parameters including git_ref"""
    route = respx.post(
        "http://localhost:8000/api/v1beta1/deployments",
        params={"project_id": "test-project"},
    ).mock(
        return_value=httpx.Response(
            201,
            json={
                "id": "new-deploy-789",
                "name": "Full Deploy",
                "project_id": "test-project",
                "repo_url": "https://github.com/test/repo.git",
                "git_ref": "v1.0.0",
                "deployment_file_path": "custom_deploy.yml",
                "status": "Pending",
                "has_personal_access_token": True,
                "secret_names": ["API_KEY", "DATABASE_URL"],
                "apiserver_url": None,
            },
        )
    )

    result = await client.create_deployment(
        DeploymentCreate(
            display_name="Full Deploy",
            repo_url="https://github.com/test/repo.git",
            git_ref="v1.0.0",
            deployment_file_path="custom_deploy.yml",
            personal_access_token="ghp_token123",
            secrets={"API_KEY": "secret1", "DATABASE_URL": "secret2"},
        )
    )

    assert result.display_name == "Full Deploy"
    assert result.git_ref == "v1.0.0"
    assert result.has_personal_access_token is True
    assert result.secret_names == ["API_KEY", "DATABASE_URL"]

    # Verify request was made with all data
    assert route.called
    request = route.calls.last.request
    request_data = (request.content or b"").decode()
    parsed_data = json.loads(request_data)
    assert parsed_data["display_name"] == "Full Deploy"
    assert parsed_data["repo_url"] == "https://github.com/test/repo.git"
    assert parsed_data["git_ref"] == "v1.0.0"
    assert parsed_data["deployment_file_path"] == "custom_deploy.yml"
    assert parsed_data["personal_access_token"] == "ghp_token123"
    assert parsed_data["secrets"] == {"API_KEY": "secret1", "DATABASE_URL": "secret2"}


@respx.mock
@pytest.mark.asyncio
async def test_delete_deployment(client: LlamaDeployClient) -> None:
    """Test delete_deployment method"""
    route = respx.delete(
        "http://localhost:8000/api/v1beta1/deployments/test-deploy-1",
        params={"project_id": "test-project"},
    ).mock(return_value=httpx.Response(200))

    await client.delete_deployment("test-deploy-1")

    assert route.called


def _sse_bytes(*events: tuple[str, str]) -> bytes:
    # Build SSE payload: sequence of (event, data_json)
    parts = []
    for event, data in events:
        parts.append(f"event: {event}\n".encode())
        parts.append(f"data: {data}\n".encode())
        parts.append(b"\n")
    return b"".join(parts)


@respx.mock
@pytest.mark.asyncio
async def test_stream_deployment_logs_parses_log_events(
    client: LlamaDeployClient,
) -> None:
    """Stream SSE and parse LogEvent items."""
    payload = _sse_bytes(
        (
            "log",
            LogEvent(
                pod="p1",
                container="c1",
                text="hello",
                timestamp=datetime.now(),
            ).model_dump_json(),
        ),
        (
            "log",
            LogEvent(
                pod="p2",
                container="c2",
                text="world",
                timestamp=datetime.now(),
            ).model_dump_json(),
        ),
    )

    route = respx.get(
        "http://localhost:8000/api/v1beta1/deployments/dep-123/logs",
    ).mock(
        return_value=httpx.Response(
            200, content=payload, headers={"Content-Type": "text/event-stream"}
        )
    )

    events = [e async for e in client.stream_deployment_logs("dep-123")]
    assert len(events) == 2
    assert (events[0].pod, events[0].container, events[0].text) == ("p1", "c1", "hello")
    assert (events[1].pod, events[1].container, events[1].text) == ("p2", "c2", "world")

    assert route.called
    req = route.calls.last.request
    # Query params
    assert req.url.params.get("project_id") == "test-project"
    assert req.url.params.get("include_init_containers") in ("False", "false")
    # Headers
    assert req.headers.get("accept") == "text/event-stream"


@respx.mock
@pytest.mark.asyncio
async def test_stream_deployment_logs_passes_optional_params(
    client: LlamaDeployClient,
) -> None:
    payload = _sse_bytes(
        (
            "log",
            LogEvent(
                pod="p", container="c", text="t", timestamp=datetime.now()
            ).model_dump_json(),
        )
    )

    route = respx.get(
        "http://localhost:8000/api/v1beta1/deployments/dep-1/logs",
    ).mock(
        return_value=httpx.Response(
            200, content=payload, headers={"Content-Type": "text/event-stream"}
        )
    )

    events = [
        e
        async for e in client.stream_deployment_logs(
            "dep-1", include_init_containers=True, since_seconds=120, tail_lines=250
        )
    ]
    assert len(events) == 1
    assert route.called
    req = route.calls.last.request
    assert req.url.params.get("project_id") == "test-project"
    assert req.url.params.get("include_init_containers") in ("True", "true")
    assert req.url.params.get("since_seconds") == "120"
    assert req.url.params.get("tail_lines") == "250"


@respx.mock
@pytest.mark.asyncio
async def test_stream_deployment_logs_ignores_non_log_events(
    client: LlamaDeployClient,
) -> None:
    payload = _sse_bytes(
        ("ping", "{}"),
        (
            "log",
            LogEvent(
                pod="p", container="c", text="ok", timestamp=datetime.now()
            ).model_dump_json(),
        ),
    )

    route = respx.get(
        "http://localhost:8000/api/v1beta1/deployments/dep-x/logs",
    ).mock(
        return_value=httpx.Response(
            200, content=payload, headers={"Content-Type": "text/event-stream"}
        )
    )

    events = [e async for e in client.stream_deployment_logs("dep-x")]
    assert len(events) == 1
    assert (events[0].pod, events[0].container, events[0].text) == ("p", "c", "ok")
    assert route.called
