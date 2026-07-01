from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from llama_agents.control_plane.build_api.build_app import build_app


@pytest.mark.anyio
async def test_health_returns_503_when_s3_not_configured() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app), base_url="http://test"
    ) as client:
        with patch(
            "llama_agents.control_plane.build_api.build_app.build_artifact_storage",
            None,
        ):
            response = await client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert "S3_BUCKET" in data["reason"]


@pytest.mark.anyio
async def test_health_returns_200_when_s3_configured() -> None:
    mock_storage = MagicMock()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app), base_url="http://test"
    ) as client:
        with patch(
            "llama_agents.control_plane.build_api.build_app.build_artifact_storage",
            mock_storage,
        ):
            response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "build-api"


# /readyz and /livez share one k8s-checking body, so both are exercised here.
K8S_PROBES = ["/readyz", "/livez"]


@pytest.mark.anyio
@pytest.mark.parametrize("endpoint", K8S_PROBES)
async def test_k8s_probe_returns_200_when_healthy(endpoint: str) -> None:
    with patch(
        "llama_agents.control_plane.k8s_client.check_k8s_connectivity",
        AsyncMock(return_value=None),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=build_app), base_url="http://test"
        ) as client:
            response = await client.get(endpoint)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
@pytest.mark.parametrize("endpoint", K8S_PROBES)
async def test_k8s_probe_returns_503_when_check_fails(endpoint: str) -> None:
    """A wedged kube-apiserver connection must fail the probe, not report 200."""
    with patch(
        "llama_agents.control_plane.k8s_client.check_k8s_connectivity",
        AsyncMock(side_effect=TimeoutError("simulated wedge")),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=build_app), base_url="http://test"
        ) as client:
            response = await client.get(endpoint)
    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
