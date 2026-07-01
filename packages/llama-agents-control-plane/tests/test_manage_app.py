from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from llama_agents.control_plane.manage_api.manage_app import app


@pytest.mark.asyncio
async def test_health_does_not_touch_k8s() -> None:
    """`/health` stays a cheap liveness-of-process check, unrelated to k8s."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# /readyz and /livez share one k8s-checking body, so both are exercised here.
K8S_PROBES = ["/readyz", "/livez"]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", K8S_PROBES)
async def test_k8s_probe_returns_200_when_healthy(endpoint: str) -> None:
    with patch(
        "llama_agents.control_plane.k8s_client.check_k8s_connectivity",
        AsyncMock(return_value=None),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(endpoint)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", K8S_PROBES)
async def test_k8s_probe_returns_503_when_check_fails(endpoint: str) -> None:
    """A wedged kube-apiserver connection must fail the probe, not report 200."""
    with patch(
        "llama_agents.control_plane.k8s_client.check_k8s_connectivity",
        AsyncMock(side_effect=TimeoutError("simulated wedge")),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(endpoint)
    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
