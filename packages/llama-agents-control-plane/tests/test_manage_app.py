from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from llama_agents.control_plane.manage_api.manage_app import app


@pytest.mark.asyncio
async def test_health_does_not_touch_k8s() -> None:
    """`/health` stays a cheap process check, unrelated to k8s; backs startup and
    liveness so an apiserver outage can't restart every replica at once."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_returns_200_when_k8s_is_healthy() -> None:
    with patch(
        "llama_agents.control_plane.k8s_client.check_k8s_connectivity",
        AsyncMock(return_value=None),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readyz_returns_503_when_k8s_check_fails() -> None:
    """A wedged kube-apiserver connection must fail /readyz, not report 200."""
    with patch(
        "llama_agents.control_plane.k8s_client.check_k8s_connectivity",
        AsyncMock(side_effect=TimeoutError("simulated wedge")),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
