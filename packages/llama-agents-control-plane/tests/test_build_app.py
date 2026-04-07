from unittest.mock import MagicMock, patch

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
