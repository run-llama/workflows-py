"""Shared fixtures for backup tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from llama_agents.control_plane.backup.storage import S3BackupStorage
from llama_agents.control_plane.manage_api.backup_service import BackupService
from llama_agents.core.client.manage_client import ControlPlaneClient


def make_deployment(name: str, project_id: str = "proj-1") -> dict[str, Any]:
    """Create a minimal CRD-like dict for testing."""
    return {
        "apiVersion": "deploy.llamaindex.ai/v1alpha1",
        "kind": "LlamaDeployment",
        "metadata": {"name": name, "namespace": "default"},
        "spec": {"image": f"registry/{name}:latest", "projectId": project_id},
    }


def make_raw_crd(
    name: str,
    project_id: str = "proj-1",
    generation: int = 1,
) -> dict[str, Any]:
    """Create a raw CRD dict as returned by k8s (with cluster metadata)."""
    return {
        "apiVersion": "deploy.llamaindex.ai/v1alpha1",
        "kind": "LlamaDeployment",
        "metadata": {
            "name": name,
            "namespace": "default",
            "resourceVersion": "12345",
            "uid": "abc-def",
            "creationTimestamp": "2025-01-01T00:00:00Z",
            "generation": generation,
        },
        "spec": {"image": f"registry/{name}:latest", "projectId": project_id},
        "status": {"ready": True},
    }


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Return an AsyncMock of S3BackupStorage."""
    storage = AsyncMock(spec=S3BackupStorage)
    storage.list_backups.return_value = []
    return storage


@pytest.fixture
def backup_service(mock_storage: AsyncMock) -> BackupService:
    """Return a BackupService with mock storage."""
    return BackupService(mock_storage)


def make_client(app: FastAPI) -> ControlPlaneClient:
    """Build a ControlPlaneClient that talks to *app* via ASGI transport."""
    client = ControlPlaneClient.__new__(ControlPlaneClient)
    client.base_url = "http://test"
    client.api_key = None
    transport = httpx.ASGITransport(app=app)
    client.client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client.hookless_client = httpx.AsyncClient(
        transport=transport, base_url="http://test"
    )
    return client


@pytest.fixture(autouse=True)
def _clear_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent host AWS config from leaking into moto tests."""
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    monkeypatch.delenv("AWS_CONFIG_FILE", raising=False)
