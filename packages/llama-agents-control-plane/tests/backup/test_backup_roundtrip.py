"""Round-trip tests: ControlPlaneClient -> FastAPI router -> BackupService -> S3.

Each test stands up a real BackupService backed by moto S3, wires it into the
FastAPI router, and drives it through ControlPlaneClient with a test transport.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import pytest
from aiomoto import mock_aws
from fastapi import FastAPI
from llama_agents.control_plane.backup.storage import S3BackupStorage
from llama_agents.control_plane.manage_api.backup_service import BackupService
from llama_agents.control_plane.manage_api.backup_v1beta1 import router
from llama_agents.core.schema.backups import RestoreRequest

from .conftest import make_client

K8S = "llama_agents.control_plane.manage_api.backup_service.k8s_client"
SETTINGS = "llama_agents.control_plane.manage_api.backup_service.settings"
SERVICE = "llama_agents.control_plane.manage_api.backup_v1beta1.backup_service"
GEN_ID = "llama_agents.control_plane.manage_api.backup_service.generate_backup_id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage() -> S3BackupStorage:
    return S3BackupStorage(
        bucket="test-bucket",
        region="us-east-1",
        access_key="testing",
        secret_key="testing",
    )


def _create_bucket() -> None:
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    s3.create_bucket(Bucket="test-bucket")


def _make_mock_k8s(crds: list[dict[str, Any]] | None = None) -> MagicMock:
    m = MagicMock()
    m.get_all_deployment_crds = AsyncMock(return_value=crds or [])
    m.get_secret_data = AsyncMock(return_value=None)
    m.get_deployment_crd_raw = AsyncMock(return_value=None)
    m.apply_deployment_crd = AsyncMock()
    m.apply_secret = AsyncMock()
    m.delete_deployment_crd = AsyncMock()
    m.get_namespace.return_value = "default"
    return m


def _make_mock_settings() -> MagicMock:
    m = MagicMock()
    m.backup_encryption_password = None
    m.s3_bucket = "test-bucket"
    return m


def _make_deployment(name: str, project_id: str = "proj-1") -> dict[str, Any]:
    return {
        "apiVersion": "deploy.llamaindex.ai/v1alpha1",
        "kind": "LlamaDeployment",
        "metadata": {"name": name, "namespace": "default", "generation": 1},
        "spec": {"image": f"registry/{name}:latest", "projectId": project_id},
        "status": {"ready": True},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_get_backup() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        svc = BackupService(storage)
        mock_k8s = _make_mock_k8s(crds=[_make_deployment("app1")])
        mock_settings = _make_mock_settings()

        app = FastAPI()
        app.include_router(router)

        with patch(SERVICE, svc), patch(K8S, mock_k8s), patch(SETTINGS, mock_settings):
            cp = make_client(app)
            created = await cp.create_backup()

            assert created.status == "completed"
            assert created.deployment_count == 1
            assert created.size_bytes is not None and created.size_bytes > 0

            fetched = await cp.get_backup(created.backup_id)

            assert fetched.backup_id == created.backup_id
            assert fetched.status == "completed"
            assert fetched.size_bytes == created.size_bytes

            await cp.aclose()


@pytest.mark.asyncio
async def test_create_and_list_backups() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        svc = BackupService(storage)
        mock_k8s = _make_mock_k8s(crds=[_make_deployment("app1")])
        mock_settings = _make_mock_settings()

        app = FastAPI()
        app.include_router(router)

        # generate_backup_id has second-level granularity, so two calls in the
        # same second produce the same ID.  Use deterministic IDs instead.
        id_iter = iter(["backup-20260101-000001", "backup-20260101-000002"])
        with (
            patch(SERVICE, svc),
            patch(K8S, mock_k8s),
            patch(SETTINGS, mock_settings),
            patch(GEN_ID, side_effect=lambda: next(id_iter)),
        ):
            cp = make_client(app)
            b1 = await cp.create_backup()
            b2 = await cp.create_backup()

            listing = await cp.list_backups()

            ids = {b.backup_id for b in listing.backups}
            assert b1.backup_id in ids
            assert b2.backup_id in ids
            assert len(listing.backups) == 2

            await cp.aclose()


@pytest.mark.asyncio
async def test_create_and_delete_backup() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        svc = BackupService(storage)
        mock_k8s = _make_mock_k8s(crds=[_make_deployment("app1")])
        mock_settings = _make_mock_settings()

        app = FastAPI()
        app.include_router(router)

        with patch(SERVICE, svc), patch(K8S, mock_k8s), patch(SETTINGS, mock_settings):
            cp = make_client(app)
            created = await cp.create_backup()
            assert created.status == "completed"

            deleted = await cp.delete_backup(created.backup_id)
            assert deleted.status == "deleted"

            fetched = await cp.get_backup(created.backup_id)
            assert fetched.status == "failed"
            assert fetched.error is not None

            await cp.aclose()


@pytest.mark.asyncio
async def test_create_and_restore_backup() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        svc = BackupService(storage)
        mock_k8s = _make_mock_k8s(crds=[_make_deployment("app1")])
        mock_settings = _make_mock_settings()

        app = FastAPI()
        app.include_router(router)

        with patch(SERVICE, svc), patch(K8S, mock_k8s), patch(SETTINGS, mock_settings):
            cp = make_client(app)
            created = await cp.create_backup()

            # Nothing exists in the cluster on restore
            mock_k8s.get_deployment_crd_raw = AsyncMock(return_value=None)

            req = RestoreRequest(backup_id=created.backup_id, conflict_mode="skip")
            restored = await cp.restore_backup(created.backup_id, req)

            assert restored.status == "completed"
            assert restored.results is not None
            assert len(restored.results) == 1
            assert restored.results[0].action == "created"
            assert restored.results[0].name == "app1"

            await cp.aclose()


@pytest.mark.asyncio
async def test_restore_overwrite_always() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        svc = BackupService(storage)
        dep = _make_deployment("app1")
        mock_k8s = _make_mock_k8s(crds=[dep])
        mock_settings = _make_mock_settings()

        app = FastAPI()
        app.include_router(router)

        with patch(SERVICE, svc), patch(K8S, mock_k8s), patch(SETTINGS, mock_settings):
            cp = make_client(app)
            created = await cp.create_backup()

            # Simulate that the deployment exists in the cluster at restore time
            mock_k8s.get_deployment_crd_raw = AsyncMock(return_value=dep)

            req = RestoreRequest(
                backup_id=created.backup_id,
                conflict_mode="overwrite-always",
            )
            restored = await cp.restore_backup(created.backup_id, req)

            assert restored.status == "completed"
            assert restored.results is not None
            assert len(restored.results) == 1
            assert restored.results[0].action == "updated"
            mock_k8s.apply_deployment_crd.assert_awaited()

            await cp.aclose()


@pytest.mark.asyncio
async def test_restore_with_deletions() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        svc = BackupService(storage)

        # Backup only contains app1
        mock_k8s = _make_mock_k8s(crds=[_make_deployment("app1")])
        mock_settings = _make_mock_settings()

        app = FastAPI()
        app.include_router(router)

        with patch(SERVICE, svc), patch(K8S, mock_k8s), patch(SETTINGS, mock_settings):
            cp = make_client(app)
            created = await cp.create_backup()

            # On restore, cluster has app1 + app2
            mock_k8s.get_deployment_crd_raw = AsyncMock(return_value=None)
            mock_k8s.get_all_deployment_crds = AsyncMock(
                return_value=[
                    _make_deployment("app1"),
                    _make_deployment("app2"),
                ]
            )

            req = RestoreRequest(
                backup_id=created.backup_id,
                include_deletions=True,
            )
            restored = await cp.restore_backup(created.backup_id, req)

            assert restored.status == "completed"
            assert restored.results is not None
            actions = {r.name: r.action for r in restored.results}
            assert actions["app1"] == "created"
            assert actions["app2"] == "deleted"

            await cp.aclose()


@pytest.mark.asyncio
async def test_backup_not_configured_returns_503() -> None:
    import httpx

    app = FastAPI()
    app.include_router(router)

    with patch(SERVICE, None):
        cp = make_client(app)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await cp.create_backup()
        assert exc_info.value.response.status_code == 503

        await cp.aclose()
