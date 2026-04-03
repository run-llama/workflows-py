"""Tests for backup/restore schema models."""

from __future__ import annotations

from datetime import datetime, timezone

from llama_agents.core.schema.backups import (
    BackupListResponse,
    BackupResponse,
    RestoreDeploymentResult,
    RestoreRequest,
    RestoreResponse,
)


def test_backup_response() -> None:
    resp = BackupResponse(
        backup_id="backup-20250101-000000",
        status="completed",
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        deployment_count=3,
        size_bytes=1024,
    )
    assert resp.backup_id == "backup-20250101-000000"
    assert resp.status == "completed"
    assert resp.deployment_count == 3


def test_backup_response_defaults() -> None:
    resp = BackupResponse(backup_id="b1", status="completed")
    assert resp.timestamp is None
    assert resp.deployment_count is None
    assert resp.size_bytes is None
    assert resp.error is None


def test_backup_list_response() -> None:
    resp = BackupListResponse(
        backups=[
            BackupResponse(backup_id="b1", status="completed"),
            BackupResponse(backup_id="b2", status="failed", error="oops"),
        ]
    )
    assert len(resp.backups) == 2


def test_restore_request_defaults() -> None:
    req = RestoreRequest(backup_id="b1")
    assert req.conflict_mode == "skip"
    assert req.include_deletions is False


def test_restore_request_custom() -> None:
    req = RestoreRequest(
        backup_id="b1",
        conflict_mode="overwrite-always",
        include_deletions=True,
    )
    assert req.conflict_mode == "overwrite-always"
    assert req.include_deletions is True


def test_restore_deployment_result() -> None:
    result = RestoreDeploymentResult(name="app1", action="created")
    assert result.error is None


def test_restore_response() -> None:
    resp = RestoreResponse(
        backup_id="b1",
        status="completed",
        results=[
            RestoreDeploymentResult(name="app1", action="created"),
            RestoreDeploymentResult(name="app2", action="skipped"),
        ],
    )
    assert resp.results is not None and len(resp.results) == 2
    assert resp.error is None
