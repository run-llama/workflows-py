"""Tests for S3 backup storage."""

from __future__ import annotations

import boto3
import pytest
from aiomoto import mock_aws
from llama_agents.control_plane.backup.storage import (
    S3BackupStorage,
    generate_backup_id,
)


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


@pytest.mark.asyncio
async def test_upload_download_round_trip() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        data = b"archive-bytes-here"
        await storage.upload("backup-20250101-000000", data)
        downloaded = await storage.download("backup-20250101-000000")
        assert downloaded == data


@pytest.mark.asyncio
async def test_list_returns_sorted_backups() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        await storage.upload("backup-20250101-000000", b"a")
        await storage.upload("backup-20250102-000000", b"b")
        await storage.upload("backup-20250103-000000", b"c")

        backups = await storage.list_backups()
        ids = [b.backup_id for b in backups]
        assert len(ids) == 3
        assert set(ids) == {
            "backup-20250101-000000",
            "backup-20250102-000000",
            "backup-20250103-000000",
        }
        timestamps = [b.timestamp for b in backups]
        assert timestamps == sorted(timestamps, reverse=True)


@pytest.mark.asyncio
async def test_delete_removes_backup() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        await storage.upload("backup-del", b"data")
        assert await storage.get_info("backup-del") is not None

        await storage.delete("backup-del")
        assert await storage.get_info("backup-del") is None


@pytest.mark.asyncio
async def test_get_info_returns_none_for_missing() -> None:
    with mock_aws():
        _create_bucket()
        storage = _make_storage()
        assert await storage.get_info("nonexistent") is None


def test_generate_backup_id_format() -> None:
    backup_id = generate_backup_id()
    assert backup_id.startswith("backup-")
    parts = backup_id.split("-")
    assert len(parts) == 3
    assert len(parts[1]) == 8  # YYYYMMDD
    assert len(parts[2]) == 6  # HHMMSS
