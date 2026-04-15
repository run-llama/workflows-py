"""S3-compatible backup storage backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from botocore.exceptions import ClientError
from llama_agents.control_plane.storage import S3ObjectStorage

logger = logging.getLogger(__name__)


@dataclass
class BackupInfo:
    """Metadata about a backup in S3."""

    backup_id: str
    timestamp: datetime
    size_bytes: int


class S3BackupStorage(S3ObjectStorage):
    """Upload, download, list, and delete backup archives in S3-compatible storage."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        key_prefix: str = "backups",
        unsigned: bool = False,
    ) -> None:
        super().__init__(
            bucket=bucket,
            endpoint_url=endpoint_url,
            region=region,
            access_key=access_key,
            secret_key=secret_key,
            key_prefix=key_prefix,
            unsigned=unsigned,
        )

    def _key(self, backup_id: str) -> str:
        return f"{self._key_prefix}/{backup_id}.tar.gz"

    async def upload(self, backup_id: str, data: bytes) -> None:
        """Upload a backup archive to S3."""
        async with self._client() as client:
            await client.put_object(
                Bucket=self._bucket,
                Key=self._key(backup_id),
                Body=data,
            )

    async def download(self, backup_id: str) -> bytes:
        """Download a backup archive from S3."""
        async with self._client() as client:
            response = await client.get_object(
                Bucket=self._bucket,
                Key=self._key(backup_id),
            )
            return await response["Body"].read()

    async def list_backups(self) -> list[BackupInfo]:
        """List all backups in S3, sorted by timestamp descending."""
        async with self._client() as client:
            paginator = client.get_paginator("list_objects_v2")
            backups: list[BackupInfo] = []
            async for page in paginator.paginate(
                Bucket=self._bucket, Prefix=f"{self._key_prefix}/"
            ):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if key is None or not key.endswith(".tar.gz"):
                        continue
                    last_modified = obj.get("LastModified")
                    size = obj.get("Size")
                    if last_modified is None or size is None:
                        continue
                    backup_id = key.removeprefix(f"{self._key_prefix}/").removesuffix(
                        ".tar.gz"
                    )
                    backups.append(
                        BackupInfo(
                            backup_id=backup_id,
                            timestamp=last_modified,
                            size_bytes=size,
                        )
                    )
            backups.sort(key=lambda b: b.timestamp, reverse=True)
            return backups

    async def delete(self, backup_id: str) -> None:
        """Delete a backup from S3."""
        async with self._client() as client:
            await client.delete_object(
                Bucket=self._bucket,
                Key=self._key(backup_id),
            )

    async def get_info(self, backup_id: str) -> BackupInfo | None:
        """Get metadata for a single backup, or None if not found."""
        async with self._client() as client:
            try:
                resp = await client.head_object(
                    Bucket=self._bucket,
                    Key=self._key(backup_id),
                )
                return BackupInfo(
                    backup_id=backup_id,
                    timestamp=resp["LastModified"],
                    size_bytes=resp["ContentLength"],
                )
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "404":
                    return None
                raise


def generate_backup_id() -> str:
    """Generate a human-readable, sortable backup ID."""
    return f"backup-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
