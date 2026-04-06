"""Schema models for backup/restore API."""

from datetime import datetime
from typing import Literal

from .base import Base


class BackupResponse(Base):
    backup_id: str
    status: Literal["completed", "failed", "deleted"]
    timestamp: datetime | None = None
    deployment_count: int | None = None
    size_bytes: int | None = None
    error: str | None = None


class BackupListResponse(Base):
    backups: list[BackupResponse]


class RestoreRequest(Base):
    backup_id: str
    conflict_mode: Literal["skip", "overwrite-if-newer", "overwrite-always"] = "skip"
    include_deletions: bool = False


class RestoreDeploymentResult(Base):
    name: str
    action: Literal["created", "updated", "skipped", "failed", "deleted"]
    error: str | None = None


class RestoreResponse(Base):
    backup_id: str
    status: Literal["completed", "failed"]
    results: list[RestoreDeploymentResult] | None = None
    error: str | None = None
