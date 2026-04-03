"""Backup/restore API router (v1beta1)."""

from fastapi import APIRouter, Depends, HTTPException
from llama_agents.core import schema

from .backup_service import BackupService, backup_service

base_router = APIRouter(prefix="/api/v1beta1")
_router = APIRouter(prefix="/backups")


def _require_service() -> BackupService:
    if backup_service is None:
        raise HTTPException(
            status_code=503,
            detail="Backup service not configured. Set BACKUP_S3_BUCKET to enable.",
        )
    return backup_service


@_router.post("", response_model=schema.BackupResponse)
async def create_backup(
    service: BackupService = Depends(_require_service),
) -> schema.BackupResponse:
    return await service.create_backup()


@_router.get("", response_model=schema.BackupListResponse)
async def list_backups(
    service: BackupService = Depends(_require_service),
) -> schema.BackupListResponse:
    return await service.list_backups()


@_router.get("/{backup_id}", response_model=schema.BackupResponse)
async def get_backup(
    backup_id: str,
    service: BackupService = Depends(_require_service),
) -> schema.BackupResponse:
    return await service.get_backup(backup_id)


@_router.delete("/{backup_id}", response_model=schema.BackupResponse)
async def delete_backup(
    backup_id: str,
    service: BackupService = Depends(_require_service),
) -> schema.BackupResponse:
    return await service.delete_backup(backup_id)


@_router.post("/restore", response_model=schema.RestoreResponse)
async def restore_backup(
    request: schema.RestoreRequest,
    service: BackupService = Depends(_require_service),
) -> schema.RestoreResponse:
    return await service.restore_backup(request)


base_router.include_router(_router)
router = base_router
