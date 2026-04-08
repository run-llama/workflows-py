"""Build artifact garbage collection.

The grace window preserves the invariant: a build Job surviving within its
TTLSecondsAfterFinished window with Status.Succeeded > 0 must still have its
artifact in S3. The operator short-circuits new builds on that assumption, so
the grace window must exceed the Job TTL.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from llama_agents.control_plane import k8s_client
from llama_agents.control_plane.build_api.build_service import build_artifact_storage
from llama_agents.control_plane.settings import settings

logger = logging.getLogger(__name__)

# Bounds concurrency when deleting aged-out build artifacts so a large
# catch-up cohort can't saturate the storage client pool.
_GC_DELETE_CONCURRENCY = 10


async def _get_referenced_build_ids_from_replicasets(
    deployment_id: str,
) -> set[str]:
    """Return build IDs referenced by ReplicaSets in the deployment's revision history."""
    build_ids: set[str] = set()

    try:
        rs_list = await k8s_client.list_replicasets_for_deployment(deployment_id)
    except Exception:
        logger.warning("Failed to list ReplicaSets for %s, skipping GC", deployment_id)
        return build_ids

    for rs in rs_list:
        if not rs.spec or not rs.spec.template or not rs.spec.template.spec:
            continue
        for container in rs.spec.template.spec.containers or []:
            for env in container.env or []:
                if env.name == "LLAMA_DEPLOY_BUILD_ID" and env.value:
                    build_ids.add(env.value)
        for container in rs.spec.template.spec.init_containers or []:
            for env in container.env or []:
                if env.name == "LLAMA_DEPLOY_BUILD_ID" and env.value:
                    build_ids.add(env.value)

    return build_ids


async def gc_build_artifacts(
    deployment_id: str,
    keep_build_ids: set[str] | None = None,
    *,
    now: datetime | None = None,
) -> int:
    """Delete artifacts that are both unreferenced and older than the grace window.

    Returns the number of artifacts deleted.
    """
    storage = build_artifact_storage
    if storage is None:
        return 0

    referenced_build_ids = await _get_referenced_build_ids_from_replicasets(
        deployment_id
    )
    if keep_build_ids:
        referenced_build_ids |= keep_build_ids
    artifacts = await storage.list_artifacts(deployment_id)

    current_time = now if now is not None else datetime.now(timezone.utc)
    grace_cutoff = current_time - timedelta(
        seconds=settings.build_artifact_gc_grace_seconds
    )

    to_delete: list[str] = []
    retained_by_grace = 0
    for artifact in artifacts:
        if artifact.build_id in referenced_build_ids:
            continue

        # S3 LastModified may come back naive in some backends; normalize to UTC.
        artifact_ts = artifact.timestamp
        if artifact_ts.tzinfo is None:
            artifact_ts = artifact_ts.replace(tzinfo=timezone.utc)

        if artifact_ts > grace_cutoff:
            retained_by_grace += 1
            continue

        to_delete.append(artifact.build_id)

    deleted = 0
    if to_delete:
        sem = asyncio.Semaphore(_GC_DELETE_CONCURRENCY)

        async def _delete(build_id: str) -> None:
            async with sem:
                logger.info(
                    "Deleting unreferenced build artifact: deployment=%s build_id=%s",
                    deployment_id,
                    build_id,
                )
                await storage.delete_artifact(deployment_id, build_id)

        results = await asyncio.gather(
            *(_delete(bid) for bid in to_delete),
            return_exceptions=True,
        )
        for build_id, result in zip(to_delete, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Failed to delete build artifact: deployment=%s build_id=%s error=%s",
                    deployment_id,
                    build_id,
                    result,
                )
            else:
                deleted += 1

    if deleted > 0 or retained_by_grace > 0:
        logger.info(
            "GC complete: deployment=%s deleted=%d retained_by_grace=%d total=%d",
            deployment_id,
            deleted,
            retained_by_grace,
            len(artifacts),
        )
    return deleted


async def gc_all_build_artifacts() -> int:
    """Run GC across all deployments. Returns total artifacts deleted."""
    if build_artifact_storage is None:
        return 0

    total_deleted = 0
    try:
        deployments = await k8s_client.list_all_deployments()
        deployment_names = {d.metadata.name for d in deployments if d.metadata}

        # For each known deployment, GC its artifacts
        for name in deployment_names:
            total_deleted += await gc_build_artifacts(name)

    except Exception:
        logger.exception("Failed to run GC across all deployments")

    return total_deleted


async def delete_all_artifacts_for_deployment(deployment_id: str) -> int:
    """Delete all build artifacts for a deployment (used on deployment deletion)."""
    if build_artifact_storage is None:
        return 0

    return await build_artifact_storage.delete_all_artifacts(deployment_id)
