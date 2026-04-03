"""Build artifact garbage collection.

Removes build artifacts from S3 that are no longer referenced by any live
ReplicaSet. Artifacts are retained as long as any RS has a pod template
with a matching LLAMA_DEPLOY_BUILD_ID env var.
"""

from __future__ import annotations

import logging

from llama_agents.control_plane import k8s_client
from llama_agents.control_plane.build_api.build_service import build_artifact_storage

logger = logging.getLogger(__name__)


async def _get_active_build_ids_for_deployment(deployment_id: str) -> set[str]:
    """Return the set of build IDs referenced by live ReplicaSets for a deployment."""
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
    deployment_id: str, keep_build_ids: set[str] | None = None
) -> int:
    """Remove unreferenced build artifacts for a deployment.

    Args:
        deployment_id: The deployment to GC artifacts for.
        keep_build_ids: Additional build IDs to retain regardless of whether
            they appear in a live ReplicaSet. This prevents a race where a
            just-uploaded artifact is GC'd before its ReplicaSet is created.

    Returns the number of artifacts deleted.
    """
    if build_artifact_storage is None:
        return 0

    active_build_ids = await _get_active_build_ids_for_deployment(deployment_id)
    if keep_build_ids:
        active_build_ids |= keep_build_ids
    artifacts = await build_artifact_storage.list_artifacts(deployment_id)

    deleted = 0
    for artifact in artifacts:
        if artifact.build_id not in active_build_ids:
            logger.info(
                "Deleting unreferenced build artifact: deployment=%s build_id=%s",
                deployment_id,
                artifact.build_id,
            )
            await build_artifact_storage.delete_artifact(
                deployment_id, artifact.build_id
            )
            deleted += 1

    if deleted > 0:
        logger.info(
            "GC complete: deployment=%s deleted=%d retained=%d",
            deployment_id,
            deleted,
            len(artifacts) - deleted,
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
