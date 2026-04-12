# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for build artifact garbage collection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from kubernetes.client import (
    V1Container,
    V1EnvVar,
    V1LabelSelector,
    V1PodSpec,
    V1PodTemplateSpec,
    V1ReplicaSet,
    V1ReplicaSetSpec,
)
from llama_agents.control_plane import k8s_client
from llama_agents.control_plane.build_api import build_gc
from llama_agents.control_plane.build_api.build_storage import ArtifactInfo


@dataclass
class FakeStorage:
    """Minimal fake of BuildArtifactStorage for GC tests.

    Tracks list/delete calls; no real S3.
    """

    artifacts: list[ArtifactInfo]
    deleted: list[tuple[str, str]]

    async def list_artifacts(self, deployment_name: str) -> list[ArtifactInfo]:
        return [a for a in self.artifacts if a.deployment_name == deployment_name]

    async def delete_artifact(self, deployment_name: str, build_id: str) -> None:
        self.deleted.append((deployment_name, build_id))
        self.artifacts = [
            a
            for a in self.artifacts
            if not (a.deployment_name == deployment_name and a.build_id == build_id)
        ]

    async def delete_all_artifacts(self, deployment_name: str) -> int:
        to_delete = [a for a in self.artifacts if a.deployment_name == deployment_name]
        for a in to_delete:
            self.deleted.append((a.deployment_name, a.build_id))
        self.artifacts = [
            a for a in self.artifacts if a.deployment_name != deployment_name
        ]
        return len(to_delete)


def _artifact(
    deployment: str, build_id: str, *, age_seconds: int, now: datetime
) -> ArtifactInfo:
    return ArtifactInfo(
        deployment_name=deployment,
        build_id=build_id,
        timestamp=now - timedelta(seconds=age_seconds),
        size_bytes=1024,
    )


def _replicaset_with_build_id(build_id: str) -> V1ReplicaSet:
    """Build a V1ReplicaSet whose pod template references the given build_id
    via the LLAMA_DEPLOY_BUILD_ID env var — matching what the GC walks."""
    return V1ReplicaSet(
        spec=V1ReplicaSetSpec(
            selector=V1LabelSelector(match_labels={}),
            template=V1PodTemplateSpec(
                spec=V1PodSpec(
                    containers=[
                        V1Container(
                            name="app",
                            env=[
                                V1EnvVar(name="LLAMA_DEPLOY_BUILD_ID", value=build_id)
                            ],
                        )
                    ],
                ),
            ),
        ),
    )


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def fake_storage() -> FakeStorage:
    return FakeStorage(artifacts=[], deleted=[])


@pytest.fixture
def patched_gc(fake_storage: FakeStorage):
    """Patch module-level deps of gc_build_artifacts: storage + k8s client."""
    with (
        patch.object(build_gc, "build_artifact_storage", fake_storage),
        patch.object(
            k8s_client,
            "list_replicasets_for_deployment",
            AsyncMock(return_value=[]),
        ) as mock_list,
    ):
        yield fake_storage, mock_list


@pytest.mark.asyncio
async def test_retains_two_recent_artifacts_for_same_deployment(
    patched_gc: tuple[FakeStorage, AsyncMock], now: datetime
) -> None:
    """Back-to-back uploads for different buildIds must both survive GC even
    when no ReplicaSet references either yet."""
    storage, _ = patched_gc
    storage.artifacts = [
        _artifact("doc-extract", "build-a", age_seconds=0, now=now),
        _artifact("doc-extract", "build-b", age_seconds=1, now=now),
    ]

    deleted = await build_gc.gc_build_artifacts("doc-extract", now=now)

    assert deleted == 0
    assert storage.deleted == []
    assert len(storage.artifacts) == 2


@pytest.mark.asyncio
async def test_retains_artifact_within_grace_window(
    patched_gc: tuple[FakeStorage, AsyncMock], now: datetime
) -> None:
    """Artifact older than a few minutes but within the grace window is kept."""
    storage, _ = patched_gc
    # 30 minutes old — well within default 4500s (75min) window
    storage.artifacts = [
        _artifact("app", "build-30m", age_seconds=30 * 60, now=now),
    ]

    deleted = await build_gc.gc_build_artifacts("app", now=now)

    assert deleted == 0
    assert storage.artifacts == [
        _artifact("app", "build-30m", age_seconds=30 * 60, now=now),
    ]


@pytest.mark.asyncio
async def test_deletes_unreferenced_artifact_past_grace_window(
    patched_gc: tuple[FakeStorage, AsyncMock], now: datetime
) -> None:
    """Artifact older than the grace window AND unreferenced by any RS is deleted."""
    storage, _ = patched_gc
    # 2 hours old — well past the default 75min grace window
    storage.artifacts = [
        _artifact("app", "build-old", age_seconds=2 * 60 * 60, now=now),
    ]

    deleted = await build_gc.gc_build_artifacts("app", now=now)

    assert deleted == 1
    assert storage.deleted == [("app", "build-old")]


@pytest.mark.asyncio
async def test_retains_referenced_artifact_regardless_of_age(
    patched_gc: tuple[FakeStorage, AsyncMock], now: datetime
) -> None:
    """An artifact referenced by a live ReplicaSet must never be deleted, even if
    it's arbitrarily old.
    """
    storage, mock_list = patched_gc
    storage.artifacts = [
        _artifact("app", "build-ancient", age_seconds=10 * 24 * 3600, now=now),
    ]

    mock_list.return_value = [_replicaset_with_build_id("build-ancient")]

    deleted = await build_gc.gc_build_artifacts("app", now=now)

    assert deleted == 0
    assert storage.deleted == []


@pytest.mark.asyncio
async def test_keep_build_ids_forces_retention(
    patched_gc: tuple[FakeStorage, AsyncMock], now: datetime
) -> None:
    """keep_build_ids is belt-and-suspenders: even for an aged-out artifact,
    an explicit keep_build_ids entry prevents deletion."""
    storage, _ = patched_gc
    storage.artifacts = [
        _artifact("app", "build-old", age_seconds=2 * 60 * 60, now=now),
    ]

    deleted = await build_gc.gc_build_artifacts(
        "app", keep_build_ids={"build-old"}, now=now
    )

    assert deleted == 0
    assert storage.deleted == []


@pytest.mark.asyncio
async def test_mixed_cohort_only_aged_out_deleted(
    patched_gc: tuple[FakeStorage, AsyncMock], now: datetime
) -> None:
    """Given a mix of recent and aged-out artifacts, only the aged-out ones
    that are also unreferenced get deleted."""
    storage, _ = patched_gc
    storage.artifacts = [
        _artifact("app", "build-recent", age_seconds=60, now=now),
        _artifact(
            "app", "build-middle", age_seconds=60 * 60, now=now
        ),  # 60m, within grace
        _artifact("app", "build-old", age_seconds=3 * 60 * 60, now=now),  # 3h
        _artifact("app", "build-ancient", age_seconds=10 * 3600, now=now),  # 10h
    ]

    deleted = await build_gc.gc_build_artifacts("app", now=now)

    assert deleted == 2
    deleted_ids = {bid for _, bid in storage.deleted}
    assert deleted_ids == {"build-old", "build-ancient"}


@pytest.mark.asyncio
async def test_handles_naive_datetime_from_storage(
    patched_gc: tuple[FakeStorage, AsyncMock], now: datetime
) -> None:
    """Some S3 backends return naive datetimes; the GC must normalize before
    comparison rather than raising a 'can't compare offset-naive to offset-aware'
    TypeError."""
    storage, _ = patched_gc
    naive_now = now.replace(tzinfo=None)
    storage.artifacts = [
        ArtifactInfo(
            deployment_name="app",
            build_id="build-naive",
            timestamp=naive_now - timedelta(hours=3),  # aged out
            size_bytes=100,
        ),
    ]

    deleted = await build_gc.gc_build_artifacts("app", now=now)

    assert deleted == 1
    assert storage.deleted == [("app", "build-naive")]


@pytest.mark.asyncio
async def test_returns_zero_when_storage_disabled(now: datetime) -> None:
    with patch.object(build_gc, "build_artifact_storage", None):
        deleted = await build_gc.gc_build_artifacts("app", now=now)
    assert deleted == 0


@pytest.mark.asyncio
async def test_delete_all_artifacts_for_deployment(
    patched_gc: tuple[FakeStorage, AsyncMock], now: datetime
) -> None:
    """Deployment deletion path deletes everything regardless of age."""
    storage, _ = patched_gc
    storage.artifacts = [
        _artifact("app", "build-a", age_seconds=1, now=now),
        _artifact("app", "build-b", age_seconds=100, now=now),
    ]

    count = await build_gc.delete_all_artifacts_for_deployment("app")

    assert count == 2
    assert len(storage.deleted) == 2
    assert storage.artifacts == []


@pytest.mark.asyncio
async def test_partial_delete_failure_does_not_abort_remaining(now: datetime) -> None:
    """One failing delete in a concurrent batch must not prevent the others
    from running; the returned count should reflect only successful deletes."""
    artifacts = [
        _artifact("app", "build-good-1", age_seconds=3 * 3600, now=now),
        _artifact("app", "build-bad", age_seconds=3 * 3600, now=now),
        _artifact("app", "build-good-2", age_seconds=3 * 3600, now=now),
    ]
    deleted: list[tuple[str, str]] = []

    async def delete_artifact(deployment_name: str, build_id: str) -> None:
        if build_id == "build-bad":
            raise RuntimeError("simulated S3 failure")
        deleted.append((deployment_name, build_id))

    fake = AsyncMock()
    fake.list_artifacts = AsyncMock(return_value=artifacts)
    fake.delete_artifact = AsyncMock(side_effect=delete_artifact)

    with (
        patch.object(build_gc, "build_artifact_storage", fake),
        patch.object(
            k8s_client,
            "list_replicasets_for_deployment",
            AsyncMock(return_value=[]),
        ),
    ):
        count = await build_gc.gc_build_artifacts("app", now=now)

    assert count == 2
    assert sorted(bid for _, bid in deleted) == ["build-good-1", "build-good-2"]
