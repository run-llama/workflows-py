"""Tests for S3 code repo storage."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from aiomoto import mock_aws
from dulwich.objects import Commit
from dulwich.refs import Ref
from dulwich.repo import Repo
from llama_agents.control_plane.code_repo.storage import CodeRepoStorage

from .conftest import create_bucket, create_test_repo, make_storage


@pytest.mark.asyncio
async def test_round_trip_bare_repo(tmp_path: Path) -> None:
    with mock_aws():
        create_bucket()
        storage = make_storage()

        repo_path = tmp_path / "repo"
        repo = create_test_repo(repo_path)
        original_ref = repo.refs[Ref(b"refs/heads/main")]

        await storage.upload_repo("deploy-1", repo_path)

        downloaded_path = await storage.download_repo("deploy-1")
        assert downloaded_path is not None
        try:
            downloaded_repo = Repo(str(downloaded_path))
            assert downloaded_repo.refs[Ref(b"refs/heads/main")] == original_ref
        finally:
            shutil.rmtree(downloaded_path.parent, ignore_errors=True)


@pytest.mark.asyncio
async def test_download_returns_none_when_no_repo() -> None:
    with mock_aws():
        create_bucket()
        storage = make_storage()

        result = await storage.download_repo("nonexistent-deploy")
        assert result is None


@pytest.mark.asyncio
async def test_repo_exists(tmp_path: Path) -> None:
    with mock_aws():
        create_bucket()
        storage = make_storage()

        repo_path = tmp_path / "repo"
        create_test_repo(repo_path)

        assert await storage.repo_exists("deploy-2") is False

        await storage.upload_repo("deploy-2", repo_path)
        assert await storage.repo_exists("deploy-2") is True

        await storage.delete_repo("deploy-2")
        assert await storage.repo_exists("deploy-2") is False


@pytest.mark.asyncio
async def test_init_bare_repo() -> None:
    repo_path = CodeRepoStorage.init_bare_repo("deploy-3")
    try:
        assert repo_path.exists()
        assert (repo_path / "HEAD").exists()
        assert (repo_path / "objects").is_dir()
    finally:
        shutil.rmtree(repo_path.parent, ignore_errors=True)


@pytest.mark.asyncio
async def test_delete_repo(tmp_path: Path) -> None:
    with mock_aws():
        create_bucket()
        storage = make_storage()

        repo_path = tmp_path / "repo"
        create_test_repo(repo_path)

        await storage.upload_repo("deploy-4", repo_path)
        assert await storage.repo_exists("deploy-4") is True

        await storage.delete_repo("deploy-4")

        result = await storage.download_repo("deploy-4")
        assert result is None


@pytest.mark.asyncio
async def test_resolve_ref_supports_commit_sha_and_tags(tmp_path: Path) -> None:
    with mock_aws():
        create_bucket()
        storage = make_storage()

        repo_path = tmp_path / "repo"
        repo = create_test_repo(repo_path)
        commit_sha = repo.refs[Ref(b"refs/heads/main")].decode()
        repo.refs[Ref(b"refs/tags/v1.0.0")] = repo.refs[Ref(b"refs/heads/main")]

        await storage.upload_repo("deploy-5", repo_path)

        assert await storage.resolve_ref("deploy-5", "main") == commit_sha
        assert await storage.resolve_ref("deploy-5", "v1.0.0") == commit_sha
        assert await storage.resolve_ref("deploy-5", commit_sha) == commit_sha


@pytest.mark.asyncio
async def test_resolve_ref_rejects_missing_or_non_commit_sha(tmp_path: Path) -> None:
    with mock_aws():
        create_bucket()
        storage = make_storage()

        repo_path = tmp_path / "repo"
        repo = create_test_repo(repo_path)
        commit = repo.get_object(repo.refs[Ref(b"refs/heads/main")])
        assert isinstance(commit, Commit)
        tree_sha = commit.tree.decode()

        await storage.upload_repo("deploy-6", repo_path)

        assert await storage.resolve_ref("deploy-6", "missing-tag") is None
        assert await storage.resolve_ref("deploy-6", tree_sha) is None
