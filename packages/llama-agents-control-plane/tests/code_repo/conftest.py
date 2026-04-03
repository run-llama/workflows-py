"""Shared test fixtures for code_repo tests."""

from __future__ import annotations

import time
from pathlib import Path

import boto3
from dulwich.objects import Blob, Commit, Tree
from dulwich.refs import Ref
from dulwich.repo import Repo
from llama_agents.control_plane.code_repo.storage import CodeRepoStorage

TEST_BUCKET = "test-bucket"
TEST_REGION = "us-east-1"
TEST_AWS_KEY = "testing"


def make_storage() -> CodeRepoStorage:
    return CodeRepoStorage(
        bucket=TEST_BUCKET,
        region=TEST_REGION,
        access_key=TEST_AWS_KEY,
        secret_key=TEST_AWS_KEY,
    )


def create_bucket() -> None:
    s3 = boto3.client(
        "s3",
        region_name=TEST_REGION,
        aws_access_key_id=TEST_AWS_KEY,
        aws_secret_access_key=TEST_AWS_KEY,
    )
    s3.create_bucket(Bucket=TEST_BUCKET)


def create_test_repo(path: Path) -> Repo:
    """Create a bare repo with a single commit."""
    path.mkdir(parents=True, exist_ok=True)
    repo = Repo.init_bare(str(path))
    blob = Blob.from_string(b"hello world")
    repo.object_store.add_object(blob)
    tree = Tree()
    tree.add(b"test.txt", 0o100644, blob.id)
    repo.object_store.add_object(tree)
    commit = Commit()
    commit.tree = tree.id
    commit.author = commit.committer = b"Test User <test@example.com>"
    commit.commit_time = commit.author_time = int(time.time())
    commit.commit_timezone = commit.author_timezone = 0
    commit.encoding = b"UTF-8"
    commit.message = b"Initial commit"
    repo.object_store.add_object(commit)
    repo.refs[Ref(b"refs/heads/main")] = commit.id
    repo.refs.set_symbolic_ref(Ref(b"HEAD"), Ref(b"refs/heads/main"))
    return repo
