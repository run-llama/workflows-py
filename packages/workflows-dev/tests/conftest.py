# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Anthropic
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def packages_dir(tmp_path: Path) -> Path:
    """Create a packages directory structure."""
    packages = tmp_path / "packages"
    packages.mkdir()
    return packages


@pytest.fixture
def create_test_package(packages_dir: Path) -> Callable[[str], Path]:
    """Factory fixture to create packages with tests subdirectory."""

    def _create(name: str) -> Path:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()
        return pkg

    return _create


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Initialize a git repository and change into it."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Dev User"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def commit_and_tag(repo_path: Path, filename: str, content: str, tag: str) -> None:
    """Create a file, commit it, and tag it."""
    file_path = repo_path / filename
    file_path.write_text(content)
    subprocess.run(["git", "add", filename], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Add {tag}"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "tag", tag], cwd=repo_path, check=True)


def is_sync_call(cmd: list[str]) -> bool:
    """Check if a subprocess command is a 'uv sync' call vs 'uv run pytest'."""
    return len(cmd) >= 2 and cmd[0] == "uv" and cmd[1] == "sync"


def sync_success() -> Mock:
    """Return a successful sync result."""
    return Mock(returncode=0, stdout="", stderr="")


def write_pyproject(path: Path, version: str) -> None:
    """Write a minimal pyproject.toml with the given version."""
    path.write_text(
        f"""
[project]
name = "example"
version = "{version}"
""".strip()
    )
