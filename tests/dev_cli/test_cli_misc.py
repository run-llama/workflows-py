# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for miscellaneous CLI commands (tag-metadata, update-index-html)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from conftest import commit_and_tag

from dev_cli.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# --- compute-tag-metadata tests ---


@pytest.mark.parametrize(
    ("from_version", "to_version", "expected_type"),
    [
        ("v1.0.0", "v1.0.1", "patch"),
        ("v1.0.0", "v1.1.0", "minor"),
        ("v1.0.0", "v2.0.0", "major"),
    ],
)
def test_compute_tag_metadata_change_types(
    runner: CliRunner,
    git_repo: Path,
    from_version: str,
    to_version: str,
    expected_type: str,
) -> None:
    commit_and_tag(git_repo, "file.txt", f"pkg@{from_version}", f"pkg@{from_version}")
    commit_and_tag(git_repo, "file.txt", f"pkg@{to_version}", f"pkg@{to_version}")

    result = runner.invoke(
        cli,
        ["compute-tag-metadata", "--tag", f"pkg@{to_version}"],
        env={},
    )
    assert result.exit_code == 0
    assert f"Change type: {expected_type}" in result.output


def test_compute_tag_metadata_writes_output_file(
    runner: CliRunner, git_repo: Path
) -> None:
    commit_and_tag(git_repo, "file.txt", "pkg@v1.0.0", "pkg@v1.0.0")
    commit_and_tag(git_repo, "file.txt", "pkg@v1.0.1", "pkg@v1.0.1")

    output_file = git_repo / "out.txt"
    result = runner.invoke(
        cli,
        ["compute-tag-metadata", "--tag", "pkg@v1.0.1", "--output", str(output_file)],
        env={},
    )
    assert result.exit_code == 0
    contents = output_file.read_text()
    assert "tag_suffix=v1.0.1" in contents
    assert "semver=1.0.1" in contents
    assert "change_type=patch" in contents


def test_compute_tag_metadata_requires_tag(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["compute-tag-metadata"])
    assert result.exit_code != 0
    assert "Missing option '--tag'" in result.output


# --- update-index-html tests ---


def test_update_index_html_success(runner: CliRunner, tmp_path: Path) -> None:
    index_path = tmp_path / "index.html"
    index_path.write_text(
        """
<html>
  <head>
    <script type="module" crossorigin src="old.js"></script>
    <link rel="stylesheet" crossorigin href="old.css">
  </head>
</html>
""".strip()
    )
    result = runner.invoke(
        cli,
        [
            "update-index-html",
            "--js-url",
            "https://cdn/js",
            "--css-url",
            "https://cdn/css",
            "--index-path",
            str(index_path),
        ],
    )
    assert result.exit_code == 0
    updated = index_path.read_text()
    assert 'src="https://cdn/js"' in updated
    assert 'href="https://cdn/css"' in updated


def test_update_index_html_missing_file(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        cli,
        [
            "update-index-html",
            "--js-url",
            "https://cdn/js",
            "--css-url",
            "https://cdn/css",
            "--index-path",
            str(tmp_path / "missing.html"),
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output
