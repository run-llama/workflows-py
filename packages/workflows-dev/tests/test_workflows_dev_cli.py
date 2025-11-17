from __future__ import annotations

import subprocess
from pathlib import Path

from click.testing import CliRunner

from workflows_dev.cli import cli


def _write_pyproject(path: Path, version: str) -> None:
    path.write_text(
        f"""
[project]
name = "example"
version = "{version}"
""".strip()
    )


def _init_git_repo(repo_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "dev@example.com"], cwd=repo_path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Dev User"], cwd=repo_path, check=True
    )


def _commit_and_tag(repo_path: Path, filename: str, content: str, tag: str) -> None:
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


def test_detect_change_type_patch() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        _init_git_repo(repo)
        _commit_and_tag(repo, "file.txt", "v1.0.0", "v1.0.0")
        _commit_and_tag(repo, "file.txt", "v1.0.1", "v1.0.1")

        output_file = Path("out.txt")
        result = runner.invoke(
            cli,
            [
                "detect-change-type",
                "--tag-glob",
                "v*",
                "--current-tag",
                "v1.0.1",
                "--output",
                str(output_file),
            ],
            env={},
        )
        assert result.exit_code == 0
        assert "Change type: patch" in result.output
        assert "change_type=patch" in output_file.read_text()


def test_detect_change_type_minor() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        _init_git_repo(repo)
        _commit_and_tag(repo, "file.txt", "v1.0.0", "v1.0.0")
        _commit_and_tag(repo, "file.txt", "v1.1.0", "v1.1.0")

        result = runner.invoke(
            cli,
            [
                "detect-change-type",
                "--tag-glob",
                "v*",
                "--current-tag",
                "v1.1.0",
            ],
            env={},
        )
        assert result.exit_code == 0
        assert "Change type: minor" in result.output


def test_detect_change_type_major() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        _init_git_repo(repo)
        _commit_and_tag(repo, "file.txt", "v1.0.0", "v1.0.0")
        _commit_and_tag(repo, "file.txt", "v2.0.0", "v2.0.0")

        result = runner.invoke(
            cli,
            [
                "detect-change-type",
                "--tag-glob",
                "v*",
                "--current-tag",
                "v2.0.0",
            ],
            env={},
        )
        assert result.exit_code == 0
        assert "Change type: major" in result.output


def test_detect_change_type_with_prefix() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        _init_git_repo(repo)
        _commit_and_tag(repo, "file.txt", "pkg@v1.0.0", "pkg@v1.0.0")
        _commit_and_tag(repo, "file.txt", "pkg@v1.0.1", "pkg@v1.0.1")

        result = runner.invoke(
            cli,
            [
                "detect-change-type",
                "--tag-glob",
                "pkg@v*",
                "--tag-prefix",
                "pkg@",
                "--current-tag",
                "pkg@v1.0.1",
            ],
            env={},
        )
        assert result.exit_code == 0
        assert "Change type: patch" in result.output


def test_detect_change_type_requires_tag() -> None:
    runner = CliRunner()
    env = {"GITHUB_REF": "", "GITHUB_REF_NAME": ""}
    result = runner.invoke(cli, ["detect-change-type"], env=env)
    assert result.exit_code != 0
    assert "Unable to determine tag" in result.output


def test_extract_tag_info_outputs_suffix() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_file = Path("tag.txt")
        result = runner.invoke(
            cli,
            [
                "extract-tag-info",
                "--tag",
                "pkg@v1.2.3",
                "--tag-prefix",
                "pkg@",
                "--output",
                str(out_file),
            ],
        )
        assert result.exit_code == 0
        contents = out_file.read_text()
        assert "tag_suffix=v1.2.3" in contents
        assert "semver=1.2.3" in contents


def test_find_previous_tag_returns_match() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        _init_git_repo(repo)
        _commit_and_tag(repo, "file.txt", "pkg@v1.0.0", "pkg@v1.0.0")
        _commit_and_tag(repo, "file.txt", "pkg@v1.1.0", "pkg@v1.1.0")

        out_file = Path("prev.txt")
        result = runner.invoke(
            cli,
            [
                "find-previous-tag",
                "--tag-prefix",
                "pkg@",
                "--current-tag",
                "pkg@v1.1.0",
                "--output",
                str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert out_file.read_text().strip() == "previous=pkg@v1.0.0"


def test_needs_release_first_release() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        _init_git_repo(repo)
        (repo / "pyproject.toml").write_text(
            """
[project]
name = "example"
version = "1.0.0"
""".strip()
        )

        output_file = Path("out.txt")
        result = runner.invoke(
            cli,
            [
                "needs-release",
                "--pyproject",
                "pyproject.toml",
                "--tag-prefix",
                "pkg@",
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        contents = output_file.read_text()
        assert "version=1.0.0" in contents
        assert "previous_tag=" in contents
        assert "change_type=major" in contents
        assert "release=true" in contents


def test_needs_release_when_version_has_not_advanced() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        (repo / "pyproject.toml").write_text(
            """
[project]
name = "example"
version = "1.0.0"
""".strip()
        )
        # initialise git and create matching tag
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "dev@example.com"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Dev User"],
            cwd=repo,
            check=True,
        )
        (repo / "file.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "tag", "pkg@v1.0.0"], cwd=repo, check=True)

        output_file = Path("out.txt")
        result = runner.invoke(
            cli,
            [
                "needs-release",
                "--pyproject",
                "pyproject.toml",
                "--tag-prefix",
                "pkg@",
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        contents = output_file.read_text()
        assert "version=1.0.0" in contents
        assert "previous_tag=pkg@v1.0.0" in contents
        assert "change_type=none" in contents
        assert "release=false" in contents


def test_update_index_html_success(tmp_path: Path) -> None:
    runner = CliRunner()
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


def test_update_index_html_missing_file(tmp_path: Path) -> None:
    runner = CliRunner()
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
