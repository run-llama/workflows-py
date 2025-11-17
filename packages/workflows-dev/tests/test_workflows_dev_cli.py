from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from click.testing import CliRunner

from workflows_dev.changesets import (
    PackageJson,
    PyProjectContainer,
    current_version,
    is_published,
    sync_package_version_with_pyproject,
)
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


# Tests for changesets.py functionality


def test_current_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "test-package"
version = "1.2.3"
dependencies = []
""".strip()
    )
    name, version = current_version(pyproject)
    assert name == "test-package"
    assert version == "1.2.3"


def test_current_version_normalizes_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "test-package"
version = "01.02.03"
dependencies = []
""".strip()
    )
    name, version = current_version(pyproject)
    assert name == "test-package"
    assert version == "1.2.3"


def test_pyproject_container_parse() -> None:
    toml_text = """
[project]
name = "my-package"
version = "0.1.0"
dependencies = ["requests>=2.0.0"]
""".strip()
    toml_doc, py_doc = PyProjectContainer.parse(toml_text)
    assert py_doc.project.name == "my-package"
    assert py_doc.project.version == "0.1.0"
    assert py_doc.project.dependencies == ["requests>=2.0.0"]


def test_is_published_returns_true_when_version_exists() -> None:
    mock_response = Mock()
    mock_response.read.return_value = json.dumps(
        {"releases": {"1.0.0": [], "1.1.0": []}}
    ).encode()

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = is_published("test-package", "1.0.0")
        assert result is True


def test_is_published_returns_false_when_version_missing() -> None:
    mock_response = Mock()
    mock_response.read.return_value = json.dumps({"releases": {"1.0.0": []}}).encode()

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = is_published("test-package", "1.1.0")
        assert result is False


def test_is_published_returns_false_when_package_not_found() -> None:
    mock_error = HTTPError("url", 404, "Not Found", {}, None)  # type: ignore

    with patch("urllib.request.urlopen", side_effect=mock_error):
        result = is_published("nonexistent-package", "1.0.0")
        assert result is False


def test_is_published_raises_on_other_http_errors() -> None:
    mock_error = HTTPError("url", 500, "Internal Server Error", {}, None)  # type: ignore

    with patch("urllib.request.urlopen", side_effect=mock_error):
        try:
            is_published("test-package", "1.0.0")
            assert False, "Expected HTTPError to be raised"
        except HTTPError as e:
            assert e.code == 500


def test_sync_package_version_with_pyproject_updates_version(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    pyproject = package_dir / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "test-package"
version = "1.0.0"
dependencies = []
""".strip()
    )

    packages = {
        "test-js-package": PackageJson(
            name="test-js-package",
            version="2.0.0",
            path=package_dir,
            private=False,
        )
    }

    sync_package_version_with_pyproject(package_dir, packages, "test-js-package")

    toml_doc, py_doc = PyProjectContainer.parse(pyproject.read_text())
    assert py_doc.project.version == "2.0.0"


def test_sync_package_version_with_pyproject_skips_when_no_pyproject(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()

    packages = {
        "test-js-package": PackageJson(
            name="test-js-package",
            version="2.0.0",
            path=package_dir,
            private=False,
        )
    }

    # Should not raise an error
    sync_package_version_with_pyproject(package_dir, packages, "test-js-package")


def test_sync_package_version_with_pyproject_skips_when_versions_match(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    pyproject = package_dir / "pyproject.toml"
    original_content = """
[project]
name = "test-package"
version = "2.0.0"
dependencies = []
""".strip()
    pyproject.write_text(original_content)

    packages = {
        "test-js-package": PackageJson(
            name="test-js-package",
            version="2.0.0",
            path=package_dir,
            private=False,
        )
    }

    sync_package_version_with_pyproject(package_dir, packages, "test-js-package")

    # Content should be unchanged
    assert pyproject.read_text() == original_content
