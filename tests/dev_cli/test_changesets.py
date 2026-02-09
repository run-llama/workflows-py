# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import pytest

from dev_cli.changesets import (
    PackageJson,
    PyProjectContainer,
    current_version,
    is_published,
    pep440_to_semver,
    semver_to_pep440,
    sync_package_version_with_pyproject,
)


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
    mock_error = HTTPError("url", 404, "Not Found", {}, None)  # type: ignore[arg-type]

    with patch("urllib.request.urlopen", side_effect=mock_error):
        result = is_published("nonexistent-package", "1.0.0")
        assert result is False


def test_is_published_raises_on_other_http_errors() -> None:
    mock_error = HTTPError("url", 500, "Internal Server Error", {}, None)  # type: ignore[arg-type]

    with patch("urllib.request.urlopen", side_effect=mock_error):
        with pytest.raises(HTTPError) as exc_info:
            is_published("test-package", "1.0.0")
        assert exc_info.value.code == 500


def test_sync_package_version_updates_version(tmp_path: Path) -> None:
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

    _, py_doc = PyProjectContainer.parse(pyproject.read_text())
    assert py_doc.project.version == "2.0.0"


def test_sync_package_version_skips_when_no_pyproject(tmp_path: Path) -> None:
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


def test_sync_package_version_skips_when_versions_match(tmp_path: Path) -> None:
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


def test_sync_package_version_converts_semver_prerelease(tmp_path: Path) -> None:
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
            version="1.2.3-a.4",
            path=package_dir,
            private=False,
        )
    }

    sync_package_version_with_pyproject(package_dir, packages, "test-js-package")

    _, py_doc = PyProjectContainer.parse(pyproject.read_text())
    assert py_doc.project.version == "1.2.3a4"


# -- semver_to_pep440 tests --


@pytest.mark.parametrize(
    ("semver", "expected"),
    [
        ("1.2.3-a.0", "1.2.3a0"),
        ("1.2.3-a.4", "1.2.3a4"),
        ("2.0.0-b.1", "2.0.0b1"),
        ("0.1.0-rc.3", "0.1.0rc3"),
        ("10.20.30-a.99", "10.20.30a99"),
    ],
)
def test_semver_to_pep440_prerelease(semver: str, expected: str) -> None:
    assert semver_to_pep440(semver) == expected


@pytest.mark.parametrize(
    "version",
    [
        "1.2.3",
        "0.0.1",
        "10.20.30",
    ],
)
def test_semver_to_pep440_stable_passthrough(version: str) -> None:
    assert semver_to_pep440(version) == version


def test_semver_to_pep440_rejects_non_pep440_label() -> None:
    with pytest.raises(ValueError, match="Unsupported pre-release label 'alpha'"):
        semver_to_pep440("1.2.3-alpha.1")


# -- pep440_to_semver tests --


@pytest.mark.parametrize(
    ("pep440", "expected"),
    [
        ("1.2.3a0", "1.2.3-a.0"),
        ("1.2.3a4", "1.2.3-a.4"),
        ("2.0.0b1", "2.0.0-b.1"),
        ("0.1.0rc3", "0.1.0-rc.3"),
        ("10.20.30a99", "10.20.30-a.99"),
    ],
)
def test_pep440_to_semver_prerelease(pep440: str, expected: str) -> None:
    assert pep440_to_semver(pep440) == expected


@pytest.mark.parametrize(
    "version",
    [
        "1.2.3",
        "0.0.1",
        "10.20.30",
    ],
)
def test_pep440_to_semver_stable_passthrough(version: str) -> None:
    assert pep440_to_semver(version) == version


# -- roundtrip tests --


@pytest.mark.parametrize(
    "semver",
    [
        "1.2.3-a.4",
        "2.0.0-b.1",
        "0.1.0-rc.3",
    ],
)
def test_roundtrip_semver_to_pep440_and_back(semver: str) -> None:
    pep440 = semver_to_pep440(semver)
    assert pep440_to_semver(pep440) == semver


@pytest.mark.parametrize(
    "pep440",
    [
        "1.2.3a4",
        "2.0.0b1",
        "0.1.0rc3",
    ],
)
def test_roundtrip_pep440_to_semver_and_back(pep440: str) -> None:
    semver = pep440_to_semver(pep440)
    assert semver_to_pep440(semver) == pep440
