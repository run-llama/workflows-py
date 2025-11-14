from __future__ import annotations

from pathlib import Path
from typing import Optional

from packaging.version import Version

try:  # pragma: no cover - Python 3.11+ includes tomllib
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


class VersionMismatchError(ValueError):
    """Raised when two versions do not match."""


def read_pyproject_version(pyproject_path: str) -> str:
    """Read the project version from a pyproject.toml file."""
    path = Path(pyproject_path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        project = data["project"]
    except KeyError as exc:  # pragma: no cover - invalid pyproject structure
        raise ValueError("Missing [project] section in pyproject.toml") from exc

    try:
        version = project["version"]
    except KeyError as exc:
        raise ValueError("Missing version metadata in pyproject.toml") from exc

    if not isinstance(version, str):  # pragma: no cover - defensive guard
        raise ValueError("Project version must be a string.")
    return version


def strip_refs_prefix(tag: str) -> str:
    """Remove the refs/tags/ prefix when present."""
    return tag.replace("refs/tags/", "") if tag.startswith("refs/tags/") else tag


def remove_tag_prefix(tag: str, tag_prefix: str) -> str:
    """Remove a package-specific prefix and optional leading v."""
    if tag_prefix:
        if not tag.startswith(tag_prefix):
            raise ValueError(f"Tag {tag} does not match expected prefix {tag_prefix}")
        tag = tag[len(tag_prefix) :]
    return tag


def extract_semver(tag: str, tag_prefix: str) -> str:
    """Return the semantic version encoded in a git tag."""
    suffix = remove_tag_prefix(strip_refs_prefix(tag), tag_prefix)
    return suffix[1:] if suffix.startswith("v") else suffix


def compute_suffix_and_version(tag: str, tag_prefix: str) -> tuple[str, str]:
    """Return the suffix after the prefix and the semantic version."""
    suffix = remove_tag_prefix(strip_refs_prefix(tag), tag_prefix)
    semver = suffix[1:] if suffix.startswith("v") else suffix
    return suffix, semver


def detect_change_type(current_version: str, previous_version: Optional[str]) -> str:
    """Return the semantic change classification between two versions."""
    if not previous_version:
        return "major"

    try:
        current = Version(current_version)
        previous = Version(previous_version)
    except Exception:  # pragma: no cover - defensive guard
        return "minor"

    if current <= previous:
        return "none"

    current_release = (current.release + (0, 0, 0))[:3]
    previous_release = (previous.release + (0, 0, 0))[:3]

    if current_release[0] > previous_release[0]:
        return "major"
    if current_release[1] > previous_release[1]:
        return "minor"
    if current_release[2] > previous_release[2]:
        return "patch"
    return "minor"


def ensure_versions_match(expected: str, actual: str, tag_name: str) -> None:
    """Raise when two version strings differ."""
    if Version(expected) != Version(actual):
        raise VersionMismatchError(
            f"Tag {tag_name} (version {actual}) doesn't match pyproject.toml version {expected}"
        )
