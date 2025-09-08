#!/usr/bin/env python3
"""Validate that release tag matches pyproject.toml version."""

import os
import re
import sys
from pathlib import Path

from packaging.version import Version


def get_pyproject_version() -> str:
    """Extract version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "r") as f:
        content = f.read()
        # Look for version = "x.x.x" in the [project] section
        # First find the [project] section
        project_match = re.search(r"\[project\]", content)
        if not project_match:
            raise ValueError("Could not find [project] section in pyproject.toml")

        # Then find the version line after [project]
        version_pattern = r'version\s*=\s*["\']([^"\']+)["\']'
        version_match = re.search(version_pattern, content[project_match.start() :])
        if not version_match:
            raise ValueError("Could not find version in pyproject.toml")

        return version_match.group(1)


def main() -> None:
    # Get pyproject.toml version
    try:
        pyproject_version = get_pyproject_version()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get tag from GitHub ref
    github_ref = os.environ.get("GITHUB_REF", "")
    if not github_ref.startswith("refs/tags/"):
        print("Error: Not a tag push")
        sys.exit(1)

    tag = github_ref.replace("refs/tags/", "")
    tag_version = tag[1:] if tag.startswith("v") else tag

    # Validate versions match using packaging.version for robust comparison
    try:
        pyproject_ver = Version(pyproject_version)
        tag_ver = Version(tag_version)

        if pyproject_ver != tag_ver:
            print(
                f"Error: Tag {tag} (version {tag_version}) doesn't match pyproject.toml version {pyproject_version}"
            )
            sys.exit(1)

        print(f"✅ Version validated: {pyproject_version} (tag: {tag})")
    except Exception as e:
        print(f"Error: Invalid version format - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
