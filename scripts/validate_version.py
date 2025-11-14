#!/usr/bin/env python3
"""Validate that release tag matches pyproject.toml version."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from packaging.version import Version


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate package version vs tag.")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).parent.parent / "pyproject.toml",
        help="Path to the pyproject.toml that should be validated.",
    )
    parser.add_argument(
        "--tag-prefix",
        default="",
        help="Static prefix that appears before the semantic version in the git tag.",
    )
    return parser.parse_args()


def get_pyproject_version(pyproject_path: Path) -> str:
    """Extract version from pyproject.toml."""
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()
        project_match = re.search(r"\[project\]", content)
        if not project_match:
            raise ValueError("Could not find [project] section in pyproject.toml")

        version_pattern = r'version\s*=\s*["\']([^"\']+)["\']'
        version_match = re.search(version_pattern, content[project_match.start() :])
        if not version_match:
            raise ValueError("Could not find version in pyproject.toml")

        return version_match.group(1)


def extract_tag_version(tag: str, prefix: str) -> str:
    """Strip the package prefix and leading 'v' from the tag value."""
    if prefix and not tag.startswith(prefix):
        raise ValueError(f"Tag {tag} does not start with expected prefix {prefix}")
    suffix = tag[len(prefix) :] if prefix else tag
    return suffix[1:] if suffix.startswith("v") else suffix


def main() -> None:
    args = parse_args()

    try:
        pyproject_version = get_pyproject_version(args.pyproject)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        sys.exit(1)

    github_ref = os.environ.get("GITHUB_REF", "")
    if not github_ref.startswith("refs/tags/"):
        print("Error: Not a tag push")
        sys.exit(1)

    tag_name = github_ref.replace("refs/tags/", "")

    try:
        tag_version = extract_tag_version(tag_name, args.tag_prefix)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    try:
        pyproject_ver = Version(pyproject_version)
        tag_ver = Version(tag_version)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: Invalid version format - {exc}")
        sys.exit(1)

    if pyproject_ver != tag_ver:
        print(
            f"Error: Tag {tag_name} (version {tag_version}) doesn't match pyproject.toml version {pyproject_version}"
        )
        sys.exit(1)

    print(f"✅ Version validated: {pyproject_version} (tag: {tag_name})")


if __name__ == "__main__":
    main()
