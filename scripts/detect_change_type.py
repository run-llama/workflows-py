#!/usr/bin/env python3
"""Detect the type of semantic change between release tags."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional

from packaging.version import Version


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Detect change type between semantic version tags."
    )
    parser.add_argument(
        "--tag-glob",
        default="v*",
        help="Glob used when listing tags (defaults to v*).",
    )
    parser.add_argument(
        "--tag-prefix",
        default="",
        help="Static prefix to strip from tags before parsing the version number.",
    )
    return parser.parse_args()


def list_tags(tag_glob: str) -> list[str]:
    """Return tags matching the provided glob, sorted newest first."""
    try:
        result = subprocess.run(
            ["git", "tag", "-l", tag_glob, "--sort=-version:refname"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_previous_tag(current_tag: str, tags: list[str]) -> Optional[str]:
    """Return the tag immediately preceding the current tag."""
    if current_tag in tags:
        idx = tags.index(current_tag)
        if idx + 1 < len(tags):
            return tags[idx + 1]
        return None
    return tags[0] if tags else None


def extract_version(tag: str, prefix: str) -> str:
    """Remove the configured prefix and leading v from the tag."""
    if prefix:
        if not tag.startswith(prefix):
            raise ValueError(f"Tag {tag} does not start with expected prefix {prefix}")
        tag = tag[len(prefix) :]
    return tag[1:] if tag.startswith("v") else tag


def detect_change_type(current_version: str, previous_version: Optional[str]) -> str:
    """Detect the type of change based on version comparison."""
    if not previous_version:
        return "major"

    try:
        curr_version = Version(current_version)
        prev_version = Version(previous_version)
    except Exception:  # noqa: BLE001
        return "minor"

    if curr_version <= prev_version:
        return "none"

    curr_release = curr_version.release
    prev_release = prev_version.release

    curr_major, curr_minor, curr_patch = (curr_release + (0, 0, 0))[:3]
    prev_major, prev_minor, prev_patch = (prev_release + (0, 0, 0))[:3]

    if curr_major > prev_major:
        return "major"
    if curr_minor > prev_minor:
        return "minor"
    if curr_patch > prev_patch:
        return "patch"
    return "minor"


def main() -> None:
    """Main entry point."""
    args = parse_args()

    github_ref = os.environ.get("GITHUB_REF", "")
    if not github_ref.startswith("refs/tags/"):
        print("Not a tag push, no change type detection needed")
        sys.exit(0)

    current_tag = github_ref.replace("refs/tags/", "")
    tags = list_tags(args.tag_glob)
    previous_tag = get_previous_tag(current_tag, tags)

    try:
        current_version = extract_version(current_tag, args.tag_prefix)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    previous_version: Optional[str] = None
    if previous_tag:
        try:
            previous_version = extract_version(previous_tag, args.tag_prefix)
        except ValueError:
            previous_version = None

    change_type = detect_change_type(current_version, previous_version)

    print(f"Current tag: {current_tag}")
    if previous_tag:
        print(f"Previous tag: {previous_tag}")
    else:
        print("No previous tag found")
    print(f"Change type: {change_type}")

    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"change_type={change_type}\n")


if __name__ == "__main__":
    main()
