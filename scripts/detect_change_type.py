#!/usr/bin/env python3
"""Detect the type of change based on version tags."""

import os
import subprocess
import sys
from typing import Optional

from packaging.version import Version


def get_previous_tag() -> Optional[str]:
    """Get the previous release tag."""
    try:
        # Get all tags sorted by version
        result = subprocess.run(
            ["git", "tag", "-l", "v*", "--sort=-version:refname"],
            capture_output=True,
            text=True,
            check=True,
        )
        tags = result.stdout.strip().split("\n")

        # Current tag is the first one (if we're on a tag)
        current_ref = os.environ.get("GITHUB_REF", "")
        if current_ref.startswith("refs/tags/"):
            current_tag = current_ref.replace("refs/tags/", "")
            # Find current tag in list and return the next one
            if current_tag in tags:
                current_index = tags.index(current_tag)
                if current_index + 1 < len(tags):
                    return tags[current_index + 1]

        # If not on a tag or no previous tag found
        return tags[0] if tags else None
    except subprocess.CalledProcessError:
        return None


def detect_change_type(current_version: str, previous_version: Optional[str]) -> str:
    """Detect the type of change based on version comparison."""
    if not previous_version:
        # First release or no previous version
        return "major"

    try:
        # Remove 'v' prefix if present
        current_clean = current_version.lstrip("v")
        previous_clean = previous_version.lstrip("v")

        curr_version = Version(current_clean)
        prev_version = Version(previous_clean)

        # Compare versions using packaging.version
        if curr_version <= prev_version:
            # Same or lower version (shouldn't happen in normal flow)
            return "none"

        # Extract major.minor.patch components for semantic comparison
        curr_release = curr_version.release
        prev_release = prev_version.release

        # Ensure we have at least 3 components (major, minor, patch)
        curr_major, curr_minor, curr_patch = (curr_release + (0, 0, 0))[:3]
        prev_major, prev_minor, prev_patch = (prev_release + (0, 0, 0))[:3]

        if curr_major > prev_major:
            return "major"
        elif curr_minor > prev_minor:
            return "minor"
        elif curr_patch > prev_patch:
            return "patch"
        else:
            # This shouldn't happen if curr_version > prev_version
            return "minor"
    except Exception:
        # If we can't parse versions, default to minor
        return "minor"


def main() -> None:
    """Main function to detect change type."""
    # Get current tag from GITHUB_REF
    github_ref = os.environ.get("GITHUB_REF", "")
    if not github_ref.startswith("refs/tags/"):
        print("Not a tag push, no change type detection needed")
        sys.exit(0)

    current_tag = github_ref.replace("refs/tags/", "")
    previous_tag = get_previous_tag()

    change_type = detect_change_type(current_tag, previous_tag)

    print(f"Current tag: {current_tag}")
    if previous_tag:
        print(f"Previous tag: {previous_tag}")
    else:
        print("No previous tag found")
    print(f"Change type: {change_type}")

    # Output for GitHub Actions
    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a") as f:
            f.write(f"change_type={change_type}\n")


if __name__ == "__main__":
    main()
