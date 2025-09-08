#!/usr/bin/env python3
"""Validate that release tag matches pyproject.toml version."""

import os
import sys
from pathlib import Path
import tomllib

def get_pyproject_version() -> str:
    """Extract version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
        return data["project"]["version"]


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
    
    # Validate versions match
    if pyproject_version != tag_version:
        print(f"Error: Tag {tag} doesn't match pyproject.toml version {pyproject_version}")
        sys.exit(1)
    
    print(f"✅ Version validated: {pyproject_version} (tag: {tag})")


if __name__ == "__main__":
    main()