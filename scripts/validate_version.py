#!/usr/bin/env python3
"""Validate that release tag matches pyproject.toml version."""

import os
import sys
from pathlib import Path


def main() -> None:
    # Get pyproject.toml version
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path) as f:
        for line in f:
            if line.startswith("version"):
                pyproject_version = line.split('"')[1]
                break
        else:
            print("Error: Version not found in pyproject.toml")
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