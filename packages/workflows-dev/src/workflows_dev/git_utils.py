from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Optional


def list_tags(repo: str | Path, tag_glob: str) -> list[str]:
    """Return tags that match the provided glob sorted newest first."""
    repo_path = Path(repo)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_path),
            "tag",
            "-l",
            tag_glob,
            "--sort=-version:refname",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def previous_tag(current_tag: str, tags: Iterable[str]) -> Optional[str]:
    """Return the tag immediately after the current entry in the sorted list."""
    tags_list = list(tags)
    if current_tag in tags_list:
        idx = tags_list.index(current_tag)
        if idx + 1 < len(tags_list):
            return tags_list[idx + 1]
        return None
    return tags_list[0] if tags_list else None


def find_previous_tag(repo: str | Path, tag_prefix: str, current_tag: str) -> str:
    """Locate the latest tag that matches a prefix but differs from the current tag."""
    matches = list_tags(repo, f"{tag_prefix}*")
    for candidate in matches:
        if candidate != current_tag:
            return candidate
    return ""
