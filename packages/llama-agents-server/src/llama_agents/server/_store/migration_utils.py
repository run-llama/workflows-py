# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

try:
    from importlib.resources.abc import Traversable  # type: ignore  # ty: ignore[unresolved-import]
except ImportError:  # pre 3.11
    from importlib.abc import Traversable  # type: ignore  # ty: ignore[unresolved-import]
import re
from importlib import import_module, resources

VERSION_PATTERN = re.compile(r"--\s*migration:\s*(\d+)")


def iter_migration_files(source_pkg: str) -> list[Traversable]:
    """Return packaged SQL migration files in lexicographic order."""
    pkg = import_module(source_pkg)
    root = resources.files(pkg)
    files = (p for p in root.iterdir() if p.name.endswith(".sql"))
    return sorted(files, key=lambda p: p.name)  # type: ignore  # ty: ignore[invalid-return-type]


def parse_target_version(sql_text: str) -> int | None:
    """Return target schema version declared in a ``-- migration: N`` comment."""
    first_line = sql_text.splitlines()[0] if sql_text else ""
    match = VERSION_PATTERN.search(first_line)
    return int(match.group(1)) if match else None
