# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

try:
    from importlib.resources.abc import Traversable  # type: ignore
except ImportError:  # pre 3.11
    from importlib.abc import Traversable  # type: ignore
import logging
import re
from importlib import import_module, resources

import asyncpg

logger = logging.getLogger(__name__)

_MIGRATIONS_PKG = "llama_agents.server._store.postgres.migrations"
_VERSION_PATTERN = re.compile(r"--\s*migration:\s*(\d+)")


def _iter_migration_files() -> list[Traversable]:
    """Yield packaged SQL migration files in lexicographic order."""
    pkg = import_module(_MIGRATIONS_PKG)
    root = resources.files(pkg)
    files = (p for p in root.iterdir() if p.name.endswith(".sql"))
    return sorted(files, key=lambda p: p.name)  # type: ignore


def _parse_target_version(sql_text: str) -> int | None:
    """Return target schema version declared in the first line comment."""
    first_line = sql_text.splitlines()[0] if sql_text else ""
    match = _VERSION_PATTERN.search(first_line)
    return int(match.group(1)) if match else None


async def run_migrations(conn: asyncpg.Connection, schema: str | None = None) -> None:
    """Apply pending migrations found under the migrations package.

    Each migration file should start with a `-- migration: N` line.
    Files are applied in lexicographic order and only when N > current_version.
    """
    if schema:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")  # noqa: S608
        await conn.execute(f"SET search_path TO {schema}")  # noqa: S608

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    row = await conn.fetchval("SELECT MAX(version) FROM schema_migrations")
    current_version = row if row is not None else 0

    for path in _iter_migration_files():
        sql_text = path.read_text()
        target_version = _parse_target_version(sql_text) or 0
        if target_version <= current_version:
            continue

        try:
            logger.debug(
                "Applying migration %s -> target version %s", path.name, target_version
            )
            async with conn.transaction():
                await conn.execute(sql_text)
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)",
                    target_version,
                )
        except Exception:
            logger.error("Failed migration %s", path.name)
            raise

        current_version = target_version
