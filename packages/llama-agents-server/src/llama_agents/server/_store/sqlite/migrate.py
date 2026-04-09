# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import logging
import sqlite3

from llama_agents.server._store import SQLITE_MIGRATION_SOURCE
from llama_agents.server._store.migration_utils import (
    iter_migration_files,
    parse_target_version,
)

logger = logging.getLogger(__name__)

_MIGRATIONS_PKG = SQLITE_MIGRATION_SOURCE[1]

_SCHEMA_MIGRATIONS_DDL = """\
CREATE TABLE IF NOT EXISTS schema_migrations (
    package TEXT NOT NULL,
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (package, version)
)
"""


def _bootstrap_schema_migrations(conn: sqlite3.Connection) -> None:
    """Create the schema_migrations table, seeding from PRAGMA user_version if needed."""
    cur = conn.cursor()

    # Check if schema_migrations already exists
    row = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if row:
        return

    # Check legacy user_version
    uv_row = cur.execute("PRAGMA user_version").fetchone()
    legacy_version = int(uv_row[0]) if uv_row else 0

    cur.executescript(_SCHEMA_MIGRATIONS_DDL)

    if legacy_version > 0:
        # Seed rows for existing server migrations
        for v in range(1, legacy_version + 1):
            cur.execute(
                "INSERT OR IGNORE INTO schema_migrations (package, version) VALUES (?, ?)",
                ("server", v),
            )
        conn.commit()
        logger.debug(
            "Bootstrapped schema_migrations from PRAGMA user_version=%d", legacy_version
        )


def run_migrations(
    conn: sqlite3.Connection,
    sources: list[tuple[str, str]] | None = None,
) -> None:
    """Apply pending migrations for one or more packages.

    Parameters
    ----------
    conn:
        An open SQLite connection.
    sources:
        List of ``(package_name, importable_migrations_pkg)`` pairs.
        Defaults to ``[("server", _MIGRATIONS_PKG)]``.
    """
    # Enable WAL mode for concurrent read/write access.
    # Switching journal mode requires a write lock.  On container/network
    # filesystems (or after a previous crash) the DB may be transiently
    # locked, so retry a few times before falling back to the default
    # (DELETE) journal mode.
    for attempt in range(3):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError:
            if attempt < 2:
                import time

                time.sleep(0.5 * (attempt + 1))
            else:
                logger.warning(
                    "Could not enable WAL journal mode; "
                    "falling back to default journal mode."
                )

    if sources is None:
        sources = [("server", _MIGRATIONS_PKG)]

    _bootstrap_schema_migrations(conn)

    cur = conn.cursor()

    for package_name, source_pkg in sources:
        # Determine already-applied versions for this package
        rows = cur.execute(
            "SELECT version FROM schema_migrations WHERE package = ?",
            (package_name,),
        ).fetchall()
        applied: set[int] = {int(r[0]) for r in rows}

        for path in iter_migration_files(source_pkg):
            sql_text = path.read_text()
            target_version = parse_target_version(sql_text) or 0
            if target_version in applied or target_version == 0:
                continue

            try:
                logger.debug(
                    "Applying migration %s:%s → version %s",
                    package_name,
                    path.name,
                    target_version,
                )
                cur.executescript("BEGIN;\n" + sql_text)
            except Exception as exc:  # noqa: BLE001 – we surface the exact error
                logger.error("Failed migration %s:%s: %s", package_name, path.name, exc)
                cur.execute("ROLLBACK")
                raise
            else:
                cur.execute(
                    "INSERT INTO schema_migrations (package, version) VALUES (?, ?)",
                    (package_name, target_version),
                )
                cur.execute("COMMIT")
                applied.add(target_version)
