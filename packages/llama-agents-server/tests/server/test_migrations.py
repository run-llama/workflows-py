# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import sqlite3
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from llama_agents.server._store import SQLITE_MIGRATION_SOURCE
from llama_agents.server._store.sqlite.migrate import run_migrations

_FAKE_MIGRATION_SQL = """\
-- migration: 1

CREATE TABLE IF NOT EXISTS fake_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq_num INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS fake_lifecycle (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'active',
    updated_at TEXT NOT NULL
);
"""


@pytest.fixture()
def fake_migrations_pkg(tmp_path: Path) -> Generator[str]:
    """Create a temporary importable migrations package and return its dotted name."""
    pkg_dir = tmp_path / "fake_migrations"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "0001_init.sql").write_text(_FAKE_MIGRATION_SQL)
    sys.path.insert(0, str(tmp_path))
    yield "fake_migrations"
    sys.path.remove(str(tmp_path))


def _get_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def _max_version(conn: sqlite3.Connection, package: str = "server") -> int:
    row = conn.execute(
        "SELECT MAX(version) FROM schema_migrations WHERE package = ?",
        (package,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def test_run_migrations_on_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        run_migrations(conn)

        # Should track versions in schema_migrations
        assert _max_version(conn) >= 2

        # Handlers table should have all extended columns
        columns = _get_table_columns(conn, "handlers")
        for expected in [
            "handler_id",
            "workflow_name",
            "status",
            "ctx",
            "run_id",
            "error",
            "result",
            "started_at",
            "updated_at",
            "completed_at",
        ]:
            assert expected in columns

        # Running again should be idempotent
        before = _max_version(conn)
        run_migrations(conn)
        after = _max_version(conn)
        assert after == before
    finally:
        conn.close()


def test_migrate_from_version_1(tmp_path: Path) -> None:
    """Simulates upgrading an existing deployment with PRAGMA user_version=1."""
    db_path = tmp_path / "test_v1.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            PRAGMA user_version=1;
            CREATE TABLE IF NOT EXISTS handlers (
                handler_id TEXT PRIMARY KEY,
                workflow_name TEXT,
                status TEXT,
                ctx TEXT
            );
            """
        )

        run_migrations(conn)

        # Legacy user_version should have been bootstrapped into schema_migrations
        assert _max_version(conn) >= 2

        # New columns should be present
        columns = _get_table_columns(conn, "handlers")
        for expected in [
            "run_id",
            "error",
            "result",
            "started_at",
            "updated_at",
            "completed_at",
        ]:
            assert expected in columns
    finally:
        conn.close()


def test_per_package_migrations(tmp_path: Path, fake_migrations_pkg: str) -> None:
    """Test running migrations with multiple package sources."""
    db_path = tmp_path / "test_multi.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        sources = [
            SQLITE_MIGRATION_SOURCE,
            ("fake", fake_migrations_pkg),
        ]
        run_migrations(conn, sources=sources)

        # Server tables should exist
        tables = _get_tables(conn)
        assert "handlers" in tables
        assert "ticks" in tables

        # Fake package tables should exist
        assert "fake_journal" in tables
        assert "fake_lifecycle" in tables

        # Both packages should have version entries
        assert _max_version(conn, "server") >= 1
        assert _max_version(conn, "fake") >= 1

        # Idempotent
        run_migrations(conn, sources=sources)
        assert _max_version(conn, "server") >= 1
    finally:
        conn.close()


def test_legacy_upgrade_with_extra_package(
    tmp_path: Path, fake_migrations_pkg: str
) -> None:
    """Simulate upgrading from released v3 deployment, then adding another package."""
    db_path = tmp_path / "test_upgrade.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        # Simulate existing v3 deployment
        conn.executescript(
            """
            PRAGMA user_version=3;
            CREATE TABLE IF NOT EXISTS handlers (
                handler_id TEXT PRIMARY KEY,
                workflow_name TEXT,
                status TEXT,
                ctx TEXT,
                run_id TEXT,
                error TEXT,
                result TEXT,
                started_at TEXT,
                updated_at TEXT,
                completed_at TEXT,
                idle_since TEXT
            );
            """
        )

        sources = [
            SQLITE_MIGRATION_SOURCE,
            ("fake", fake_migrations_pkg),
        ]
        run_migrations(conn, sources=sources)

        # Server should be seeded at 3 and advanced to 4+
        assert _max_version(conn, "server") >= 4

        # Fake package tables should be created
        tables = _get_tables(conn)
        assert "fake_journal" in tables
        assert "fake_lifecycle" in tables
    finally:
        conn.close()
