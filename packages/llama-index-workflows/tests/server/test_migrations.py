from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import List

from workflows.server.sqlite.migrate import run_migrations


def _get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def _get_user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def test_run_migrations_on_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        # Before migrations, version should be 0
        assert _get_user_version(conn) == 0

        run_migrations(conn)

        # Should be at latest version declared by migrations
        assert _get_user_version(conn) >= 2

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

        # Running again should be idempotent and keep version unchanged
        before = _get_user_version(conn)
        run_migrations(conn)
        after = _get_user_version(conn)
        assert after == before
    finally:
        conn.close()


def test_migrate_from_version_1(tmp_path: Path) -> None:
    db_path = tmp_path / "test_v1.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        # Simulate an existing v1 schema (minimal table and user_version=1)
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

        assert _get_user_version(conn) == 1

        run_migrations(conn)

        # Should advance to at least version 2
        assert _get_user_version(conn) >= 2

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
