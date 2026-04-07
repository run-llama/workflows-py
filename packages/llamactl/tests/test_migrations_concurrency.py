"""Tests for concurrent migration protection in _migrations.py."""

import sqlite3
import tempfile
import threading
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from llama_agents.cli.config._migrations import (
    _file_lock,
    _iter_migration_files,
    _parse_target_version,
    run_migrations,
)


def _latest_migration_version() -> int:
    """Determine the latest schema version from migration files."""
    versions = []
    for path in _iter_migration_files():
        v = _parse_target_version(path.read_text())
        if v is not None:
            versions.append(v)
    return max(versions)


@pytest.fixture
def temp_db() -> Generator[Path, None, None]:
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        yield db_path


def test_file_lock_serializes_access(temp_db: Path) -> None:
    """Test that file lock prevents concurrent access."""
    lock_path = temp_db.with_suffix(".db.lock")
    results: list[str] = []
    lock_held = threading.Event()

    def thread_a() -> None:
        with _file_lock(lock_path):
            results.append("a_start")
            lock_held.set()
            time.sleep(0.1)
            results.append("a_end")

    def thread_b() -> None:
        lock_held.wait()  # Wait for thread_a to acquire lock
        with _file_lock(lock_path):
            results.append("b_start")
            results.append("b_end")

    t1 = threading.Thread(target=thread_a)
    t2 = threading.Thread(target=thread_b)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Thread B should only start after thread A finishes
    assert results == ["a_start", "a_end", "b_start", "b_end"]


def test_run_migrations_with_lock_prevents_duplicate_migrations(temp_db: Path) -> None:
    """Test that concurrent migrations don't cause duplicate schema changes."""
    # Create the initial database with schema version 0
    with sqlite3.connect(temp_db) as conn:
        conn.execute("PRAGMA user_version=0")
        conn.commit()

    errors: list[Exception] = []
    success_count: list[int] = []
    barrier = threading.Barrier(2)

    def run_migration() -> None:
        try:
            barrier.wait()  # Synchronize both threads to start at the same time
            with sqlite3.connect(temp_db) as conn:
                run_migrations(conn, temp_db)
            success_count.append(1)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=run_migration)
    t2 = threading.Thread(target=run_migration)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both should complete without errors
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(success_count) == 2

    # Verify the schema version is correct
    with sqlite3.connect(temp_db) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        # Should be at version 2 (the latest migration)
        assert version == _latest_migration_version()


def test_run_migrations_without_lock_path_still_works(temp_db: Path) -> None:
    """Test that migrations work when db_path is not provided (no locking)."""
    with sqlite3.connect(temp_db) as conn:
        conn.execute("PRAGMA user_version=0")
        conn.commit()
        run_migrations(conn, db_path=None)

    with sqlite3.connect(temp_db) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == _latest_migration_version()
