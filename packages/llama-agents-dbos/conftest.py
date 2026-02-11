"""Root conftest.py - shared test utilities for all test directories.

This file is discovered by pytest and provides common utilities
for both tests/ (SQLite) and tests_postgres/ (PostgreSQL).
"""

from __future__ import annotations

from pathlib import Path

from dbos import DBOSConfig


def make_test_dbos_config(
    name: str,
    db_path: Path,
) -> DBOSConfig:
    """Create a DBOS config for testing with sensible defaults (SQLite backend).

    Args:
        name: The application name for DBOS.
        db_path: Path to the SQLite database file.

    Returns:
        A DBOSConfig dictionary ready for use with DBOS().
    """
    system_db_url = f"sqlite+pysqlite:///{db_path}?check_same_thread=false"
    return {
        "name": name,
        "system_database_url": system_db_url,
        "run_admin_server": False,
        "notification_listener_polling_interval_sec": 0.01,
    }
