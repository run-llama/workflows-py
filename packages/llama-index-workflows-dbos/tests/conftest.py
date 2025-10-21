"""Shared test configuration and fixtures for DBOS tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbos import DBOSConfig


# Default test configuration values
TEST_DBOS_DEFAULTS: dict[str, object] = {
    "run_admin_server": False,
    "internal_polling_interval_sec": 0.01,
}

# For subprocess tests that embed config in f-strings, these lines can be included
# after the "system_database_url" line. Use with double braces in f-strings.
TEST_DBOS_CONFIG_INLINE = """\
                "run_admin_server": False,
                "internal_polling_interval_sec": 0.01,"""


def make_test_dbos_config(
    name: str,
    db_path: Path,
) -> "DBOSConfig":
    """Create a DBOS config for testing with sensible defaults.

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
        **TEST_DBOS_DEFAULTS,
    }  # type: ignore[return-value]
