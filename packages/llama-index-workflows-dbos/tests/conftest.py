"""Shared test configuration and fixtures for DBOS tests.

PostgreSQL Testing with py-pglite
---------------------------------
Tests can run with either SQLite (default) or PostgreSQL (via py-pglite).

To run tests with PostgreSQL:
    DBOS_TEST_POSTGRES=1 pytest

Or use the postgres marker:
    pytest -m postgres

PostgreSQL tests may take longer due to PGlite startup time (~2-3s per module).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import pytest

if TYPE_CHECKING:
    from dbos import DBOSConfig
    from py_pglite import PGliteManager


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "postgres: mark test to run with PostgreSQL backend via py-pglite"
    )


def _use_postgres(request: pytest.FixtureRequest) -> bool:
    """Check if PostgreSQL should be used for this test."""
    env_var = os.environ.get("DBOS_TEST_POSTGRES", "").lower() in ("1", "true", "yes")
    marker = request.node.get_closest_marker("postgres") is not None
    return env_var or marker


@pytest.fixture(scope="module")
def pglite_manager(
    request: pytest.FixtureRequest,
) -> Generator[PGliteManager, None, None]:
    """Module-scoped PGlite instance - starts PostgreSQL once per test module.

    Only starts if DBOS_TEST_POSTGRES=1 or any test in the module has @pytest.mark.postgres.
    """
    if not _use_postgres(request):
        pytest.skip("Skipping PGlite fixture - PostgreSQL not requested")

    from py_pglite import PGliteManager as PGliteManagerClass

    manager = PGliteManagerClass()
    manager.start()
    try:
        yield manager
    finally:
        manager.stop()


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
        **TEST_DBOS_DEFAULTS,
    }  # type: ignore[return-value]


def make_test_dbos_postgres_config(
    name: str,
    pglite_manager: "PGliteManager",
) -> "DBOSConfig":
    """Create a DBOS config for testing with PostgreSQL via py-pglite.

    Args:
        name: The application name for DBOS.
        pglite_manager: A running PGliteManager instance.

    Returns:
        A DBOSConfig dictionary ready for use with DBOS().

    Note:
        PGlite only supports a single connection at a time, so we set
        sys_db_pool_size=1 to avoid connection pool issues.
    """
    # PGlite provides a psycopg-compatible connection URI
    psycopg_uri = pglite_manager.get_psycopg_uri()
    return {
        "name": name,
        "system_database_url": psycopg_uri,
        # PGlite only supports single connection - must use pool_size=1
        "sys_db_pool_size": 1,
        **TEST_DBOS_DEFAULTS,
    }  # type: ignore[return-value]
