# Re-export fixtures from server_test_fixtures for pytest discovery
from __future__ import annotations

from collections.abc import Generator

import pytest
from llama_agents_integration_tests.postgres import (
    get_asyncpg_dsn,
    postgres_container,
)
from server_test_fixtures import *  # noqa: F401, F403


@pytest.fixture(scope="module")
def postgres_dsn() -> Generator[str, None, None]:
    """Module-scoped disposable Postgres via testcontainers; yields an asyncpg DSN."""
    with postgres_container() as pg:
        yield get_asyncpg_dsn(pg)
