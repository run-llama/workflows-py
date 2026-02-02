# SPDX-License-Identifier: MIT
# Copyright (c) 2025, LlamaIndex

"""PostgreSQL testcontainers connectivity tests.

These tests verify that the testcontainers PostgreSQL integration works correctly.
They are marked with the 'docker' marker and require Docker to be running.

Run with: pytest -m docker
"""

from __future__ import annotations

from typing import Generator

import pytest
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="module")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Module-scoped PostgreSQL container for connectivity tests.

    Requires Docker to be running.
    """
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16", driver=None) as postgres:
        yield postgres


@pytest.mark.docker
def test_postgres_container_starts(postgres_container: PostgresContainer) -> None:
    """Test that the PostgreSQL container starts successfully."""
    # Container should be running if we got here
    assert postgres_container is not None
    connection_url = postgres_container.get_connection_url()
    assert "postgresql" in connection_url or "postgres" in connection_url


@pytest.mark.docker
def test_postgres_basic_query(postgres_container: PostgresContainer) -> None:
    """Test basic SQL query execution against the PostgreSQL container."""
    import psycopg

    connection_url = postgres_container.get_connection_url()

    with psycopg.connect(connection_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS test_value")
            result = cur.fetchone()
            assert result is not None
            assert result[0] == 1
