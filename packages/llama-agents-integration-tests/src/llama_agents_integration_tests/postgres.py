# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Reusable PostgreSQL testcontainers utilities for integration tests."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from testcontainers.postgres import PostgresContainer


@contextmanager
def postgres_container(
    image: str = "postgres:16",
) -> Generator[PostgresContainer, None, None]:
    """Start a disposable Postgres container. Yields the container object.

    Use ``get_connection_url()`` on the result to get a connection string.
    The ``driver=None`` argument ensures the raw ``postgresql://`` scheme
    is used (no psycopg2/psycopg suffix).
    """
    with PostgresContainer(image, driver=None) as pg:
        yield pg


def get_asyncpg_dsn(container: PostgresContainer) -> str:
    """Return a plain ``postgresql://`` DSN suitable for asyncpg."""
    url = container.get_connection_url()
    # testcontainers may return psycopg2-style URLs
    return url.replace("postgresql+psycopg2://", "postgresql://")
