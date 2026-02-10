# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import os

import asyncpg
import pytest
from llama_agents.server._store.postgres.migrate import (
    _iter_migration_files,
    _parse_target_version,
    run_migrations,
)

pytestmark = [pytest.mark.no_cover, pytest.mark.asyncio]

POSTGRES_DSN = os.environ.get("TEST_POSTGRES_DSN")
requires_postgres = pytest.mark.skipif(
    POSTGRES_DSN is None,
    reason="TEST_POSTGRES_DSN not set",
)


# ── Unit tests (no DB) ──────────────────────────────────────────────


def test_parse_target_version_valid() -> None:
    assert _parse_target_version("-- migration: 1\nCREATE TABLE ...") == 1
    assert _parse_target_version("-- migration: 42\n") == 42


def test_parse_target_version_missing() -> None:
    assert _parse_target_version("CREATE TABLE ...") is None
    assert _parse_target_version("") is None


def test_iter_migration_files_returns_sorted_sql() -> None:
    files = _iter_migration_files()
    assert len(files) >= 1
    assert all(f.name.endswith(".sql") for f in files)
    names = [f.name for f in files]
    assert names == sorted(names)


def test_first_migration_has_version_1() -> None:
    files = _iter_migration_files()
    sql = files[0].read_text()
    assert _parse_target_version(sql) == 1


# ── Integration tests (require Postgres) ────────────────────────────


@requires_postgres
async def test_run_migrations_fresh_db() -> None:
    assert POSTGRES_DSN is not None
    conn = await asyncpg.connect(POSTGRES_DSN)
    schema = "test_migrate_fresh"
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await run_migrations(conn, schema=schema)

        # schema_migrations should exist and have version 1
        version = await conn.fetchval(
            f"SELECT MAX(version) FROM {schema}.schema_migrations"
        )
        assert version == 1

        # Tables should exist
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = $1",
            schema,
        )
        table_names = {r["table_name"] for r in tables}
        assert "wf_handlers" in table_names
        assert "wf_events" in table_names
        assert "workflow_state" in table_names
        assert "schema_migrations" in table_names
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


@requires_postgres
async def test_run_migrations_idempotent() -> None:
    assert POSTGRES_DSN is not None
    conn = await asyncpg.connect(POSTGRES_DSN)
    schema = "test_migrate_idempotent"
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await run_migrations(conn, schema=schema)
        await run_migrations(conn, schema=schema)

        version = await conn.fetchval(
            f"SELECT MAX(version) FROM {schema}.schema_migrations"
        )
        assert version == 1

        # Should have exactly one row in schema_migrations
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {schema}.schema_migrations")
        assert count == 1
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


@requires_postgres
async def test_run_migrations_no_schema() -> None:
    assert POSTGRES_DSN is not None
    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        # Clean up from any previous run
        await conn.execute("DROP TABLE IF EXISTS schema_migrations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_handlers CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_events CASCADE")
        await conn.execute("DROP TABLE IF EXISTS workflow_state CASCADE")

        await run_migrations(conn, schema=None)

        version = await conn.fetchval("SELECT MAX(version) FROM schema_migrations")
        assert version == 1
    finally:
        await conn.execute("DROP TABLE IF EXISTS schema_migrations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_handlers CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_events CASCADE")
        await conn.execute("DROP TABLE IF EXISTS workflow_state CASCADE")
        await conn.close()
