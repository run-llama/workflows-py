# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio

import asyncpg
import pytest
from llama_agents.dbos._store import POSTGRES_MIGRATION_SOURCE
from llama_agents.server._store import (
    POSTGRES_MIGRATION_SOURCE as SERVER_POSTGRES_MIGRATION_SOURCE,
)
from llama_agents.server._store.migration_utils import (
    iter_migration_files,
    parse_target_version,
)
from llama_agents.server._store.postgres.migrate import run_migrations

# ── Unit tests (no DB) ──────────────────────────────────────────────


_PG_MIGRATIONS_PKG = SERVER_POSTGRES_MIGRATION_SOURCE[1]


def test_parse_target_version_valid() -> None:
    assert parse_target_version("-- migration: 1\nCREATE TABLE ...") == 1
    assert parse_target_version("-- migration: 42\n") == 42


def test_parse_target_version_missing() -> None:
    assert parse_target_version("CREATE TABLE ...") is None
    assert parse_target_version("") is None


def test_iter_migration_files_returns_sorted_sql() -> None:
    files = iter_migration_files(_PG_MIGRATIONS_PKG)
    assert len(files) >= 1
    assert all(f.name.endswith(".sql") for f in files)
    names = [f.name for f in files]
    assert names == sorted(names)


def test_first_migration_has_version_1() -> None:
    files = iter_migration_files(_PG_MIGRATIONS_PKG)
    sql = files[0].read_text()
    assert parse_target_version(sql) == 1


# ── Integration tests (require Docker) ──────────────────────────────

EXPECTED_VERSION = len(iter_migration_files(_PG_MIGRATIONS_PKG))


@pytest.mark.docker
async def test_run_migrations_fresh_db(postgres_dsn: str) -> None:
    conn = await asyncpg.connect(postgres_dsn)
    schema = "test_migrate_fresh"
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await run_migrations(conn, schema=schema)

        version = await conn.fetchval(
            f"SELECT MAX(version) FROM {schema}.schema_migrations WHERE package = 'server'"
        )
        assert version == EXPECTED_VERSION

        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = $1",
            schema,
        )
        table_names = {r["table_name"] for r in tables}
        assert "wf_handlers" in table_names
        assert "wf_events" in table_names
        assert "wf_ticks" in table_names
        assert "workflow_state" in table_names
        assert "schema_migrations" in table_names
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


@pytest.mark.docker
async def test_run_migrations_idempotent(postgres_dsn: str) -> None:
    conn = await asyncpg.connect(postgres_dsn)
    schema = "test_migrate_idempotent"
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await run_migrations(conn, schema=schema)
        await run_migrations(conn, schema=schema)

        version = await conn.fetchval(
            f"SELECT MAX(version) FROM {schema}.schema_migrations WHERE package = 'server'"
        )
        assert version == EXPECTED_VERSION

        count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {schema}.schema_migrations WHERE package = 'server'"
        )
        assert count == EXPECTED_VERSION
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


@pytest.mark.docker
async def test_run_migrations_no_schema(postgres_dsn: str) -> None:
    conn = await asyncpg.connect(postgres_dsn)
    try:
        await conn.execute("DROP TABLE IF EXISTS schema_migrations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_handlers CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_events CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_ticks CASCADE")
        await conn.execute("DROP TABLE IF EXISTS workflow_state CASCADE")

        await run_migrations(conn, schema=None)

        version = await conn.fetchval(
            "SELECT MAX(version) FROM schema_migrations WHERE package = 'server'"
        )
        assert version == EXPECTED_VERSION
    finally:
        await conn.execute("DROP TABLE IF EXISTS schema_migrations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_handlers CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_events CASCADE")
        await conn.execute("DROP TABLE IF EXISTS wf_ticks CASCADE")
        await conn.execute("DROP TABLE IF EXISTS workflow_state CASCADE")
        await conn.close()


@pytest.mark.docker
@pytest.mark.timeout(30)
async def test_concurrent_migrations_with_advisory_lock(
    postgres_dsn: str,
) -> None:
    """Run migrations from N concurrent connections — only one should win the
    race; the rest should observe the lock and skip gracefully."""
    schema = "test_concurrent_mig"
    concurrency = 8

    conn = await asyncpg.connect(postgres_dsn)
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        await conn.close()

    async def migrate_once() -> None:
        c = await asyncpg.connect(postgres_dsn)
        try:
            await run_migrations(c, schema=schema)
        finally:
            await c.close()

    results = await asyncio.gather(
        *[migrate_once() for _ in range(concurrency)],
        return_exceptions=True,
    )

    for i, result in enumerate(results):
        assert not isinstance(result, Exception), f"Migration task {i} failed: {result}"

    conn = await asyncpg.connect(postgres_dsn)
    try:
        count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {schema}.schema_migrations WHERE package = 'server'"
        )
        assert count == EXPECTED_VERSION, (
            f"Expected {EXPECTED_VERSION} migration rows, got {count}"
        )

        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = $1",
            schema,
        )
        table_names = {r["table_name"] for r in tables}
        assert "wf_handlers" in table_names
        assert "wf_events" in table_names
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


@pytest.mark.docker
async def test_per_package_migrations(postgres_dsn: str) -> None:
    """Test running migrations with both server and DBOS sources."""
    conn = await asyncpg.connect(postgres_dsn)
    schema = "test_multi_pkg"
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        sources = [
            SERVER_POSTGRES_MIGRATION_SOURCE,
            POSTGRES_MIGRATION_SOURCE,
        ]
        await run_migrations(conn, schema=schema, sources=sources)

        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = $1",
            schema,
        )
        table_names = {r["table_name"] for r in tables}
        # Server tables
        assert "wf_handlers" in table_names
        assert "wf_events" in table_names
        assert "wf_ticks" in table_names
        assert "workflow_state" in table_names
        # DBOS tables
        assert "workflow_journal" in table_names
        assert "run_lifecycle" in table_names

        # Both packages tracked
        server_ver = await conn.fetchval(
            f"SELECT MAX(version) FROM {schema}.schema_migrations WHERE package = 'server'"
        )
        dbos_ver = await conn.fetchval(
            f"SELECT MAX(version) FROM {schema}.schema_migrations WHERE package = 'dbos'"
        )
        assert server_ver >= 1
        assert dbos_ver >= 1
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()
