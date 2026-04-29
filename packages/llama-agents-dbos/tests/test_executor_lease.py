# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ExecutorLeaseManager."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import asyncpg
import pytest
from llama_agents.dbos.executor_lease import ExecutorLeaseManager
from llama_agents.server._store.postgres.migrate import run_migrations


@pytest.fixture
async def lease_dsn(postgres_dsn: str) -> AsyncGenerator[str]:
    """Set up a clean schema with migrations for lease tests."""
    conn = await asyncpg.connect(postgres_dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS test_lease CASCADE")
        await run_migrations(
            conn,
            schema="test_lease",
            sources=[("dbos", "llama_agents.dbos._store.postgres.migrations")],
        )
    finally:
        await conn.close()
    yield postgres_dsn


def make_manager(
    dsn: str,
    pool_size: int = 3,
    heartbeat_interval: float = 0.5,
    lease_timeout: float = 5.0,
) -> ExecutorLeaseManager:
    return ExecutorLeaseManager(
        dsn=dsn,
        pool_size=pool_size,
        schema="test_lease",
        heartbeat_interval=heartbeat_interval,
        lease_timeout=lease_timeout,
    )


def test_owns_pool_when_no_factory_provided() -> None:
    mgr = ExecutorLeaseManager(dsn="postgresql://x/y", pool_size=1)
    assert mgr._owns_pool is True
    assert mgr._external_ensure_pool is None


def test_borrows_pool_when_ensure_pool_provided() -> None:
    async def factory() -> Any:
        raise AssertionError("not called in this test")

    mgr = ExecutorLeaseManager(
        dsn="postgresql://x/y",
        pool_size=1,
        ensure_pool=factory,
    )
    assert mgr._owns_pool is False
    assert mgr._external_ensure_pool is factory


@pytest.mark.docker
@pytest.mark.asyncio
async def test_acquire_returns_slot_id(lease_dsn: str) -> None:
    mgr = make_manager(lease_dsn)
    slot = await mgr.acquire()
    try:
        assert slot == "executor-0"
        assert mgr.executor_id == "executor-0"
    finally:
        await mgr.release()


@pytest.mark.docker
@pytest.mark.asyncio
async def test_executor_id_raises_before_acquire(lease_dsn: str) -> None:
    mgr = make_manager(lease_dsn)
    with pytest.raises(RuntimeError, match="Lease not acquired"):
        mgr.executor_id


@pytest.mark.docker
@pytest.mark.asyncio
async def test_acquire_blocks_when_full(lease_dsn: str) -> None:
    managers = [make_manager(lease_dsn, pool_size=2) for _ in range(2)]
    slots = []
    for m in managers:
        slots.append(await m.acquire())

    assert set(slots) == {"executor-0", "executor-1"}

    # Third acquire should time out since pool is full
    blocked = make_manager(lease_dsn, pool_size=2)
    with pytest.raises(TimeoutError):
        await blocked.acquire(timeout=0.3)

    for m in managers:
        await m.release()


@pytest.mark.docker
@pytest.mark.asyncio
async def test_release_frees_slot(lease_dsn: str) -> None:
    m1 = make_manager(lease_dsn, pool_size=1)
    await m1.acquire()
    await m1.release()

    m2 = make_manager(lease_dsn, pool_size=1)
    slot = await m2.acquire()
    assert slot == "executor-0"
    await m2.release()


@pytest.mark.docker
@pytest.mark.asyncio
async def test_stale_heartbeat_allows_reclaim(lease_dsn: str) -> None:
    m1 = make_manager(lease_dsn, pool_size=1, lease_timeout=1.0)
    await m1.acquire()

    # Stop heartbeat and wait for lease to go stale
    assert m1._heartbeat_task is not None
    m1._heartbeat_task.cancel()
    try:
        await m1._heartbeat_task
    except asyncio.CancelledError:
        pass
    m1._heartbeat_task = None

    # Manually set heartbeat_at to the past
    assert m1._pool is not None
    await m1._pool.execute(
        f"UPDATE {m1._table} SET heartbeat_at = NOW() - INTERVAL '10 seconds' "
        f"WHERE slot_id = $1",
        m1._slot_id,
    )

    # Another manager should be able to claim the stale slot
    m2 = make_manager(lease_dsn, pool_size=1, lease_timeout=1.0)
    slot = await m2.acquire(timeout=2.0)
    assert slot == "executor-0"

    await m2.release()
    # Clean up m1's pool without trying to release the slot (already taken)
    if m1._pool is not None:
        await m1._pool.close()


@pytest.mark.docker
@pytest.mark.asyncio
async def test_lost_lease_sets_event(lease_dsn: str) -> None:
    m1 = make_manager(lease_dsn, pool_size=1, heartbeat_interval=0.1)
    await m1.acquire()

    # Simulate another process stealing the lease by changing the holder
    assert m1._pool is not None
    await m1._pool.execute(
        f"UPDATE {m1._table} SET holder = 'stolen' WHERE slot_id = $1",
        m1._slot_id,
    )

    # Wait for heartbeat to detect the loss
    await asyncio.wait_for(m1.lease_lost_event.wait(), timeout=1.0)
    assert m1.lease_lost_event.is_set()

    # Clean up
    if m1._pool is not None:
        await m1._pool.close()


@pytest.mark.docker
@pytest.mark.asyncio
async def test_concurrent_acquires(lease_dsn: str) -> None:
    """Multiple concurrent acquires don't get the same slot."""
    pool_size = 3
    managers = [make_manager(lease_dsn, pool_size=pool_size) for _ in range(pool_size)]

    slots = await asyncio.gather(*(m.acquire() for m in managers))
    assert len(set(slots)) == pool_size

    await asyncio.gather(*(m.release() for m in managers))


@pytest.mark.docker
@pytest.mark.asyncio
async def test_context_manager_releases_on_exit(lease_dsn: str) -> None:
    async with make_manager(lease_dsn, pool_size=1) as mgr:
        assert mgr.executor_id == "executor-0"

    # Slot should be free now
    async with make_manager(lease_dsn, pool_size=1) as mgr2:
        assert mgr2.executor_id == "executor-0"
