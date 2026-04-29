# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
from typing import cast

import asyncpg
import pytest
from llama_agents.server._pool import PoolProvider


class FakePool:
    def __init__(self) -> None:
        self.close_calls = 0
        self.terminate_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    def terminate(self) -> None:
        self.terminate_calls += 1


def pool_factory(pool: FakePool) -> asyncpg.Pool:
    return cast(asyncpg.Pool, pool)


def create_owned_provider(
    monkeypatch: pytest.MonkeyPatch,
    pool: FakePool,
    calls: list[tuple[str, int, int]] | None = None,
) -> PoolProvider:
    async def create_pool(
        dsn: str,
        *,
        min_size: int,
        max_size: int,
    ) -> asyncpg.Pool:
        if calls is not None:
            calls.append((dsn, min_size, max_size))
        return pool_factory(pool)

    monkeypatch.setattr(asyncpg, "create_pool", create_pool)
    return PoolProvider.create("postgresql://example/db", min_size=2, max_size=7)


async def test_create_uses_asyncpg_create_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool()
    calls: list[tuple[str, int, int]] = []

    provider = create_owned_provider(monkeypatch, pool, calls)

    assert await provider.get() is pool
    assert calls == [("postgresql://example/db", 2, 7)]


async def test_owned_provider_closes_cached_pool_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool()
    provider = create_owned_provider(monkeypatch, pool)

    assert await provider.get() is pool
    await provider.close()
    await provider.close()

    assert pool.close_calls == 1
    assert pool.terminate_calls == 0


async def test_borrowed_provider_close_and_terminate_do_not_affect_pool() -> None:
    pool = FakePool()
    provider = PoolProvider.borrowed(
        lambda: asyncio.sleep(0, result=pool_factory(pool))
    )

    assert await provider.get() is pool
    await provider.close()
    provider.terminate()

    assert pool.close_calls == 0
    assert pool.terminate_calls == 0
    assert await provider.get() is pool


async def test_owned_provider_terminates_cached_pool_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool()
    provider = create_owned_provider(monkeypatch, pool)

    assert await provider.get() is pool
    provider.terminate()
    provider.terminate()

    assert pool.close_calls == 0
    assert pool.terminate_calls == 1


async def test_close_and_terminate_before_get_do_not_call_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int]] = []

    closed = create_owned_provider(monkeypatch, FakePool(), calls)
    await closed.close()

    terminated = create_owned_provider(monkeypatch, FakePool(), calls)
    terminated.terminate()

    assert calls == []


async def test_close_after_terminate_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool()
    provider = create_owned_provider(monkeypatch, pool)

    assert await provider.get() is pool
    provider.terminate()
    await provider.close()

    assert pool.close_calls == 0
    assert pool.terminate_calls == 1


async def test_terminate_after_close_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool()
    provider = create_owned_provider(monkeypatch, pool)

    assert await provider.get() is pool
    await provider.close()
    provider.terminate()

    assert pool.close_calls == 1
    assert pool.terminate_calls == 0


async def test_get_after_close_raises() -> None:
    provider = PoolProvider(
        lambda: asyncio.sleep(0, result=pool_factory(FakePool())),
        owns_pool=True,
    )

    await provider.close()

    with pytest.raises(RuntimeError, match="closed"):
        await provider.get()


async def test_get_after_terminate_raises() -> None:
    provider = PoolProvider(
        lambda: asyncio.sleep(0, result=pool_factory(FakePool())),
        owns_pool=True,
    )

    provider.terminate()

    with pytest.raises(RuntimeError, match="terminated"):
        await provider.get()


async def test_factory_exception_is_not_cached() -> None:
    pool = FakePool()
    calls = 0

    async def factory() -> asyncpg.Pool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("failed")
        return pool_factory(pool)

    provider = PoolProvider.borrowed(factory)

    with pytest.raises(ValueError, match="failed"):
        await provider.get()

    assert await provider.get() is pool
    assert calls == 2


async def test_concurrent_get_invokes_factory_once() -> None:
    pool = FakePool()
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def factory() -> asyncpg.Pool:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return pool_factory(pool)

    provider = PoolProvider.borrowed(factory)

    first = asyncio.create_task(provider.get())
    await started.wait()
    second = asyncio.create_task(provider.get())
    release.set()

    assert await asyncio.gather(first, second) == [pool, pool]
    assert calls == 1
