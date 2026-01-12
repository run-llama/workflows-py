# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
"""Tests for KeyedLock utility."""

from __future__ import annotations

import asyncio

import pytest
from workflows.server.keyed_lock import KeyedLock


@pytest.fixture
def locks() -> KeyedLock:
    return KeyedLock()


async def test_basic_locking(locks: KeyedLock) -> None:
    """Test that basic lock acquisition and release works."""
    async with locks("key1"):
        pass
    # Lock should be reusable after release
    async with locks("key1"):
        pass


async def test_mutual_exclusion(locks: KeyedLock) -> None:
    """Test that the lock provides mutual exclusion."""
    order: list[str] = []

    async def task(name: str, delay: float) -> None:
        async with locks("shared"):
            order.append(f"{name}_enter")
            await asyncio.sleep(delay)
            order.append(f"{name}_exit")

    t1 = asyncio.create_task(task("first", 0.01))
    await asyncio.sleep(0.001)  # Let first task acquire lock
    t2 = asyncio.create_task(task("second", 0.01))

    await asyncio.gather(t1, t2)

    assert order == ["first_enter", "first_exit", "second_enter", "second_exit"]


async def test_exception_in_critical_section(locks: KeyedLock) -> None:
    """Test that lock is released even if exception occurs."""
    with pytest.raises(ValueError, match="test error"):
        async with locks("key"):
            raise ValueError("test error")

    # Lock should still be usable after exception
    async with locks("key"):
        pass


async def test_cancellation_cleanup(locks: KeyedLock) -> None:
    """Test that lock is released on task cancellation."""
    started = asyncio.Event()

    async def task() -> None:
        async with locks("key"):
            started.set()
            await asyncio.sleep(10)  # Long sleep to be cancelled

    t = asyncio.create_task(task())
    await started.wait()

    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    # Lock should be usable after cancellation
    async with locks("key"):
        pass


async def test_waiter_cancelled_while_waiting(locks: KeyedLock) -> None:
    """Test that cancellation while waiting cleans up properly."""
    holder_started = asyncio.Event()
    waiter_started = asyncio.Event()

    async def holder() -> None:
        async with locks("key"):
            holder_started.set()
            await asyncio.sleep(0.1)

    async def waiter() -> None:
        await holder_started.wait()
        waiter_started.set()
        async with locks("key"):
            pass  # Should never get here

    t1 = asyncio.create_task(holder())
    t2 = asyncio.create_task(waiter())

    await waiter_started.wait()
    await asyncio.sleep(0.01)  # Let waiter register

    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2

    await t1
    # Lock should be usable again after cancellation
    async with locks("key"):
        pass


async def test_parallel_access_mutual_exclusion_with_race_detection(
    locks: KeyedLock,
) -> None:
    """Test mutual exclusion using race condition detection."""
    shared_value = 0
    iterations = 50

    async def increment_task() -> None:
        nonlocal shared_value
        async with locks("key"):
            current = shared_value
            await asyncio.sleep(0)  # Yield to event loop
            shared_value = current + 1

    tasks = [asyncio.create_task(increment_task()) for _ in range(iterations)]
    await asyncio.gather(*tasks)

    assert shared_value == iterations
