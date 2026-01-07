# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
"""Tests for KeyedLock utility."""

import asyncio

import pytest
from workflows.server.keyed_lock import KeyedLock


@pytest.fixture
def locks() -> KeyedLock:
    return KeyedLock()


async def test_basic_locking(locks: KeyedLock) -> None:
    """Test that basic lock acquisition and release works."""
    async with locks("key1"):
        # We're inside the lock
        pass
    # Lock should be cleaned up
    assert "key1" not in locks._locks


async def test_mutual_exclusion(locks: KeyedLock) -> None:
    """Test that the lock provides mutual exclusion."""
    order: list[str] = []

    async def task(name: str, delay: float) -> None:
        async with locks("shared"):
            order.append(f"{name}_enter")
            await asyncio.sleep(delay)
            order.append(f"{name}_exit")

    # Start two tasks that try to acquire the same lock
    t1 = asyncio.create_task(task("first", 0.01))
    await asyncio.sleep(0.001)  # Let first task acquire lock
    t2 = asyncio.create_task(task("second", 0.01))

    await asyncio.gather(t1, t2)

    # Second task should not enter until first exits
    assert order == ["first_enter", "first_exit", "second_enter", "second_exit"]


async def test_different_keys_independent(locks: KeyedLock) -> None:
    """Test that different keys don't block each other."""
    order: list[str] = []

    async def task(key: str, name: str) -> None:
        async with locks(key):
            order.append(f"{name}_enter")
            await asyncio.sleep(0.01)
            order.append(f"{name}_exit")

    # Start two tasks with different keys
    t1 = asyncio.create_task(task("key1", "first"))
    await asyncio.sleep(0.001)  # Let first task start
    t2 = asyncio.create_task(task("key2", "second"))

    await asyncio.gather(t1, t2)

    # Both should enter before either exits (parallel execution)
    assert order[0] == "first_enter"
    assert order[1] == "second_enter"


async def test_cleanup_after_release(locks: KeyedLock) -> None:
    """Test that locks are cleaned up after release."""
    assert len(locks._locks) == 0
    assert len(locks._refs) == 0

    async with locks("key1"):
        assert "key1" in locks._locks
        assert locks._refs["key1"] == 1

    # After release, lock should be removed
    assert "key1" not in locks._locks
    assert "key1" not in locks._refs


async def test_multiple_waiters_cleanup(locks: KeyedLock) -> None:
    """Test cleanup only happens when last waiter releases."""
    barrier = asyncio.Event()
    second_acquired = asyncio.Event()

    async def holder() -> None:
        async with locks("key"):
            barrier.set()
            await asyncio.sleep(0.02)

    async def waiter() -> None:
        await barrier.wait()
        async with locks("key"):
            second_acquired.set()
            await asyncio.sleep(0.01)

    t1 = asyncio.create_task(holder())
    t2 = asyncio.create_task(waiter())

    # Wait for second to be waiting
    await barrier.wait()
    await asyncio.sleep(0.005)

    # Both should be registered
    assert locks._refs["key"] == 2

    # Wait for first to release
    await t1
    # Lock still exists because second is holding it
    assert "key" in locks._locks
    assert locks._refs["key"] == 1

    await t2
    # Now lock should be cleaned up
    assert "key" not in locks._locks
    assert "key" not in locks._refs


async def test_reuse_after_cleanup(locks: KeyedLock) -> None:
    """Test that a key can be reused after cleanup."""
    async with locks("key"):
        pass

    assert "key" not in locks._locks

    # Should be able to acquire again
    async with locks("key"):
        assert "key" in locks._locks

    assert "key" not in locks._locks


async def test_exception_in_critical_section(locks: KeyedLock) -> None:
    """Test that lock is released even if exception occurs."""
    with pytest.raises(ValueError, match="test error"):
        async with locks("key"):
            raise ValueError("test error")

    # Lock should still be cleaned up
    assert "key" not in locks._locks
    assert "key" not in locks._refs


async def test_cancellation_cleanup(locks: KeyedLock) -> None:
    """Test that lock is released on task cancellation."""
    started = asyncio.Event()

    async def task() -> None:
        async with locks("key"):
            started.set()
            await asyncio.sleep(10)  # Long sleep to be cancelled

    t = asyncio.create_task(task())
    await started.wait()

    assert "key" in locks._locks

    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    # Lock should be cleaned up
    assert "key" not in locks._locks


async def test_many_concurrent_tasks(locks: KeyedLock) -> None:
    """Test with many concurrent tasks on the same key."""
    counter = 0
    max_concurrent = 0

    async def task() -> None:
        nonlocal counter, max_concurrent
        async with locks("key"):
            counter += 1
            max_concurrent = max(max_concurrent, counter)
            await asyncio.sleep(0.001)
            counter -= 1

    tasks = [asyncio.create_task(task()) for _ in range(10)]
    await asyncio.gather(*tasks)

    # Should never have more than one concurrent execution
    assert max_concurrent == 1
    # Lock should be cleaned up
    assert "key" not in locks._locks


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

    # Both should be registered
    assert locks._refs["key"] == 2

    # Cancel the waiter
    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2

    # Waiter should have cleaned up its reference
    assert locks._refs["key"] == 1

    await t1
    # Now fully cleaned up
    assert "key" not in locks._locks
