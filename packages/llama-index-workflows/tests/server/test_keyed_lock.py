# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
"""Tests for KeyedLock utility."""

from __future__ import annotations

import asyncio
import sys

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


async def test_parallel_access_mutual_exclusion_with_race_detection(
    locks: KeyedLock,
) -> None:
    """Test that parallel access is mutually exclusive using race condition detection.

    This test would fail if two tasks could execute their critical sections
    in parallel, as the non-atomic increment/decrement would produce wrong results.
    """
    # Shared state that would be corrupted by parallel access
    shared_value = 0
    iterations = 50

    async def increment_task() -> None:
        nonlocal shared_value
        async with locks("key"):
            # Read-modify-write pattern that would fail under parallel access
            current = shared_value
            await asyncio.sleep(0)  # Yield to event loop
            shared_value = current + 1

    # Launch all tasks simultaneously
    tasks = [asyncio.create_task(increment_task()) for _ in range(iterations)]
    await asyncio.gather(*tasks)

    # If mutual exclusion works, we should have exactly `iterations` increments
    # If parallel access occurred, we'd likely have less due to lost updates
    assert shared_value == iterations


async def test_parallel_access_no_interleaving(locks: KeyedLock) -> None:
    """Test that operations within a lock never interleave.

    Verifies that for any key, the sequence of enter/exit events is always
    properly nested (enter, exit, enter, exit) and never interleaved
    (enter, enter, exit, exit).
    """
    events: list[tuple[str, str]] = []  # (task_name, event_type)
    num_tasks = 20

    async def task(name: str) -> None:
        async with locks("shared"):
            events.append((name, "enter"))
            await asyncio.sleep(0.001)  # Yield to allow interleaving if broken
            events.append((name, "exit"))

    # Start all tasks at roughly the same time
    tasks = [asyncio.create_task(task(f"task_{i}")) for i in range(num_tasks)]
    await asyncio.gather(*tasks)

    # Verify proper nesting: each enter must be immediately followed by exit
    # from the same task (no interleaving)
    assert len(events) == num_tasks * 2
    for i in range(0, len(events), 2):
        enter_event = events[i]
        exit_event = events[i + 1]
        assert enter_event[1] == "enter", f"Expected enter at position {i}"
        assert exit_event[1] == "exit", f"Expected exit at position {i + 1}"
        assert enter_event[0] == exit_event[0], (
            f"Interleaving detected: {enter_event[0]} entered but "
            f"{exit_event[0]} exited at positions {i}, {i + 1}"
        )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="asyncio.Barrier is not available in Python < 3.11",
)
async def test_parallel_access_simultaneous_start(locks: KeyedLock) -> None:
    """Test mutual exclusion when tasks start at exactly the same time.

    Uses a barrier to ensure all tasks begin acquiring the lock simultaneously.
    """
    num_tasks = 10
    barrier = asyncio.Barrier(num_tasks)  # type: ignore[unresolved-attribute]
    active_count = 0
    max_active = 0
    completed_order: list[int] = []

    async def task(task_id: int) -> None:
        nonlocal active_count, max_active
        # Wait for all tasks to be ready
        await barrier.wait()
        # Now all tasks try to acquire the lock at once
        async with locks("key"):
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.001)
            completed_order.append(task_id)
            active_count -= 1

    tasks = [asyncio.create_task(task(i)) for i in range(num_tasks)]
    await asyncio.gather(*tasks)

    # Only one task should ever be active at a time
    assert max_active == 1, f"Max concurrent was {max_active}, expected 1"
    # All tasks should have completed
    assert len(completed_order) == num_tasks
    # Each task should appear exactly once
    assert sorted(completed_order) == list(range(num_tasks))


async def test_parallel_access_stress_test(locks: KeyedLock) -> None:
    """Stress test with high concurrency to detect subtle race conditions."""
    num_tasks = 100
    num_iterations = 10
    results: list[int] = []
    current_holder: int | None = None

    async def task(task_id: int) -> None:
        nonlocal current_holder
        for _ in range(num_iterations):
            async with locks("stress_key"):
                # Check no other task holds the lock
                assert current_holder is None, (
                    f"Task {task_id} entered while {current_holder} was holding"
                )
                current_holder = task_id
                await asyncio.sleep(0)  # Yield to event loop
                # Verify we still hold the lock
                assert current_holder == task_id, (
                    f"Task {task_id} lost lock to {current_holder}"
                )
                results.append(task_id)
                current_holder = None

    tasks = [asyncio.create_task(task(i)) for i in range(num_tasks)]
    await asyncio.gather(*tasks)

    # Should have recorded all iterations from all tasks
    assert len(results) == num_tasks * num_iterations
    # Lock should be cleaned up
    assert "stress_key" not in locks._locks
