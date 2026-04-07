import asyncio
from typing import AsyncGenerator, List

import pytest
from llama_agents.core.iter_utils import (
    debounced_sorted_prefix,
    merge_generators,
)


async def _gen_with_delays(
    values: List[str], delays: List[float]
) -> AsyncGenerator[str, None]:
    for value, delay in zip(values, delays):
        if delay:
            await asyncio.sleep(delay)
        yield value


async def _gen_raises(after_items: int = 0) -> AsyncGenerator[str, None]:
    produced = 0
    while produced < after_items:
        await asyncio.sleep(0)
        produced += 1
        yield f"ok-{produced}"
    raise RuntimeError("boom")


async def _gen_slow_with_close_flag(flag: asyncio.Event) -> AsyncGenerator[str, None]:
    try:
        i = 0
        while True:
            await asyncio.sleep(0.05)
            i += 1
            yield f"slow-{i}"
    finally:
        flag.set()


@pytest.mark.asyncio
async def test_merge_generators_interleaves_in_arrival_order() -> None:
    g1 = _gen_with_delays(["a", "c"], [0.01, 0.04])  # emits at ~0.01 and ~0.05
    g2 = _gen_with_delays(["b", "d"], [0.02, 0.06])  # emits at ~0.02 and ~0.08

    merged: List[str] = []
    async for item in merge_generators(g1, g2):
        merged.append(item)

    assert merged == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_merge_generators_propagates_exception_immediately() -> None:
    g_ok = _gen_with_delays(["x", "y"], [0.0, 0.1])
    g_err = _gen_raises(after_items=1)

    items: List[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for item in merge_generators(g_ok, g_err):
            items.append(item)

    # We should have received only items produced before the error
    assert items[0] in {"x", "ok-1"}
    assert len(items) <= 2


@pytest.mark.asyncio
async def test_merge_generators_stop_on_first_completion_cancels_others() -> None:
    closed_event = asyncio.Event()
    g_slow = _gen_slow_with_close_flag(closed_event)
    g_fast = _gen_with_delays(["done"], [0.01])  # completes quickly

    collected: List[str] = []
    async for item in merge_generators(g_slow, g_fast, stop_on_first_completion=True):
        collected.append(item)

    # Only the fast item should be seen deterministically
    assert collected == ["done"]
    # The slow generator should have been closed/cancelled
    assert closed_event.is_set()


@pytest.mark.asyncio
async def test_debounced_sorted_prefix_sorts_then_passthrough() -> None:
    async def inner() -> AsyncGenerator[int, None]:
        # Initial burst (unsorted order)
        for value in [3, 1, 2]:
            yield value
        # Wait longer than debounce to ensure the first flush occurs
        await asyncio.sleep(0.12)
        # Subsequent items should pass through in arrival order
        for value in [4, 5]:
            await asyncio.sleep(0.01)
            yield value

    output: List[int] = []
    async for item in debounced_sorted_prefix(
        inner(), key=lambda x: x, debounce_seconds=0.05, max_window_seconds=0.1
    ):
        output.append(item)

    # First three should be sorted, rest in order
    assert output == [1, 2, 3, 4, 5]
