# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Regression test: streaming tick replay must not materialize full history.

If this test fails, the streaming replay path has regressed back to an O(N)
allocation pattern, which would reintroduce the idle-release OOM this work
targets. The bound is deliberately generous (a small multiple of
batch_size * payload_size) so it triggers only on true O(N) regressions,
not allocator noise.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest
from llama_agents.server._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    StoredTick,
)
from workflows.events import Event
from workflows.runtime.control_loop import rebuild_state_from_ticks_stream
from workflows.runtime.types.internal_state import BrokerConfig, BrokerState
from workflows.runtime.types.ticks import TickPublishEvent, WorkflowTickAdapter


class _BigEvent(Event):
    payload: str


class _FakeStreamingStore:
    """Minimal fake that generates synthetic ticks lazily.

    Not a full AbstractWorkflowStore - only implements stream_ticks/query_ticks,
    which is all the replay path needs. Generating ticks on demand instead of
    holding them in a list lets the test isolate whether the *consumer* holds
    the full history.
    """

    def __init__(self, count: int, payload_size: int) -> None:
        self._count = count
        self._payload_size = payload_size

    async def query_ticks(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[StoredTick]:
        start = 0 if after_sequence is None else after_sequence + 1
        end = self._count if limit is None else min(self._count, start + limit)
        return [self._make_stored_tick(i) for i in range(start, end)]

    async def stream_ticks(
        self, run_id: str, *, batch_size: int = 100
    ) -> AsyncIterator[StoredTick]:
        cursor: int | None = None
        while True:
            batch = await self.query_ticks(
                run_id, after_sequence=cursor, limit=batch_size
            )
            for tick in batch:
                yield tick
                cursor = tick.sequence
            if len(batch) < batch_size:
                return

    def _make_stored_tick(self, seq: int) -> StoredTick:
        # Unique per-tick payload so Python string interning doesn't collapse
        # memory cost across ticks — otherwise the test measures nothing.
        payload = f"{seq:08d}" + ("x" * (self._payload_size - 8))
        tick = TickPublishEvent(event=_BigEvent(payload=payload))
        return StoredTick(
            run_id="r",
            sequence=seq,
            timestamp=datetime.now(timezone.utc),
            tick_data=tick.model_dump(),
        )


def _empty_state() -> BrokerState:
    return BrokerState(
        is_running=True, config=BrokerConfig(steps={}, timeout=None), workers={}
    )


@pytest.mark.asyncio
async def test_stream_replay_peak_memory_bounded_by_batch_size() -> None:
    """Peak memory during replay must scale with batch_size, not tick count."""
    payload_size = 200_000  # ~200 KB per tick
    total_ticks = 200
    batch_size = 10

    store = _FakeStreamingStore(count=total_ticks, payload_size=payload_size)

    # Warm up caches (Pydantic TypeAdapter, class tables, etc.)
    async for stored in store.stream_ticks("r", batch_size=batch_size):
        WorkflowTickAdapter.validate_python(stored.tick_data)
        break

    async def _stream() -> AsyncIterator:
        async for stored in store.stream_ticks("r", batch_size=batch_size):
            yield WorkflowTickAdapter.validate_python(stored.tick_data)

    tracemalloc.start()
    try:
        await rebuild_state_from_ticks_stream(_empty_state(), _stream())
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Generous bound: if we accidentally materialize all N ticks, peak will be
    # total_ticks * payload_size (~40 MB). batch_size * payload_size * 5 is
    # ~10 MB — a comfortable gap.
    bound = batch_size * payload_size * 5
    assert peak < bound, (
        f"Peak memory {peak} bytes exceeded bound {bound} bytes "
        f"(total_ticks={total_ticks}, batch_size={batch_size}, "
        f"payload_size={payload_size}). Streaming replay appears to be "
        f"materializing the full history."
    )


@pytest.mark.asyncio
async def test_stream_replay_memory_bound_detects_full_materialization() -> None:
    """Sanity check: the bound would fail if ticks were materialized upfront.

    This guards against a future refactor that silently converts the stream
    into a list — the peak assertion in the positive test would lose its
    teeth without this companion check.
    """
    payload_size = 200_000
    total_ticks = 200
    batch_size = 10

    store = _FakeStreamingStore(count=total_ticks, payload_size=payload_size)

    async def _all_then_stream() -> AsyncIterator:
        materialized = [
            WorkflowTickAdapter.validate_python(s.tick_data)
            async for s in store.stream_ticks("r", batch_size=batch_size)
        ]
        for t in materialized:
            yield t

    tracemalloc.start()
    try:
        await rebuild_state_from_ticks_stream(_empty_state(), _all_then_stream())
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    bound = batch_size * payload_size * 5
    assert peak >= bound, (
        f"Expected full materialization to blow the bound ({peak} < {bound}). "
        f"If this assertion fires, the memory bound is too loose — the "
        f"streaming test is not actually exercising the bound."
    )

