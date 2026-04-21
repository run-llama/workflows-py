# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import tracemalloc
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest
from llama_agents.server._store.abstract_workflow_store import (
    StoredTick,
)
from workflows.events import Event
from workflows.runtime.control_loop import rebuild_state_from_ticks_stream
from workflows.runtime.types.internal_state import BrokerConfig, BrokerState
from workflows.runtime.types.ticks import TickPublishEvent, WorkflowTickAdapter


class _BigEvent(Event):
    payload: str


class _FakeStreamingStore:
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
        # Unique per-tick payload defeats string interning so each tick
        # contributes its own allocation.
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
    payload_size = 200_000  # ~200 KB per tick
    total_ticks = 200
    batch_size = 10

    store = _FakeStreamingStore(count=total_ticks, payload_size=payload_size)

    # Warm up caches (Pydantic TypeAdapter, class tables, etc.) before measuring.
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

    bound = batch_size * payload_size * 5
    assert peak < bound, (
        f"Peak memory {peak} bytes exceeded bound {bound} bytes "
        f"(total_ticks={total_ticks}, batch_size={batch_size}, "
        f"payload_size={payload_size})."
    )


@pytest.mark.asyncio
async def test_stream_replay_memory_bound_detects_full_materialization() -> None:
    # Sanity: the bound in the positive test must actually fire on full
    # materialization, otherwise the assertion is toothless.
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
        f"Expected full materialization to blow the bound ({peak} < {bound})."
    )
