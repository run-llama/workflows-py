# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import hashlib

import pytest
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.retry_policy import (
    ConstantDelayRetryPolicy,
    ExponentialBackoffRetryPolicy,
)
from workflows.testing import WorkflowTestRunner
from workflows.workflow import Workflow


@pytest.mark.asyncio
async def test_retry_e2e() -> None:
    class CountEvent(Event):
        """Empty event to signal a step to increment a counter in the Context."""

    class DummyWorkflow(Workflow):
        @step(retry_policy=ConstantDelayRetryPolicy(delay=0.2, maximum_attempts=4))
        async def flaky_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            count = await ctx.store.get("counter", default=0)
            ctx.send_event(CountEvent())
            if count < 3:
                raise ValueError("Something bad happened!")
            return StopEvent(result="All good!")

        @step
        async def counter(self, ctx: Context, ev: CountEvent) -> None:
            count = await ctx.store.get("counter", default=0)
            await ctx.store.set("counter", count + 1)

    res = await WorkflowTestRunner(DummyWorkflow(disable_validation=True)).run()
    assert res.result == "All good!"


def test_ConstantDelayRetryPolicy_init() -> None:
    p = ConstantDelayRetryPolicy()
    assert p.maximum_attempts == 3
    assert p.delay == 5


def test_ConstantDelayRetryPolicy_next() -> None:
    delay = 4.2
    p = ConstantDelayRetryPolicy(maximum_attempts=5, delay=delay)
    assert p.next(elapsed_time=0.0, attempts=4, error=Exception()) == delay
    assert p.next(elapsed_time=0.0, attempts=5, error=Exception()) is None
    # This should never happen but ensure the code is resilient
    assert p.next(elapsed_time=0.0, attempts=999, error=Exception()) is None


# --- ExponentialBackoffRetryPolicy ---


def test_ExponentialBackoffRetryPolicy_init() -> None:
    p = ExponentialBackoffRetryPolicy()
    assert p.maximum_attempts == 5
    assert p.initial_delay == 1.0
    assert p.multiplier == 2.0
    assert p.max_delay == 60.0
    assert p.jitter is True


def test_ExponentialBackoffRetryPolicy_next_basic() -> None:
    p = ExponentialBackoffRetryPolicy(
        initial_delay=1.0, multiplier=2.0, max_delay=100.0, jitter=False
    )
    err = Exception()
    assert p.next(elapsed_time=0.0, attempts=0, error=err) == 1.0  # 1 * 2^0
    assert p.next(elapsed_time=0.0, attempts=1, error=err) == 2.0  # 1 * 2^1
    assert p.next(elapsed_time=0.0, attempts=2, error=err) == 4.0  # 1 * 2^2
    assert p.next(elapsed_time=0.0, attempts=3, error=err) == 8.0  # 1 * 2^3


def test_ExponentialBackoffRetryPolicy_max_delay_cap() -> None:
    p = ExponentialBackoffRetryPolicy(
        initial_delay=1.0, multiplier=10.0, max_delay=50.0, jitter=False
    )
    # 1 * 10^2 = 100 → capped to 50
    assert p.next(elapsed_time=0.0, attempts=2, error=Exception()) == 50.0
    assert p.next(elapsed_time=0.0, attempts=5, error=Exception()) is None


def test_ExponentialBackoffRetryPolicy_gives_up() -> None:
    p = ExponentialBackoffRetryPolicy(maximum_attempts=3, jitter=False)
    err = Exception()
    assert p.next(elapsed_time=0.0, attempts=2, error=err) is not None
    assert p.next(elapsed_time=0.0, attempts=3, error=err) is None
    assert p.next(elapsed_time=0.0, attempts=999, error=err) is None


def test_ExponentialBackoffRetryPolicy_jitter() -> None:
    p = ExponentialBackoffRetryPolicy(
        initial_delay=1.0, multiplier=2.0, max_delay=100.0, jitter=True
    )
    err = Exception()
    for attempt in range(5):
        computed = min(1.0 * 2.0**attempt, 100.0)
        delay = p.next(elapsed_time=0.0, attempts=attempt, error=err, seed=attempt)
        assert delay is not None
        assert 0 <= delay <= computed


def test_ExponentialBackoffRetryPolicy_jitter_deterministic() -> None:
    """Same seed must produce the same delay on every call (DBOS replay determinism)."""
    p = ExponentialBackoffRetryPolicy(
        initial_delay=1.0, multiplier=2.0, max_delay=100.0, jitter=True
    )
    err = Exception()
    for attempt in range(5):
        seed = (
            int(
                hashlib.sha256(f"run-abc:my_step:{attempt + 1}".encode()).hexdigest(),
                16,
            )
            & 0xFFFF_FFFF
        )
        first = p.next(elapsed_time=0.0, attempts=attempt, error=err, seed=seed)
        second = p.next(elapsed_time=0.0, attempts=attempt, error=err, seed=seed)
        assert first == second


def test_ExponentialBackoffRetryPolicy_no_jitter() -> None:
    p = ExponentialBackoffRetryPolicy(
        initial_delay=0.5, multiplier=3.0, max_delay=100.0, jitter=False
    )
    err = Exception()
    assert p.next(elapsed_time=0.0, attempts=0, error=err) == 0.5  # 0.5 * 3^0
    assert p.next(elapsed_time=0.0, attempts=1, error=err) == 1.5  # 0.5 * 3^1
    assert p.next(elapsed_time=0.0, attempts=2, error=err) == 4.5  # 0.5 * 3^2
    assert p.next(elapsed_time=0.0, attempts=3, error=err) == 13.5  # 0.5 * 3^3


@pytest.mark.asyncio
async def test_retry_e2e_exponential() -> None:
    class CountEvent(Event):
        """Event to increment a counter in the Context."""

    class DummyWorkflow(Workflow):
        @step(
            retry_policy=ExponentialBackoffRetryPolicy(
                initial_delay=0.05, multiplier=2.0, maximum_attempts=4, jitter=False
            )
        )
        async def flaky_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            count = await ctx.store.get("counter", default=0)
            ctx.send_event(CountEvent())
            if count < 3:
                raise ValueError("Something bad happened!")
            return StopEvent(result="All good!")

        @step
        async def counter(self, ctx: Context, ev: CountEvent) -> None:
            count = await ctx.store.get("counter", default=0)
            await ctx.store.set("counter", count + 1)

    res = await WorkflowTestRunner(DummyWorkflow(disable_validation=True)).run()
    assert res.result == "All good!"
