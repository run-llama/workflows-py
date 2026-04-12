# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import hashlib

import pytest
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.retry_policy import (
    ComposableRetryPolicy,
    ConstantDelayRetryPolicy,
    ExponentialBackoffRetryPolicy,
    RetryPolicy,
    retry_if_exception_message,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_chain,
    wait_exponential,
    wait_exponential_jitter,
    wait_fixed,
    wait_random,
)
from workflows.testing import WorkflowTestRunner
from workflows.workflow import Workflow

# ---------------------------------------------------------------------------
# Retry conditions
# ---------------------------------------------------------------------------


def test_retry_if_exception_type_matches() -> None:
    cond = retry_if_exception_type(ValueError, TypeError)
    assert cond(ValueError("bad")) is True
    assert cond(TypeError("bad")) is True
    assert cond(RuntimeError("bad")) is False


def test_retry_if_exception_type_subclass() -> None:
    cond = retry_if_exception_type(OSError)
    assert cond(ConnectionError("refused")) is True


def test_retry_if_not_exception_type() -> None:
    cond = retry_if_not_exception_type(ValueError, KeyError)
    assert cond(ValueError("bad")) is False
    assert cond(KeyError("bad")) is False
    assert cond(RuntimeError("bad")) is True


def test_retry_if_exception_message_match() -> None:
    cond = retry_if_exception_message(match="rate limit|throttl")
    assert cond(Exception("rate limit exceeded")) is True
    assert cond(Exception("request throttled")) is True
    assert cond(Exception("invalid input")) is False


def test_retry_if_exception_message_regex() -> None:
    cond = retry_if_exception_message(match=r"code=\d{3}")
    assert cond(Exception("code=429")) is True
    assert cond(Exception("code=abc")) is False


# ---------------------------------------------------------------------------
# Wait strategies
# ---------------------------------------------------------------------------


def test_wait_fixed() -> None:
    w = wait_fixed(3.5)
    assert w(0) == 3.5
    assert w(5) == 3.5


def test_wait_exponential() -> None:
    w = wait_exponential(initial=1.0, multiplier=2.0, max=100.0)
    assert w(0) == 1.0  # 1 * 2^0
    assert w(1) == 2.0  # 1 * 2^1
    assert w(2) == 4.0  # 1 * 2^2
    assert w(3) == 8.0  # 1 * 2^3


def test_wait_exponential_cap() -> None:
    w = wait_exponential(initial=1.0, multiplier=10.0, max=50.0)
    assert w(2) == 50.0  # 1 * 10^2 = 100 → capped


def test_wait_random_range() -> None:
    w = wait_random(min=1.0, max=5.0)
    for _ in range(20):
        val = w(0)
        assert 1.0 <= val <= 5.0


def test_wait_random_deterministic_with_seed() -> None:
    w = wait_random(min=0.0, max=10.0)
    assert w(0, seed=42) == w(0, seed=42)


def test_wait_exponential_jitter_base() -> None:
    w = wait_exponential_jitter(initial=1.0, exp_base=2.0, max=100.0, jitter=0.0)
    assert w(0) == 1.0
    assert w(1) == 2.0
    assert w(2) == 4.0


def test_wait_exponential_jitter_adds_jitter() -> None:
    w = wait_exponential_jitter(initial=1.0, exp_base=2.0, max=100.0, jitter=1.0)
    for attempt in range(5):
        base = min(1.0 * 2.0**attempt, 100.0)
        val = w(attempt, seed=attempt)
        assert base <= val <= base + 1.0


def test_wait_exponential_jitter_deterministic() -> None:
    w = wait_exponential_jitter(initial=1.0, exp_base=2.0, max=60.0, jitter=1.0)
    assert w(3, seed=99) == w(3, seed=99)


def test_wait_chain_sequence() -> None:
    w = wait_chain(wait_fixed(1), wait_fixed(2), wait_fixed(5))
    assert w(0) == 1.0
    assert w(1) == 2.0
    assert w(2) == 5.0


def test_wait_chain_repeats_last() -> None:
    w = wait_chain(wait_fixed(1), wait_fixed(10))
    assert w(0) == 1.0
    assert w(1) == 10.0
    assert w(2) == 10.0
    assert w(99) == 10.0


def test_wait_chain_requires_strategies() -> None:
    with pytest.raises(ValueError, match="at least one"):
        wait_chain()


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------


def test_stop_after_attempt() -> None:
    s = stop_after_attempt(3)
    assert s(2, 0.0) is False
    assert s(3, 0.0) is True
    assert s(4, 0.0) is True


def test_stop_after_delay() -> None:
    s = stop_after_delay(10.0)
    assert s(1, 9.9) is False
    assert s(1, 10.0) is True
    assert s(1, 15.0) is True


# ---------------------------------------------------------------------------
# Composable RetryPolicy
# ---------------------------------------------------------------------------


def test_retry_policy_defaults() -> None:
    p = ComposableRetryPolicy()
    err = Exception("fail")
    assert p.next(0.0, 0, err) == 5.0
    assert p.next(0.0, 1, err) == 5.0
    assert p.next(0.0, 2, err) == 5.0
    assert p.next(0.0, 3, err) is None  # stop_after_attempt(3)


def test_retry_policy_with_retry_condition() -> None:
    p = ComposableRetryPolicy(
        retry=retry_if_exception_type(ValueError),
        stop=stop_after_attempt(5),
    )
    assert p.next(0.0, 0, ValueError("bad")) == 5.0
    assert p.next(0.0, 0, RuntimeError("bad")) is None  # not retryable


def test_retry_policy_with_wait_strategy() -> None:
    p = ComposableRetryPolicy(
        wait=wait_exponential(initial=1, multiplier=2, max=100),
        stop=stop_after_attempt(5),
    )
    assert p.next(0.0, 0, Exception()) == 1.0
    assert p.next(0.0, 1, Exception()) == 2.0
    assert p.next(0.0, 2, Exception()) == 4.0


def test_retry_policy_stop_after_delay() -> None:
    p = ComposableRetryPolicy(
        wait=wait_fixed(1),
        stop=stop_after_delay(10),
    )
    assert p.next(9.0, 100, Exception()) == 1.0
    assert p.next(10.0, 100, Exception()) is None


def test_retry_policy_seed_forwarded() -> None:
    p = ComposableRetryPolicy(
        wait=wait_random(min=0, max=10),
        stop=stop_after_attempt(5),
    )
    a = p.next(0.0, 0, Exception(), seed=42)
    b = p.next(0.0, 0, Exception(), seed=42)
    assert a == b


def test_retry_policy_retry_none_retries_all() -> None:
    p = ComposableRetryPolicy(retry=None, stop=stop_after_attempt(2))
    assert p.next(0.0, 0, ValueError("a")) == 5.0
    assert p.next(0.0, 0, RuntimeError("b")) == 5.0


def test_retry_policy_all_three_composed() -> None:
    p = ComposableRetryPolicy(
        retry=retry_if_exception_type(ConnectionError),
        wait=wait_exponential(initial=0.5, multiplier=3, max=50),
        stop=stop_after_attempt(3),
    )
    err = ConnectionError("refused")
    assert p.next(0.0, 0, err) == 0.5
    assert p.next(0.0, 1, err) == 1.5
    assert p.next(0.0, 2, err) == 4.5
    assert p.next(0.0, 3, err) is None  # stopped
    assert p.next(0.0, 0, ValueError("bad")) is None  # not retryable


def test_retry_policy_protocol_structural_match() -> None:
    class CustomPolicy:
        def next(
            self,
            elapsed_time: float,
            attempts: int,
            error: Exception,
            *,
            seed: int | None = None,
        ) -> float | None:
            return None

    assert isinstance(CustomPolicy(), RetryPolicy)
    assert isinstance(ComposableRetryPolicy(), RetryPolicy)
    assert isinstance(ConstantDelayRetryPolicy(), RetryPolicy)
    assert isinstance(ExponentialBackoffRetryPolicy(), RetryPolicy)


# ---------------------------------------------------------------------------
# Legacy policies — existing tests preserved
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# E2E — legacy policies still work through the full workflow
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# E2E — new composable RetryPolicy through the full workflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_e2e_composable_policy() -> None:
    """ComposableRetryPolicy with retry condition filters exceptions correctly in a live workflow."""

    class CountEvent(Event):
        pass

    class DummyWorkflow(Workflow):
        @step(
            retry_policy=ComposableRetryPolicy(
                retry=retry_if_exception_type(ValueError),
                wait=wait_fixed(0.1),
                stop=stop_after_attempt(4),
            )
        )
        async def flaky_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            count = await ctx.store.get("counter", default=0)
            ctx.send_event(CountEvent())
            if count < 3:
                raise ValueError("transient failure")
            return StopEvent(result="recovered")

        @step
        async def counter(self, ctx: Context, ev: CountEvent) -> None:
            count = await ctx.store.get("counter", default=0)
            await ctx.store.set("counter", count + 1)

    res = await WorkflowTestRunner(DummyWorkflow(disable_validation=True)).run()
    assert res.result == "recovered"


@pytest.mark.asyncio
async def test_retry_e2e_composable_policy_non_retryable() -> None:
    """ComposableRetryPolicy does not retry non-matching exception types."""

    class DummyWorkflow(Workflow):
        @step(
            retry_policy=ComposableRetryPolicy(
                retry=retry_if_exception_type(ConnectionError),
                wait=wait_fixed(0.1),
                stop=stop_after_attempt(5),
            )
        )
        async def always_fails(self, ev: StartEvent) -> StopEvent:
            raise ValueError("permanent failure")

    with pytest.raises(ValueError, match="permanent failure"):
        await WorkflowTestRunner(DummyWorkflow(disable_validation=True)).run()
