# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import random
import re
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Protocol — structural type for any retry policy (including custom ones)
# ---------------------------------------------------------------------------


@runtime_checkable
class RetryPolicyProto(Protocol):
    """
    Structural interface for step retry policies.

    Any object with a compatible ``next`` method satisfies this protocol,
    including the built-in :class:`RetryPolicy`, :class:`ConstantDelayRetryPolicy`,
    :class:`ExponentialBackoffRetryPolicy`, and user-defined policies.

    See Also:
        - [RetryPolicy][workflows.retry_policy.RetryPolicy]
        - [step][workflows.decorators.step]
    """

    def next(
        self,
        elapsed_time: float,
        attempts: int,
        error: Exception,
        *,
        seed: int | None = None,
    ) -> float | None:
        """
        Decide if another retry should occur and the delay before it.

        Args:
            elapsed_time: Seconds since the first failure.
            attempts: Number of attempts made so far.
            error: The last exception encountered.
            seed: Optional RNG seed for deterministic jitter (DBOS replay).

        Returns:
            Seconds to wait before retrying, or ``None`` to stop.
        """


# ---------------------------------------------------------------------------
# Structural protocols for the three composable concerns
# ---------------------------------------------------------------------------


class RetryCondition(Protocol):
    """Predicate that decides whether an exception is retryable."""

    def __call__(self, error: Exception) -> bool: ...


class WaitStrategy(Protocol):
    """Compute the delay in seconds before the next retry attempt."""

    def __call__(self, attempts: int, *, seed: int | None = None) -> float: ...


class StopCondition(Protocol):
    """Predicate that decides whether retries should stop."""

    def __call__(self, attempts: int, elapsed_time: float) -> bool: ...


# ---------------------------------------------------------------------------
# Retry conditions — decide *whether* to retry based on the exception
# ---------------------------------------------------------------------------


class retry_if_exception_type:
    """Retry only when the exception is an instance of one of the given types.

    Examples:
        ```python
        retry=retry_if_exception_type(TimeoutError, ConnectionError)
        ```
    """

    def __init__(self, *exception_types: type[Exception]) -> None:
        self.exception_types = exception_types

    def __call__(self, error: Exception) -> bool:
        return isinstance(error, self.exception_types)


class retry_if_not_exception_type:
    """Retry unless the exception is an instance of one of the given types.

    Examples:
        ```python
        retry=retry_if_not_exception_type(ValueError, KeyError)
        ```
    """

    def __init__(self, *exception_types: type[Exception]) -> None:
        self.exception_types = exception_types

    def __call__(self, error: Exception) -> bool:
        return not isinstance(error, self.exception_types)


class retry_if_exception_message:
    """Retry when the exception message matches a regex pattern.

    Examples:
        ```python
        retry=retry_if_exception_message(match="rate limit|throttl")
        ```
    """

    def __init__(self, match: str) -> None:
        self._pattern = re.compile(match)

    def __call__(self, error: Exception) -> bool:
        return bool(self._pattern.search(str(error)))


# ---------------------------------------------------------------------------
# Wait strategies — decide *how long* to wait before the next attempt
# ---------------------------------------------------------------------------


class wait_fixed:
    """Wait a fixed number of seconds between attempts.

    Examples:
        ```python
        wait=wait_fixed(5)
        ```
    """

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def __call__(self, attempts: int, *, seed: int | None = None) -> float:
        return self.delay


class wait_exponential:
    """Wait with exponentially increasing delays, clamped to a maximum.

    Computes ``initial * multiplier ** attempts``, capped at *max*.

    Examples:
        ```python
        wait=wait_exponential(initial=1, multiplier=2, max=60)
        ```
    """

    def __init__(
        self,
        initial: float = 1.0,
        multiplier: float = 2.0,
        max: float = 60.0,
    ) -> None:
        self.initial = initial
        self.multiplier = multiplier
        self.max = max

    def __call__(self, attempts: int, *, seed: int | None = None) -> float:
        return min(self.initial * self.multiplier**attempts, self.max)


class wait_random:
    """Wait a random duration uniformly sampled from ``[min, max]``.

    Examples:
        ```python
        wait=wait_random(min=1, max=5)
        ```
    """

    def __init__(self, min: float = 0.0, max: float = 1.0) -> None:
        self.min = min
        self.max = max

    def __call__(self, attempts: int, *, seed: int | None = None) -> float:
        rng = random.Random(seed) if seed is not None else random
        return rng.uniform(self.min, self.max)


class wait_exponential_jitter:
    """Exponential backoff with additive random jitter.

    Computes ``min(initial * exp_base ** attempts, max) + uniform(0, jitter)``.

    Examples:
        ```python
        wait=wait_exponential_jitter(initial=1, max=60, jitter=1)
        ```
    """

    def __init__(
        self,
        initial: float = 1.0,
        exp_base: float = 2.0,
        max: float = 60.0,
        jitter: float = 1.0,
    ) -> None:
        self.initial = initial
        self.exp_base = exp_base
        self.max = max
        self.jitter = jitter

    def __call__(self, attempts: int, *, seed: int | None = None) -> float:
        rng = random.Random(seed) if seed is not None else random
        base = min(self.initial * self.exp_base**attempts, self.max)
        return base + rng.uniform(0, self.jitter)


class wait_chain:
    """Use a different wait strategy for each attempt in order.

    Once the list is exhausted the last strategy is repeated.

    Examples:
        ```python
        wait=wait_chain(wait_fixed(1), wait_fixed(2), wait_fixed(5))
        ```
    """

    def __init__(self, *strategies: WaitStrategy) -> None:
        if not strategies:
            raise ValueError("wait_chain requires at least one strategy")
        self.strategies = strategies

    def __call__(self, attempts: int, *, seed: int | None = None) -> float:
        idx = min(attempts, len(self.strategies) - 1)
        return self.strategies[idx](attempts, seed=seed)


# ---------------------------------------------------------------------------
# Stop conditions — decide *when* to give up
# ---------------------------------------------------------------------------


class stop_after_attempt:
    """Stop after a fixed number of attempts.

    Examples:
        ```python
        stop=stop_after_attempt(3)
        ```
    """

    def __init__(self, max_attempts: int) -> None:
        self.max_attempts = max_attempts

    def __call__(self, attempts: int, elapsed_time: float) -> bool:
        return attempts >= self.max_attempts


class stop_after_delay:
    """Stop after a maximum elapsed time in seconds.

    Examples:
        ```python
        stop=stop_after_delay(120)
        ```
    """

    def __init__(self, max_delay: float) -> None:
        self.max_delay = max_delay

    def __call__(self, attempts: int, elapsed_time: float) -> bool:
        return elapsed_time >= self.max_delay


# ---------------------------------------------------------------------------
# Composable RetryPolicy
# ---------------------------------------------------------------------------


class RetryPolicy:
    """Composable retry policy built from retry conditions, wait strategies, and stop conditions.

    Decomposes retry behavior into three orthogonal concerns:

    - **retry**: Should we retry this error? (default: retry any exception)
    - **wait**: How long to wait before the next attempt?
    - **stop**: When to give up?

    Examples:
        Retry only transient API errors with exponential backoff:

        ```python
        @step(retry_policy=RetryPolicy(
            retry=retry_if_exception_type(RateLimitError, APITimeoutError),
            wait=wait_exponential(initial=1, multiplier=2, max=60),
            stop=stop_after_attempt(5),
        ))
        async def call_api(self, ev: StartEvent) -> StopEvent:
            ...
        ```

        Stop after 2 minutes of total elapsed time:

        ```python
        @step(retry_policy=RetryPolicy(
            wait=wait_fixed(10),
            stop=stop_after_delay(120),
        ))
        async def flaky_step(self, ev: StartEvent) -> StopEvent:
            ...
        ```
    """

    def __init__(
        self,
        retry: RetryCondition | None = None,
        wait: WaitStrategy = wait_fixed(5),
        stop: StopCondition = stop_after_attempt(3),
    ) -> None:
        self.retry = retry
        self.wait = wait
        self.stop = stop

    def next(
        self,
        elapsed_time: float,
        attempts: int,
        error: Exception,
        *,
        seed: int | None = None,
    ) -> float | None:
        """Return the delay before the next retry, or ``None`` to stop."""
        if self.stop(attempts, elapsed_time):
            return None
        if self.retry is not None and not self.retry(error):
            return None
        return self.wait(attempts, seed=seed)


# ---------------------------------------------------------------------------
# Legacy convenience policies — delegate to RetryPolicy internally
# ---------------------------------------------------------------------------


class ConstantDelayRetryPolicy:
    """Retry at a fixed interval up to a maximum number of attempts.

    Examples:
        ```python
        @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=10))
        async def flaky(self, ev: StartEvent) -> StopEvent:
            ...
        ```
    """

    def __init__(self, maximum_attempts: int = 3, delay: float = 5) -> None:
        """
        Initialize the policy.

        Args:
            maximum_attempts: Maximum consecutive attempts. Defaults to 3.
            delay: Seconds to wait between attempts. Defaults to 5.
        """
        self.maximum_attempts = maximum_attempts
        self.delay = delay
        self._inner = RetryPolicy(
            wait=wait_fixed(delay),
            stop=stop_after_attempt(maximum_attempts),
        )

    def next(
        self,
        elapsed_time: float,
        attempts: int,
        error: Exception,
        *,
        seed: int | None = None,
    ) -> float | None:
        """Return the fixed delay while attempts remain; otherwise `None`."""
        return self._inner.next(elapsed_time, attempts, error, seed=seed)


class ExponentialBackoffRetryPolicy:
    """Retry with exponentially increasing delays, optional jitter, and a cap.

    Each attempt waits ``initial_delay * multiplier ** attempts`` seconds,
    clamped to *max_delay*.  When *jitter* is enabled the actual delay is
    drawn uniformly from ``[0, computed_delay]`` to spread out concurrent
    retries (thundering-herd mitigation).

    Examples:
        ```python
        @step(retry_policy=ExponentialBackoffRetryPolicy(
            initial_delay=1, multiplier=2, max_delay=30, maximum_attempts=5,
        ))
        async def call_api(self, ev: StartEvent) -> StopEvent:
            ...
        ```
    """

    def __init__(
        self,
        maximum_attempts: int = 5,
        initial_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        """
        Initialize the policy.

        Args:
            maximum_attempts: Maximum consecutive attempts. Defaults to 5.
            initial_delay: Delay in seconds before the first retry. Defaults to 1.0.
            multiplier: Factor applied to the delay after each attempt. Defaults to 2.0.
            max_delay: Upper bound on the computed delay in seconds. Defaults to 60.0.
            jitter: When ``True``, randomise the delay uniformly between
                0 and the computed value. Defaults to ``True``.
        """
        self.maximum_attempts = maximum_attempts
        self.initial_delay = initial_delay
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.jitter = jitter
        self._inner = RetryPolicy(
            wait=_ExponentialWithFullJitter(
                initial=initial_delay,
                multiplier=multiplier,
                max=max_delay,
                jitter=jitter,
            ),
            stop=stop_after_attempt(maximum_attempts),
        )

    def next(
        self,
        elapsed_time: float,
        attempts: int,
        error: Exception,
        *,
        seed: int | None = None,
    ) -> float | None:
        """Return an exponentially growing delay while attempts remain; otherwise ``None``."""
        return self._inner.next(elapsed_time, attempts, error, seed=seed)


class _ExponentialWithFullJitter:
    """Internal wait strategy preserving the original ExponentialBackoffRetryPolicy behavior.

    When jitter is enabled, delay is drawn uniformly from ``[0, computed_delay]``
    (full jitter), which differs from ``wait_exponential_jitter`` (additive jitter).
    """

    def __init__(
        self,
        initial: float,
        multiplier: float,
        max: float,
        jitter: bool,
    ) -> None:
        self.initial = initial
        self.multiplier = multiplier
        self.max = max
        self.jitter = jitter

    def __call__(self, attempts: int, *, seed: int | None = None) -> float:
        delay = min(self.initial * self.multiplier**attempts, self.max)
        if self.jitter:
            rng = random.Random(seed) if seed is not None else random
            delay = rng.uniform(0, delay)
        return delay
