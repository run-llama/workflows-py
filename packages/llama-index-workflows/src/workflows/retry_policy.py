# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable


@runtime_checkable
class RetryPolicy(Protocol):
    """
    Policy interface to control step retries after failures.

    Implementations decide whether to retry and how long to wait before the next
    attempt based on elapsed time, number of attempts, and the last error.

    See Also:
        - [ConstantDelayRetryPolicy][workflows.retry_policy.ConstantDelayRetryPolicy]
        - [ExponentialBackoffRetryPolicy][workflows.retry_policy.ExponentialBackoffRetryPolicy]
        - [step][workflows.decorators.step]
    """

    def next(
        self, elapsed_time: float, attempts: int, error: Exception
    ) -> float | None:
        """
        Decide if another retry should occur and the delay before it.

        Args:
            elapsed_time (float): Seconds since the first failure.
            attempts (int): Number of attempts made so far.
            error (Exception): The last exception encountered.

        Returns:
            float | None: Seconds to wait before retrying, or `None` to stop.
        """


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
            maximum_attempts (int): Maximum consecutive attempts. Defaults to 3.
            delay (float): Seconds to wait between attempts. Defaults to 5.
        """
        self.maximum_attempts = maximum_attempts
        self.delay = delay

    def next(
        self, elapsed_time: float, attempts: int, error: Exception
    ) -> float | None:
        """Return the fixed delay while attempts remain; otherwise `None`."""
        if attempts >= self.maximum_attempts:
            return None

        return self.delay


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
            maximum_attempts (int): Maximum consecutive attempts. Defaults to 5.
            initial_delay (float): Delay in seconds before the first retry.
                Defaults to 1.0.
            multiplier (float): Factor applied to the delay after each attempt.
                Defaults to 2.0.
            max_delay (float): Upper bound on the computed delay in seconds.
                Defaults to 60.0.
            jitter (bool): When ``True``, randomise the delay uniformly between
                0 and the computed value. Defaults to ``True``.
        """
        self.maximum_attempts = maximum_attempts
        self.initial_delay = initial_delay
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.jitter = jitter

    def next(
        self, elapsed_time: float, attempts: int, error: Exception
    ) -> float | None:
        """Return an exponentially growing delay while attempts remain; otherwise ``None``."""
        if attempts >= self.maximum_attempts:
            return None

        delay = min(self.initial_delay * self.multiplier**attempts, self.max_delay)
        if self.jitter:
            delay = random.uniform(0, delay)
        return delay
