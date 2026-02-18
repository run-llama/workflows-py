# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RetryPolicy(Protocol):
    """
    Policy interface to control step retries after failures.

    Implementations decide whether to retry and how long to wait before the next
    attempt based on elapsed time, number of attempts, and the last error.

    See Also:
        - [ConstantDelayRetryPolicy][workflows.retry_policy.ConstantDelayRetryPolicy]
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
