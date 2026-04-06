from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

_T = TypeVar("_T")


def _is_transient_httpx_error(exc: BaseException) -> bool:
    """Return True for network-level httpx errors that are safe to retry.

    - Retries on httpx.RequestError (connection errors, timeouts, etc.)
    - Never retries httpx.HTTPStatusError (4xx/5xx responses from the server)
    """
    return isinstance(exc, httpx.RequestError) and not isinstance(
        exc, httpx.HTTPStatusError
    )


async def run_with_network_retries(
    operation: Callable[[], Awaitable[_T]],
    *,
    max_attempts: int = 3,
) -> _T:
    """Run an async operation with standard network retry semantics.

    Retries transient httpx network errors with exponential backoff, but does not retry
    HTTP status errors. After the final attempt, the last exception is re-raised.
    """
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_transient_httpx_error),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1),
        reraise=True,
    ):
        with attempt:
            return await operation()

    # This line should be unreachable because AsyncRetrying either yields an
    # attempt or re-raises the last exception.
    raise RuntimeError("run_with_network_retries reached an unexpected state")
