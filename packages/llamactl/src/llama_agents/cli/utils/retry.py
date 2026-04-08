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
    """Return True for network-level httpx errors that are safe to retry
    for idempotent operations.

    - Retries on httpx.RequestError (connection errors, timeouts, etc.)
    - Never retries httpx.HTTPStatusError (4xx/5xx responses from the server)
    """
    return isinstance(exc, httpx.RequestError) and not isinstance(
        exc, httpx.HTTPStatusError
    )


def _is_connect_phase_error(exc: BaseException) -> bool:
    """Return True for httpx errors that are guaranteed to have occurred
    before the request reached the server — safe to retry even for
    non-idempotent operations.

    Excludes read/write errors and RemoteProtocolError, any of which may
    have happened after the server accepted (and possibly processed) the
    request.
    """
    return isinstance(
        exc,
        (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout),
    )


async def run_with_network_retries(
    operation: Callable[[], Awaitable[_T]],
    *,
    max_attempts: int = 3,
    idempotent: bool = True,
) -> _T:
    """Run an async operation with standard network retry semantics.

    Retries transient httpx network errors with exponential backoff, but does not retry
    HTTP status errors. After the final attempt, the last exception is re-raised.

    When ``idempotent`` is False, retries are restricted to connect-phase
    errors (ConnectError, ConnectTimeout, PoolTimeout) — i.e. failures where
    the request is guaranteed to have never reached the server. This lets
    callers stay resilient to initial-connectivity blips without risking
    duplicate server-side effects from a read-timeout retry.
    """
    classifier = _is_transient_httpx_error if idempotent else _is_connect_phase_error
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(classifier),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1),
        reraise=True,
    ):
        with attempt:
            return await operation()

    # This line should be unreachable because AsyncRetrying either yields an
    # attempt or re-raises the last exception.
    raise RuntimeError("run_with_network_retries reached an unexpected state")
