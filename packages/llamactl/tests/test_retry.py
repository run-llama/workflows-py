# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for run_with_network_retries classifier behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from llama_agents.cli.utils.retry import run_with_network_retries


@pytest.fixture(autouse=True)
def _no_sleep():
    """Skip real back-off sleeps so these tests stay fast."""
    with patch("tenacity.nap.time.sleep"), patch("asyncio.sleep", AsyncMock()):
        yield


async def _raising(exc: BaseException, counter: list[int]):
    counter.append(1)
    raise exc


@pytest.mark.asyncio
async def test_idempotent_true_retries_read_timeout() -> None:
    counter: list[int] = []
    with pytest.raises(httpx.ReadTimeout):
        await run_with_network_retries(
            lambda: _raising(httpx.ReadTimeout("slow"), counter),
            max_attempts=3,
        )
    assert len(counter) == 3


@pytest.mark.asyncio
async def test_idempotent_true_retries_connect_error() -> None:
    counter: list[int] = []
    with pytest.raises(httpx.ConnectError):
        await run_with_network_retries(
            lambda: _raising(httpx.ConnectError("refused"), counter),
            max_attempts=3,
        )
    assert len(counter) == 3


@pytest.mark.asyncio
async def test_idempotent_false_does_not_retry_read_timeout() -> None:
    """Read-phase error — request may have reached the server. No retry."""
    counter: list[int] = []
    with pytest.raises(httpx.ReadTimeout):
        await run_with_network_retries(
            lambda: _raising(httpx.ReadTimeout("slow"), counter),
            max_attempts=3,
            idempotent=False,
        )
    assert len(counter) == 1


@pytest.mark.asyncio
async def test_idempotent_false_does_not_retry_remote_protocol_error() -> None:
    """RemoteProtocolError may have happened after the server accepted the
    request — no retry for non-idempotent callers."""
    counter: list[int] = []
    with pytest.raises(httpx.RemoteProtocolError):
        await run_with_network_retries(
            lambda: _raising(httpx.RemoteProtocolError("eof"), counter),
            max_attempts=3,
            idempotent=False,
        )
    assert len(counter) == 1


@pytest.mark.asyncio
async def test_idempotent_false_retries_connect_error() -> None:
    counter: list[int] = []
    with pytest.raises(httpx.ConnectError):
        await run_with_network_retries(
            lambda: _raising(httpx.ConnectError("refused"), counter),
            max_attempts=3,
            idempotent=False,
        )
    assert len(counter) == 3


@pytest.mark.asyncio
async def test_idempotent_false_retries_connect_timeout() -> None:
    counter: list[int] = []
    with pytest.raises(httpx.ConnectTimeout):
        await run_with_network_retries(
            lambda: _raising(httpx.ConnectTimeout("no syn-ack"), counter),
            max_attempts=3,
            idempotent=False,
        )
    assert len(counter) == 3


@pytest.mark.asyncio
async def test_idempotent_false_retries_pool_timeout() -> None:
    counter: list[int] = []
    with pytest.raises(httpx.PoolTimeout):
        await run_with_network_retries(
            lambda: _raising(httpx.PoolTimeout("saturated"), counter),
            max_attempts=3,
            idempotent=False,
        )
    assert len(counter) == 3


@pytest.mark.asyncio
async def test_never_retries_http_status_error() -> None:
    """HTTPStatusError is a server response, not a transient failure."""
    counter: list[int] = []
    resp = httpx.Response(500, request=httpx.Request("POST", "https://x"))
    err = httpx.HTTPStatusError("500", request=resp.request, response=resp)
    with pytest.raises(httpx.HTTPStatusError):
        await run_with_network_retries(
            lambda: _raising(err, counter),
            max_attempts=3,
        )
    assert len(counter) == 1


@pytest.mark.asyncio
async def test_succeeds_on_second_attempt() -> None:
    counter: list[int] = []

    async def op() -> str:
        counter.append(1)
        if len(counter) < 2:
            raise httpx.ConnectError("first try")
        return "ok"

    result = await run_with_network_retries(op, max_attempts=3, idempotent=False)
    assert result == "ok"
    assert len(counter) == 2
