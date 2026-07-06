# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Timeout and (read-only) retry behavior for AgentDataClient."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
import pytest
from llama_agents.server._store.agent_data_client import AgentDataClient

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, **kwargs: Any) -> AgentDataClient:
    client = AgentDataClient(
        base_url="http://backend",
        api_key="k",
        project_id="p",
        deployment_name="d",
        **kwargs,
    )
    client._shared_client = httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.MockTransport(handler),
    )
    return client


@pytest.fixture
def retry_delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def _instant(delay: float) -> None:
        delays.append(delay)
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)
    return delays


async def test_search_retries_5xx_then_succeeds(
    retry_delays: list[float],
) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"detail": "slow"})
        return httpx.Response(200, json={"items": [{"id": "x"}]})

    items = await _client(handler).search("col")

    assert items == [{"id": "x"}]
    assert calls["n"] == 3
    assert retry_delays == [0.5, 1.0]


async def test_search_retries_transport_error_then_succeeds(
    retry_delays: list[float],
) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"items": []})

    items = await _client(handler).search("col")

    assert items == []
    assert calls["n"] == 2
    assert retry_delays == [0.5]


async def test_search_does_not_retry_4xx() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, json={"detail": "nope"})

    with pytest.raises(httpx.HTTPStatusError):
        await _client(handler).search("col")

    assert calls["n"] == 1


async def test_search_raises_after_exhausting_attempts(
    retry_delays: list[float],
) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"detail": "down"})

    with pytest.raises(httpx.HTTPStatusError):
        await _client(handler, max_attempts=2).search("col")

    assert calls["n"] == 2
    assert retry_delays == [0.5]


@pytest.mark.parametrize("operation", ["create", "update", "delete", "delete_many"])
async def test_writes_are_not_retried(operation: str) -> None:
    """create/update/delete must not replay: the API has no idempotency keys."""
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"detail": "slow"})

    with pytest.raises(httpx.HTTPStatusError):
        client = _client(handler)
        if operation == "create":
            await client.create("col", {"a": 1})
        elif operation == "update":
            await client.update_item("id", {"a": 1})
        elif operation == "delete":
            await client.delete_item("id")
        else:
            await client.delete_many("col", {"a": 1})

    assert calls["n"] == 1


async def test_default_timeout_is_applied() -> None:
    client = AgentDataClient(
        base_url="http://backend", api_key="k", project_id="p", deployment_name="d"
    )
    assert client._timeout.connect == 5.0
    assert client._timeout.read == 30.0


async def test_float_timeout_is_normalized() -> None:
    client = AgentDataClient(
        base_url="http://backend",
        api_key="k",
        project_id="p",
        deployment_name="d",
        timeout=10.0,
    )
    assert client._timeout.connect == 10.0
    assert client._timeout.read == 10.0


async def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        AgentDataClient(
            base_url="http://backend",
            api_key="k",
            project_id="p",
            deployment_name="d",
            max_attempts=0,
        )
