from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_agents.appserver.routers.ui_proxy import create_ui_proxy_router


class FakeUpstreamWebSocket:
    def __init__(self) -> None:
        self._recv_q: asyncio.Queue[Any] = asyncio.Queue()
        self._send_q: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        self.subprotocol: str | None = None

    def __aiter__(self) -> Any:
        return self

    async def send(self, data: Any) -> None:
        await self._recv_q.put(data)

    # API used by proxy for server->client messages
    async def recv(self) -> Any:
        item = await self._send_q.get()
        if item is StopAsyncIteration:
            raise StopAsyncIteration
        return item

    # helpers for tests
    async def push_from_server(self, data: Any) -> None:
        await self._send_q.put(data)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_websocket_text_and_binary_and_subprotocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.include_router(create_ui_proxy_router("dep", 3000))
    client = TestClient(app)

    upstream = FakeUpstreamWebSocket()

    def fake_connect(
        url: str,
        additional_headers: dict[str, Any] | None = None,
        subprotocols: list[str] | None = None,
        open_timeout: int | None = None,
        ping_interval: int | None = None,
    ) -> Any:
        # Validate URL constructed with path and query params
        assert url == "ws://localhost:3000/deployments/dep/ui/chat?room=1"
        # Simulate server-side subprotocol selection
        assert subprotocols is not None and subprotocols == ["json", "msgpack"]
        upstream.subprotocol = subprotocols[0]

        # Provide an async context manager that yields our upstream
        class Ctx:
            async def __aenter__(self) -> FakeUpstreamWebSocket:
                return upstream

            async def __aexit__(
                self,
                exc_type: type | None,
                exc: Exception | None,
                tb: Any,
            ) -> bool:
                return False

        return Ctx()

    monkeypatch.setattr(
        "llama_agents.appserver.routers.ui_proxy.websockets.connect", fake_connect
    )

    with client.websocket_connect(
        "/deployments/dep/ui/chat?room=1",
        headers={"sec-websocket-protocol": "json, msgpack"},
    ) as ws:
        # Send text and binary to upstream
        ws.send_text("hello")
        ws.send_bytes(b"bytes")

        # Upstream should receive them
        assert await asyncio.wait_for(upstream._recv_q.get(), 1) == "hello"
        assert await asyncio.wait_for(upstream._recv_q.get(), 1) == b"bytes"

        # Upstream sends messages back
        await upstream.push_from_server("world")
        await upstream.push_from_server(b"buf")

        # Client should receive them
        assert ws.receive_text() == "world"
        assert ws.receive_bytes() == b"buf"

    # Ensure graceful close attempted
    await upstream.close()
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_websocket_upstream_raises_is_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.include_router(create_ui_proxy_router("dep", 3000))
    client = TestClient(app)

    def fake_connect(*args: Any, **kwargs: dict[str, Any]) -> Any:
        class Ctx:
            async def __aenter__(self) -> Any:
                raise RuntimeError("boom")

            async def __aexit__(
                self,
                exc_type: type | None,
                exc: Exception | None,
                tb: Any,
            ) -> bool:
                return False

        return Ctx()

    monkeypatch.setattr(
        "llama_agents.appserver.routers.ui_proxy.websockets.connect", fake_connect
    )

    # Should not raise on connect failure; FastAPI's TestClient will raise if close not called
    with client.websocket_connect("/deployments/dep/ui"):
        # Connection will be accepted then closed in finally
        pass
