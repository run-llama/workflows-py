from __future__ import annotations

import httpx
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_agents.appserver.routers.ui_proxy import create_ui_proxy_router


def build_client() -> TestClient:
    app = FastAPI()
    app.include_router(create_ui_proxy_router("dep", 3000))
    return TestClient(app)


@respx.mock
def test_proxy_success() -> None:
    client = build_client()
    respx.get("http://localhost:3000/deployments/dep/ui/x.js").mock(
        return_value=httpx.Response(200, content=b"x")
    )
    r = client.get("/deployments/dep/ui/x.js")
    assert r.status_code == 200
    assert r.content == b"x"


@respx.mock
def test_proxy_timeout() -> None:
    client = build_client()
    respx.get("http://localhost:3000/deployments/dep/ui/x").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    r = client.get("/deployments/dep/ui/x")
    assert r.status_code == 504


@respx.mock
def test_proxy_connect_error_results_in_502() -> None:
    client = build_client()
    respx.get("http://localhost:3000/deployments/dep/ui/y").mock(
        side_effect=httpx.ConnectError("boom")
    )
    r = client.get("/deployments/dep/ui/y")
    assert r.status_code == 502


@respx.mock
def test_header_filtering_and_methods_with_body_and_headers() -> None:
    client = build_client()

    route = respx.post("http://localhost:3000/deployments/dep/ui/submit").mock(
        return_value=httpx.Response(
            201,
            json={"ok": True},
            headers={"X-Foo": "bar", "Transfer-Encoding": "chunked"},
        )
    )

    r = client.post(
        "/deployments/dep/ui/submit",
        content=b"payload",
        headers={
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
            "X-Custom": "123",
        },
    )

    assert r.status_code == 201
    # Hop-by-hop headers should be stripped from response
    assert "transfer-encoding" not in {k.lower() for k in r.headers.keys()}
    # Custom header from upstream should pass through
    assert r.headers.get("X-Foo") == "bar"

    # Verify request seen by upstream filtered hop-by-hop headers and preserved payload
    assert route.called
    req = route.calls.last.request
    assert req.read() == b"payload"
    # hop-by-hop stripped where applicable (httpx may add 'connection' back)
    assert "transfer-encoding" not in {k.lower() for k in req.headers.keys()}
    # custom header retained
    assert req.headers.get("X-Custom") == "123"
