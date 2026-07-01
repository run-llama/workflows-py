"""Tests for the kube-apiserver wedge safeguards: request timeouts and the
k8s-checking readiness probe (`check_k8s_connectivity` behind `/readyz`).
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, Generator, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from kubernetes.client.api_client import ApiClient
from kubernetes.client.configuration import Configuration
from kubernetes.client.exceptions import ApiException
from llama_agents.control_plane import k8s_client
from llama_agents.control_plane.k8s_client import (
    K8sClient,
    _TimeoutApiClient,
    check_k8s_connectivity,
)
from llama_agents.control_plane.settings import settings
from urllib3.exceptions import MaxRetryError
from urllib3.util import Timeout


@pytest.fixture
def initialized_client() -> K8sClient:
    """A K8sClient whose ApiClients are built without a real cluster."""

    def fake_incluster() -> None:
        cfg = Configuration()
        cfg.host = "https://kube.test:443"
        # kubernetes stubs omit set_default; cast like k8s_client.py does for the
        # library's incomplete typing.
        cast(Any, Configuration).set_default(cfg)

    with patch(
        "llama_agents.control_plane.k8s_client.k8s_config.load_incluster_config",
        side_effect=fake_incluster,
    ):
        client = K8sClient()
        client._ensure_k8s_client()
    return client


@pytest.fixture
def mock_k8s() -> Generator[MagicMock, None, None]:
    with patch("llama_agents.control_plane.k8s_client._k8s_client") as mock_k8s:
        yield mock_k8s


def test_control_client_gets_default_timeout(initialized_client: K8sClient) -> None:
    """The shared control pool gets a default (connect, read) timeout from settings."""
    control = initialized_client._control_api_client
    assert isinstance(control, _TimeoutApiClient)
    assert control._default_timeout == (
        settings.k8s_request_timeout_seconds,
        settings.k8s_request_timeout_seconds,
    )


def test_streaming_client_has_no_default_timeout(initialized_client: K8sClient) -> None:
    """The streaming pool must stay unbounded so `follow=True` reads never die."""
    streaming = initialized_client._streaming_api_client
    assert not isinstance(streaming, _TimeoutApiClient)


def test_timeout_api_client_fills_in_default_when_unset() -> None:
    """`request()` injects the default timeout when the caller passes none."""
    client = _TimeoutApiClient(
        default_request_timeout=17.0, configuration=Configuration()
    )
    with patch.object(ApiClient, "request") as base_request:
        client.request("GET", "http://example.test")
    assert base_request.call_args.kwargs["_request_timeout"] == (17.0, 17.0)


def test_default_timeout_reaches_urllib3_as_a_real_timeout(
    initialized_client: K8sClient,
) -> None:
    """Regression test for a silent no-op: `kubernetes/client/rest.py` only builds
    a `urllib3.Timeout` when `_request_timeout` is an `int` or a 2-tuple —
    `isinstance(20.0, int)` is False, so a bare float default would reach urllib3
    as `timeout=None` (no timeout at all) despite `_TimeoutApiClient` "setting" it.
    Exercise the real request() -> rest.py -> urllib3 chain, not just the
    ApiClient-level plumbing, so this can't regress silently again.
    """
    control = initialized_client._control_api_client
    assert isinstance(control, _TimeoutApiClient)
    captured: dict[str, object] = {}

    def fake_pool_request(method: str, url: str, **kwargs: object) -> Mock:
        captured["timeout"] = kwargs.get("timeout")
        return Mock(status=200)

    # kubernetes stubs omit `rest_client` from the curated ApiClient surface (it's
    # client-internal); cast like k8s_client.py does for the library's incomplete
    # typing.
    with patch.object(
        cast(Any, control).rest_client.pool_manager,
        "request",
        side_effect=fake_pool_request,
    ):
        control.request("GET", "https://kube.test/api", _preload_content=False)

    timeout = captured["timeout"]
    assert isinstance(timeout, Timeout)
    assert timeout.connect_timeout == settings.k8s_request_timeout_seconds
    timeout.start_connect()
    assert timeout.read_timeout == settings.k8s_request_timeout_seconds


def test_timeout_api_client_preserves_explicit_timeout() -> None:
    """An explicit `_request_timeout` from the caller is never overridden."""
    client = _TimeoutApiClient(
        default_request_timeout=17.0, configuration=Configuration()
    )
    with patch.object(ApiClient, "request") as base_request:
        client.request("GET", "http://example.test", _request_timeout=(1.0, None))
    assert base_request.call_args.kwargs["_request_timeout"] == (1.0, None)


def test_timeout_api_client_preserves_falsy_explicit_timeout() -> None:
    """A falsy-but-explicit `_request_timeout` (e.g. `0`) must not be treated as
    unset — only `None` means "caller didn't pass one"."""
    client = _TimeoutApiClient(
        default_request_timeout=17.0, configuration=Configuration()
    )
    with patch.object(ApiClient, "request") as base_request:
        client.request("GET", "http://example.test", _request_timeout=0)
    assert base_request.call_args.kwargs["_request_timeout"] == 0


@pytest.mark.asyncio
async def test_stream_container_logs_uses_connect_only_timeout(
    mock_k8s: MagicMock,
) -> None:
    """Log streaming must set a connect-only timeout, never a read timeout."""
    mock_k8s.k8s_core_v1_streaming.read_namespaced_pod_log.return_value = "hello\n"

    await k8s_client.stream_container_logs("pod-1", "app", follow=True)

    _, kwargs = mock_k8s.k8s_core_v1_streaming.read_namespaced_pod_log.call_args
    connect_timeout, read_timeout = kwargs["_request_timeout"]
    assert connect_timeout == settings.k8s_streaming_connect_timeout_seconds
    assert read_timeout is None


@pytest.mark.asyncio
async def test_stream_container_logs_retries_on_connect_failure(
    mock_k8s: MagicMock,
) -> None:
    """A connect failure opening the stream (raised by urllib3, not ApiException,
    now that a connect-only timeout is set) must be retried like a 400/404 —
    not left to crash the caller with a raw urllib3 exception.
    """
    connect_error = MaxRetryError(
        pool=MagicMock(), url="https://kube.test/log", reason=None
    )
    mock_k8s.k8s_core_v1_streaming.read_namespaced_pod_log.side_effect = [
        connect_error,
        "hello\n",
    ]

    with patch(
        "llama_agents.control_plane.k8s_client.asyncio.sleep", AsyncMock()
    ) as mock_sleep:
        cancel, gen = await k8s_client.stream_container_logs("pod-1", "app")
        line = await anext(gen)

    assert line == "hello"
    mock_sleep.assert_awaited_once()
    await cancel()


@pytest.mark.asyncio
async def test_check_k8s_connectivity_success(mock_k8s: MagicMock) -> None:
    """A healthy `GET /version` completes without raising, with a short timeout."""
    mock_k8s.k8s_version.get_code.return_value = object()

    await check_k8s_connectivity()

    mock_k8s.k8s_version.get_code.assert_called_once()
    _, kwargs = mock_k8s.k8s_version.get_code.call_args
    timeout = settings.k8s_health_check_timeout_seconds
    assert kwargs["_request_timeout"] == (timeout, timeout)


@pytest.mark.asyncio
async def test_check_k8s_connectivity_propagates_api_errors(
    mock_k8s: MagicMock,
) -> None:
    """A failed apiserver read is surfaced to the caller (probes treat it as down)."""
    mock_k8s.k8s_version.get_code.side_effect = ApiException(status=500)

    with pytest.raises(ApiException):
        await check_k8s_connectivity()


@pytest.mark.asyncio
async def test_check_k8s_connectivity_times_out_on_a_wedged_connection(
    mock_k8s: MagicMock,
) -> None:
    """A hung call (the actual wedge symptom) must not block the probe forever."""

    # Simulate the wedge as an immediate timeout rather than a real sleep, so the
    # test doesn't depend on wall-clock settings.k8s_request_timeout_seconds. Close
    # the underlying to_thread coroutine to avoid an "never awaited" warning.
    def _simulate_timeout(
        coro: Coroutine[object, object, object], timeout: float
    ) -> None:
        coro.close()
        raise asyncio.TimeoutError

    with patch(
        "llama_agents.control_plane.k8s_client.asyncio.wait_for",
        side_effect=_simulate_timeout,
    ):
        with pytest.raises(asyncio.TimeoutError):
            await check_k8s_connectivity()
