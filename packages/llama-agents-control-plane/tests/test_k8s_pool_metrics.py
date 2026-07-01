"""Tests for k8s connection-pool isolation and saturation metrics."""

from typing import Any, cast
from unittest.mock import patch

import pytest
from kubernetes.client.configuration import Configuration
from llama_agents.control_plane.k8s_client import K8sClient
from llama_agents.control_plane.k8s_metrics import _K8sPoolCollector
from prometheus_client import CollectorRegistry, generate_latest


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


def test_control_apis_share_one_pool(initialized_client: K8sClient) -> None:
    """Short control-plane reads/writes share a single ApiClient/pool."""
    # kubernetes stubs omit `api_client` from the curated Api surface; cast like
    # k8s_client.py does for the library's incomplete typing.
    control = cast(Any, initialized_client.k8s_core_v1).api_client
    assert cast(Any, initialized_client.k8s_custom_objects).api_client is control
    assert cast(Any, initialized_client.k8s_networking_v1).api_client is control
    assert cast(Any, initialized_client.k8s_apps_v1).api_client is control


def test_streaming_pool_is_isolated(initialized_client: K8sClient) -> None:
    """Log streaming uses a distinct ApiClient so it cannot starve control reads."""
    assert (
        cast(Any, initialized_client.k8s_core_v1_streaming).api_client
        is not cast(Any, initialized_client.k8s_core_v1).api_client
    )


def test_pool_sizes_come_from_settings(initialized_client: K8sClient) -> None:
    """connection_pool_maxsize is driven by settings, not the urllib3 default of 4."""
    from llama_agents.control_plane.settings import settings

    # `_control_api_client`/`_streaming_api_client` are declared `ApiClient | None`
    # (None only before `_ensure_k8s_client` runs); cast past that for the fixture's
    # post-init state, and past the stubs' incomplete `.configuration` typing.
    assert (
        cast(
            Any, initialized_client._control_api_client
        ).configuration.connection_pool_maxsize
        == settings.k8s_connection_pool_maxsize
    )
    assert (
        cast(
            Any, initialized_client._streaming_api_client
        ).configuration.connection_pool_maxsize
        == settings.k8s_streaming_connection_pool_maxsize
    )


def test_iter_pool_clients_empty_before_init() -> None:
    """The metrics collector must tolerate an uninitialized client."""
    assert K8sClient().iter_pool_clients() == {}


def test_iter_pool_clients_after_init(initialized_client: K8sClient) -> None:
    assert set(initialized_client.iter_pool_clients()) == {"control", "streaming"}


def _render(collector: _K8sPoolCollector) -> str:
    registry = CollectorRegistry()
    registry.register(collector)
    return generate_latest(registry).decode()


def test_collector_emits_pool_families(initialized_client: K8sClient) -> None:
    """Collector reports maxsize/in_use/idle/created/requests per client pool."""
    # Force a pool to exist on each client (lazily created on first host use).
    for api_client in initialized_client.iter_pool_clients().values():
        cast(Any, api_client).rest_client.pool_manager.connection_from_host(
            "kube.test", 443, "https"
        )

    with patch(
        "llama_agents.control_plane.k8s_metrics._k8s_client", initialized_client
    ):
        out = _render(_K8sPoolCollector())

    assert 'k8s_client_pool_maxsize{client="control",host="kube.test"}' in out
    assert 'k8s_client_pool_maxsize{client="streaming",host="kube.test"}' in out
    assert "k8s_client_pool_in_use_connections" in out
    assert "k8s_client_pool_idle_connections" in out
    assert "k8s_client_pool_connections_created_total" in out
    assert "k8s_client_pool_requests_total" in out


def test_collector_reports_in_use_on_checkout(initialized_client: K8sClient) -> None:
    """in_use == maxsize - idle tracks checked-out connections."""
    control = cast(Any, initialized_client._control_api_client)
    pool = control.rest_client.pool_manager.connection_from_host(
        "kube.test", 443, "https"
    )
    maxsize = pool.pool.maxsize
    pool._get_conn()  # check one out
    pool._get_conn()  # and another

    with patch(
        "llama_agents.control_plane.k8s_metrics._k8s_client", initialized_client
    ):
        out = _render(_K8sPoolCollector())

    line = next(
        ln
        for ln in out.splitlines()
        if ln.startswith("k8s_client_pool_in_use_connections")
        and 'client="control"' in ln
    )
    assert line.rsplit(" ", 1)[1] == "2.0"
    idle_line = next(
        ln
        for ln in out.splitlines()
        if ln.startswith("k8s_client_pool_idle_connections")
        and 'client="control"' in ln
    )
    assert idle_line.rsplit(" ", 1)[1] == f"{float(maxsize - 2)}"


def test_collector_yields_nothing_when_uninitialized() -> None:
    """No pools, no error, no samples."""
    with patch("llama_agents.control_plane.k8s_metrics._k8s_client", K8sClient()):
        out = _render(_K8sPoolCollector())
    # Metric families are declared (HELP/TYPE) but carry no samples.
    assert "k8s_client_pool_maxsize" in out
    assert "k8s_client_pool_maxsize{" not in out
