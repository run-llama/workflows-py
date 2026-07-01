"""Prometheus metrics for the Kubernetes client connection pools.

Exposes per-pool saturation via a custom collector computed at scrape time (no
background task, always fresh). Pool starvation — long-lived log streams holding
every warm connection while short reads open cold TLS connections — is invisible
to CPU/latency dashboards, so we surface it directly:

- ``k8s_client_pool_in_use_connections`` at ``maxsize`` (idle at 0) means the pool
  is fully checked out.
- ``k8s_client_pool_connections_created`` climbing means cold-connection churn: the
  pool is too small (or contended) and urllib3 is opening throwaway connections.
"""

import logging
from collections.abc import Iterator

from llama_agents.control_plane.k8s_client import _k8s_client
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.metrics_core import Metric
from prometheus_client.registry import REGISTRY, Collector, CollectorRegistry

logger = logging.getLogger(__name__)

_LABELS = ["client", "host"]


class _K8sPoolCollector(Collector):
    """Samples every registered kube-apiserver client's urllib3 pools on scrape."""

    def collect(self) -> Iterator[Metric]:
        maxsize = GaugeMetricFamily(
            "k8s_client_pool_maxsize",
            "Configured max warm connections in the kube-apiserver client pool",
            labels=_LABELS,
        )
        in_use = GaugeMetricFamily(
            "k8s_client_pool_in_use_connections",
            "Connections currently checked out of the pool (maxsize - idle slots)",
            labels=_LABELS,
        )
        idle = GaugeMetricFamily(
            "k8s_client_pool_idle_connections",
            "Idle connection slots currently available in the pool",
            labels=_LABELS,
        )
        created = CounterMetricFamily(
            "k8s_client_pool_connections_created",
            "Connections created by the pool; a rising rate indicates cold-connection churn",
            labels=_LABELS,
        )
        requests = CounterMetricFamily(
            "k8s_client_pool_requests",
            "Requests issued through the pool",
            labels=_LABELS,
        )

        # Never let a metrics scrape fail because of pool introspection.
        try:
            self._sample(maxsize, in_use, idle, created, requests)
        except Exception:
            logger.warning(
                "Failed to sample k8s connection pool metrics", exc_info=True
            )

        yield from (maxsize, in_use, idle, created, requests)

    @staticmethod
    def _sample(
        maxsize: GaugeMetricFamily,
        in_use: GaugeMetricFamily,
        idle: GaugeMetricFamily,
        created: CounterMetricFamily,
        requests: CounterMetricFamily,
    ) -> None:
        for name, api_client in _k8s_client.iter_pool_clients().items():
            pool_manager = getattr(
                getattr(api_client, "rest_client", None), "pool_manager", None
            )
            if pool_manager is None:
                continue
            pools = pool_manager.pools
            for key in list(pools.keys()):
                try:
                    pool = pools[key]
                except KeyError:
                    # Evicted between snapshot and read — skip.
                    continue
                queue = getattr(pool, "pool", None)
                if queue is None:
                    continue
                pool_max = queue.maxsize
                free = queue.qsize()
                labels = [name, getattr(pool, "host", "") or ""]
                maxsize.add_metric(labels, pool_max)
                in_use.add_metric(labels, max(0, pool_max - free))
                idle.add_metric(labels, free)
                created.add_metric(labels, getattr(pool, "num_connections", 0))
                requests.add_metric(labels, getattr(pool, "num_requests", 0))


_collector: _K8sPoolCollector | None = None


def register_k8s_pool_metrics(registry: CollectorRegistry = REGISTRY) -> None:
    """Idempotently register the k8s connection-pool collector.

    Safe to call from multiple app lifespans in the same process; only the first
    call registers.
    """
    global _collector
    if _collector is not None:
        return
    collector = _K8sPoolCollector()
    try:
        registry.register(collector)
    except ValueError:
        # Duplicate registration (e.g. a second app lifespan) — treat as done.
        _collector = collector
        return
    _collector = collector
