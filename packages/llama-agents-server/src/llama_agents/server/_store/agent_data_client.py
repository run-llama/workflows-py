# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""AgentDataClient — shared HTTP client for the LlamaCloud Agent Data API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# A slow-but-alive backend should not be cut at httpx's 5s default, and a
# transient blip on a *read* should not surface as a hard, permanent-looking
# failure. Writes are not retried: the Agent Data API has no idempotency keys
# yet, so replaying a create/update/delete after an ambiguous timeout could
# duplicate or clobber data. Only idempotent reads opt in to retries.
_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 0.5
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


def _normalize_timeout(timeout: httpx.Timeout | float | None) -> httpx.Timeout:
    if timeout is None:
        return _DEFAULT_TIMEOUT
    if isinstance(timeout, httpx.Timeout):
        return timeout
    return httpx.Timeout(timeout)


class AgentDataClient:
    """HTTP client for the LlamaCloud Agent Data API.

    Holds connection parameters and exposes search/create/update/delete methods.
    Both AgentDataStore and AgentDataStateStore use this instead of duplicating
    HTTP helpers.

    Uses a shared ``httpx.AsyncClient`` for connection pooling. The client is
    lazily created on first use to avoid requiring an event loop at init time.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        project_id: str,
        deployment_name: str,
        timeout: httpx.Timeout | float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._project_id = project_id
        self._deployment_name = deployment_name
        self._timeout = _normalize_timeout(timeout)
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._max_attempts = max_attempts
        self._shared_client: httpx.AsyncClient | None = None

    @property
    def deployment_name(self) -> str:
        return self._deployment_name

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def http_client(self) -> httpx.AsyncClient:
        """Return the shared async HTTP client, creating it lazily.

        The client is reused across operations for connection pooling.
        ``httpx.AsyncClient`` is safe for concurrent use.
        """
        if self._shared_client is None or self._shared_client.is_closed:
            self._shared_client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers(),
                params={"project_id": self._project_id},
                timeout=self._timeout,
            )
        return self._shared_client

    async def close(self) -> None:
        """Close the shared HTTP client and release connections."""
        if self._shared_client is not None and not self._shared_client.is_closed:
            await self._shared_client.aclose()
            self._shared_client = None

    async def _request(
        self,
        method: str,
        url: str,
        *,
        retry_transient_errors: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        """Issue a request under the configured timeout.

        When ``retry_transient_errors`` is set (idempotent reads only),
        connection/read errors and 5xx responses are retried with exponential
        backoff. Writes do not opt in because the API has no idempotency keys.
        """
        client = self.http_client()
        attempts = self._max_attempts if retry_transient_errors else 1
        for attempt in range(1, attempts + 1):
            try:
                resp = await client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                if (
                    not retry_transient_errors
                    or exc.response.status_code not in _RETRYABLE_STATUS
                    or attempt == attempts
                ):
                    raise
                await self._sleep_before_retry(method, url, attempt, attempts, exc)
            except httpx.TransportError as exc:
                if not retry_transient_errors or attempt == attempts:
                    raise
                await self._sleep_before_retry(method, url, attempt, attempts, exc)
        raise RuntimeError("unreachable retry loop exit")

    async def _sleep_before_retry(
        self,
        method: str,
        url: str,
        attempt: int,
        attempts: int,
        exc: Exception,
    ) -> None:
        delay = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
        logger.warning(
            "Agent Data %s %s failed (attempt %d/%d), retrying in %.1fs: %r",
            method,
            url,
            attempt,
            attempts,
            delay,
            exc,
        )
        await asyncio.sleep(delay)

    async def search(
        self,
        collection: str,
        filters: dict[str, Any] | None = None,
        page_size: int = 100,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the Agent Data API and return matching items."""
        body: dict[str, Any] = {
            "deployment_name": self._deployment_name,
            "collection": collection,
            "page_size": page_size,
        }
        if filters:
            body["filter"] = filters
        if order_by:
            body["order_by"] = order_by
        resp = await self._request(
            "POST",
            "/api/v1/beta/agent-data/:search",
            json=body,
            retry_transient_errors=True,
        )
        return resp.json().get("items", [])

    async def create(self, collection: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create an item in the Agent Data API."""
        body = {
            "deployment_name": self._deployment_name,
            "collection": collection,
            "data": data,
        }
        resp = await self._request("POST", "/api/v1/beta/agent-data", json=body)
        return resp.json()

    async def update_item(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update an existing item by its Agent Data API ID."""
        resp = await self._request(
            "PUT",
            f"/api/v1/beta/agent-data/{item_id}",
            json={"data": data},
        )
        return resp.json()

    async def delete_item(self, item_id: str) -> None:
        """Delete an item by its Agent Data API ID."""
        await self._request("DELETE", f"/api/v1/beta/agent-data/{item_id}")

    async def delete_many(
        self,
        collection: str,
        filters: dict[str, Any],
    ) -> int:
        """Delete items matching the given filters. Returns the number deleted."""
        body: dict[str, Any] = {
            "deployment_name": self._deployment_name,
            "collection": collection,
            "filter": filters,
        }
        resp = await self._request("POST", "/api/v1/beta/agent-data/:delete", json=body)
        return resp.json().get("deleted_count", 0)
