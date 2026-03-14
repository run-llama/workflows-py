# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""AgentDataClient — shared HTTP client for the LlamaCloud Agent Data API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._project_id = project_id
        self._deployment_name = deployment_name
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
            )
        return self._shared_client

    async def close(self) -> None:
        """Close the shared HTTP client and release connections."""
        if self._shared_client is not None and not self._shared_client.is_closed:
            await self._shared_client.aclose()
            self._shared_client = None

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
        client = self.http_client()
        resp = await client.post("/api/v1/beta/agent-data/:search", json=body)
        resp.raise_for_status()
        return resp.json().get("items", [])

    async def create(self, collection: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create an item in the Agent Data API."""
        body = {
            "deployment_name": self._deployment_name,
            "collection": collection,
            "data": data,
        }
        client = self.http_client()
        resp = await client.post("/api/v1/beta/agent-data", json=body)
        resp.raise_for_status()
        return resp.json()

    async def update_item(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update an existing item by its Agent Data API ID."""
        client = self.http_client()
        resp = await client.put(
            f"/api/v1/beta/agent-data/{item_id}",
            json={"data": data},
        )
        resp.raise_for_status()
        return resp.json()

    async def delete_item(self, item_id: str) -> None:
        """Delete an item by its Agent Data API ID."""
        client = self.http_client()
        resp = await client.delete(f"/api/v1/beta/agent-data/{item_id}")
        resp.raise_for_status()

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
        client = self.http_client()
        resp = await client.post("/api/v1/beta/agent-data/:delete", json=body)
        resp.raise_for_status()
        return resp.json().get("deleted_count", 0)
