# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Fake Agent Data API backend for testing AgentDataStore."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest
from llama_agents.server import AgentDataStore


class FakeAgentDataBackend:
    """In-memory backend that simulates the Agent Data API HTTP endpoints.

    Stores items keyed by (deployment_name, collection) and provides
    search/create/update/delete semantics matching the real API.
    """

    def __init__(self) -> None:
        # (deployment_name, collection) → list[{id, deployment_name, collection, data}]
        self._items: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def _key(self, deployment_name: str, collection: str) -> tuple[str, str]:
        return (deployment_name, collection)

    def _get_items(self, deployment_name: str, collection: str) -> list[dict[str, Any]]:
        return self._items.setdefault(self._key(deployment_name, collection), [])

    def search(
        self,
        deployment_name: str,
        collection: str,
        filters: dict[str, Any] | None = None,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        items = self._get_items(deployment_name, collection)
        if not filters:
            return items[:page_size]

        matched = []
        for item in items:
            data = item["data"]
            if self._matches(data, filters):
                matched.append(item)
        return matched[:page_size]

    @staticmethod
    def _matches(data: dict[str, Any], filters: dict[str, Any]) -> bool:
        for field, ops in filters.items():
            value = data.get(field)
            for op, expected in ops.items():
                if op == "eq" and value != expected:
                    return False
                if op == "includes" and value not in expected:
                    return False
                if op == "gte" and (value is None or value < expected):
                    return False
        return True

    def create(
        self, deployment_name: str, collection: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        items = self._get_items(deployment_name, collection)
        item = {
            "id": str(uuid.uuid4()),
            "deployment_name": deployment_name,
            "collection": collection,
            "data": data,
        }
        items.append(item)
        return item

    def update_item(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        for items_list in self._items.values():
            for item in items_list:
                if item["id"] == item_id:
                    item["data"] = data
                    return item
        raise ValueError(f"Item {item_id} not found")

    def delete_item(self, item_id: str) -> None:
        for items_list in self._items.values():
            for i, item in enumerate(items_list):
                if item["id"] == item_id:
                    items_list.pop(i)
                    return
        raise ValueError(f"Item {item_id} not found")

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Route an httpx.Request to the appropriate handler."""
        path = request.url.path
        method = request.method

        if method == "POST" and path == "/api/v1/beta/agent-data/:search":
            body = json.loads(request.content)
            items = self.search(
                body["deployment_name"],
                body["collection"],
                body.get("filter"),
                body.get("page_size", 100),
            )
            return httpx.Response(200, json={"items": items})

        if method == "POST" and path == "/api/v1/beta/agent-data":
            body = json.loads(request.content)
            item = self.create(
                body["deployment_name"], body["collection"], body["data"]
            )
            return httpx.Response(200, json=item)

        if method == "PUT" and path.startswith("/api/v1/beta/agent-data/"):
            item_id = path.split("/")[-1]
            body = json.loads(request.content)
            item = self.update_item(item_id, body["data"])
            return httpx.Response(200, json=item)

        if method == "DELETE" and path.startswith("/api/v1/beta/agent-data/"):
            item_id = path.split("/")[-1]
            self.delete_item(item_id)
            return httpx.Response(200, json={})

        return httpx.Response(404, json={"error": "not found"})


def create_agent_data_store(
    backend: FakeAgentDataBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> AgentDataStore:
    """Create an AgentDataStore with httpx patched to use the fake backend."""
    store = AgentDataStore(
        base_url="https://fake-api.example.com",
        api_key="test-key",
        project_id="test-project",
        deployment_name="test-deploy",
        collection="handlers",
    )

    original_client = store._client

    def patched_client() -> httpx.AsyncClient:
        client = original_client()
        client._transport = httpx.MockTransport(backend.handle_request)
        return client

    monkeypatch.setattr(store, "_client", patched_client)
    return store
