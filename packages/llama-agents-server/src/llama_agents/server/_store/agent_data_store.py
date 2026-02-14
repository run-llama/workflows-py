# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""AgentDataStore — AbstractWorkflowStore backed by the LlamaCloud Agent Data API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, List

import httpx
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from workflows.context.serializers import BaseSerializer
from workflows.context.state_store import StateStore

from .._keyed_lock import KeyedLock
from .._lru_cache import LRUCache
from .abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
    StoredEvent,
    StoredTick,
)
from .agent_data_state_store import AgentDataStateStore

logger = logging.getLogger(__name__)


class AgentDataStore(AbstractWorkflowStore):
    """Workflow store backed by the LlamaCloud Agent Data API.

    Persists handlers, events, and ticks in separate Agent Data API
    collections. Event subscription uses in-process ``asyncio.Condition``
    notifications (single-node only, no polling).

    State stores are in-memory (``InMemoryStateStore``) — workflow state
    is reconstructed from ticks on reload, so cloud persistence of the
    mutable state object is unnecessary.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        project_id: str,
        deployment_name: str,
        collection: str = "workflow_contexts",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._project_id = project_id
        self._deployment_name = deployment_name
        self._collection = collection

        # Derived collection names for events and ticks
        self._events_collection = f"{collection}_events"
        self._ticks_collection = f"{collection}_ticks"

        # handler_id → agent_data_id cache (avoids search-before-update)
        self._id_cache: LRUCache[str, str] = LRUCache(maxsize=256)

        # Per-key lock for serializing updates to the same handler
        self._locks = KeyedLock()

        # Per-run_id sequence counters for events and ticks
        self._event_sequences: dict[str, int] = {}
        self._tick_sequences: dict[str, int] = {}
        self._seq_lock = asyncio.Lock()

        # Per-run_id asyncio.Condition for event subscription
        self._conditions: dict[str, asyncio.Condition] = {}

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _client(self) -> httpx.AsyncClient:
        """Create a fresh async HTTP client.

        Following the cloud pattern: a new client per operation to avoid
        issues with shared state across concurrent async tasks.
        """
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers(),
            params={"project_id": self._project_id},
        )

    async def _search(
        self,
        collection: str,
        filters: dict[str, Any] | None = None,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Search the Agent Data API and return matching items."""
        body: dict[str, Any] = {
            "deployment_name": self._deployment_name,
            "collection": collection,
            "page_size": page_size,
        }
        if filters:
            body["filter"] = filters
        async with self._client() as client:
            resp = await client.post("/api/v1/beta/agent-data/:search", json=body)
            resp.raise_for_status()
            return resp.json().get("items", [])

    async def _create(self, collection: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create an item in the Agent Data API."""
        body = {
            "deployment_name": self._deployment_name,
            "collection": collection,
            "data": data,
        }
        async with self._client() as client:
            resp = await client.post("/api/v1/beta/agent-data", json=body)
            resp.raise_for_status()
            return resp.json()

    async def _update_item(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update an existing item by its Agent Data API ID."""
        async with self._client() as client:
            resp = await client.put(
                f"/api/v1/beta/agent-data/{item_id}",
                json={"data": data},
            )
            resp.raise_for_status()
            return resp.json()

    async def _delete_item(self, item_id: str) -> None:
        """Delete an item by its Agent Data API ID."""
        async with self._client() as client:
            resp = await client.delete(f"/api/v1/beta/agent-data/{item_id}")
            resp.raise_for_status()

    # ------------------------------------------------------------------
    # Condition helpers (same pattern as MemoryWorkflowStore)
    # ------------------------------------------------------------------

    def _get_condition(self, run_id: str) -> asyncio.Condition:
        if run_id not in self._conditions:
            self._conditions[run_id] = asyncio.Condition()
        return self._conditions[run_id]

    # ------------------------------------------------------------------
    # Handler CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _build_handler_filters(query: HandlerQuery) -> dict[str, Any] | None:
        """Convert a HandlerQuery to Agent Data API filter format."""
        filters: dict[str, Any] = {}

        if query.handler_id_in is not None:
            if len(query.handler_id_in) == 0:
                return None  # empty list → match nothing
            if len(query.handler_id_in) == 1:
                filters["handler_id"] = {"eq": query.handler_id_in[0]}
            else:
                filters["handler_id"] = {"includes": query.handler_id_in}

        if query.run_id_in is not None:
            if len(query.run_id_in) == 0:
                return None
            if len(query.run_id_in) == 1:
                filters["run_id"] = {"eq": query.run_id_in[0]}
            else:
                filters["run_id"] = {"includes": query.run_id_in}

        if query.workflow_name_in is not None:
            if len(query.workflow_name_in) == 0:
                return None
            if len(query.workflow_name_in) == 1:
                filters["workflow_name"] = {"eq": query.workflow_name_in[0]}
            else:
                filters["workflow_name"] = {"includes": query.workflow_name_in}

        if query.status_in is not None:
            if len(query.status_in) == 0:
                return None
            if len(query.status_in) == 1:
                filters["status"] = {"eq": query.status_in[0]}
            else:
                filters["status"] = {"includes": query.status_in}

        # is_idle filter: idle_since is non-null when idle
        # The Agent Data API doesn't have direct null checks, so we filter
        # client-side for is_idle in query() instead.

        return filters if filters else None

    @staticmethod
    def _item_to_handler(item: dict[str, Any]) -> PersistentHandler:
        """Convert an Agent Data API item to a PersistentHandler."""
        data = item["data"]
        return PersistentHandler.model_validate(data)

    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        filters = self._build_handler_filters(query)
        # None means an empty filter list was given → match nothing
        if filters is None and (
            query.handler_id_in is not None
            or query.run_id_in is not None
            or query.workflow_name_in is not None
            or query.status_in is not None
        ):
            return []

        items = await self._search(self._collection, filters)
        handlers = [self._item_to_handler(item) for item in items]

        # Client-side is_idle filtering (API can't filter on null/non-null)
        if query.is_idle is not None:
            handlers = [
                h for h in handlers if (h.idle_since is not None) == query.is_idle
            ]

        return handlers

    async def update(self, handler: PersistentHandler) -> None:
        data = handler.model_dump(mode="json")
        handler_id = handler.handler_id

        async with self._locks(handler_id):
            # Check cache for existing agent_data_id
            cached_id = self._id_cache.get(handler_id)
            if cached_id is not None:
                await self._update_item(cached_id, data)
                return

            # Search for existing item
            items = await self._search(
                self._collection,
                {"handler_id": {"eq": handler_id}},
            )
            if items:
                item_id = items[0]["id"]
                self._id_cache.put(handler_id, item_id)
                await self._update_item(item_id, data)
            else:
                result = await self._create(self._collection, data)
                self._id_cache.put(handler_id, result["id"])

    async def delete(self, query: HandlerQuery) -> int:
        handlers = await self.query(query)
        count = 0
        for handler in handlers:
            handler_id = handler.handler_id
            async with self._locks(handler_id):
                cached_id = self._id_cache.get(handler_id)
                if cached_id is not None:
                    await self._delete_item(cached_id)
                    self._id_cache.delete(handler_id)
                    count += 1
                else:
                    items = await self._search(
                        self._collection,
                        {"handler_id": {"eq": handler_id}},
                    )
                    for item in items:
                        await self._delete_item(item["id"])
                        count += 1
                    self._id_cache.delete(handler_id)
        return count

    # ------------------------------------------------------------------
    # Event journal
    # ------------------------------------------------------------------

    async def _next_event_sequence(self, run_id: str) -> int:
        async with self._seq_lock:
            seq = self._event_sequences.get(run_id, -1) + 1
            self._event_sequences[run_id] = seq
            return seq

    async def append_event(self, run_id: str, event: EventEnvelopeWithMetadata) -> None:
        seq = await self._next_event_sequence(run_id)
        now = datetime.now(timezone.utc)
        stored = StoredEvent(
            run_id=run_id,
            sequence=seq,
            timestamp=now,
            event=event,
        )
        await self._create(
            self._events_collection,
            stored.model_dump(mode="json"),
        )
        # Notify subscribers
        condition = self._get_condition(run_id)
        async with condition:
            condition.notify_all()

    async def query_events(
        self,
        run_id: str,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[StoredEvent]:
        filters: dict[str, Any] = {"run_id": {"eq": run_id}}
        if after_sequence is not None:
            filters["sequence"] = {"gte": after_sequence + 1}

        items = await self._search(
            self._events_collection,
            filters,
            page_size=limit or 1000,
        )

        events = [StoredEvent.model_validate(item["data"]) for item in items]
        # Sort client-side (API may not guarantee order)
        events.sort(key=lambda e: e.sequence)
        if limit is not None:
            events = events[:limit]
        return events

    # ------------------------------------------------------------------
    # Event subscription (condition-based, same as MemoryWorkflowStore)
    # ------------------------------------------------------------------

    async def subscribe_events(
        self, run_id: str, after_sequence: int = -1
    ) -> AsyncIterator[StoredEvent]:
        """Condition-based subscription — no polling.

        Uses the same pattern as ``MemoryWorkflowStore``: an
        ``asyncio.Condition`` per run_id is notified by ``append_event``,
        and subscribers wait on it.
        """
        condition = self._get_condition(run_id)
        cursor = after_sequence

        while True:
            events = await self.query_events(run_id, after_sequence=cursor)
            for event in events:
                yield event
                cursor = event.sequence
                if self._is_terminal_event(event):
                    self._conditions.pop(run_id, None)
                    return

            # Check if run already terminated (late subscriber)
            if cursor >= 0:
                all_events = await self.query_events(run_id)
                if all_events and self._is_terminal_event(all_events[-1]):
                    self._conditions.pop(run_id, None)
                    return

            async with condition:
                # Re-check before blocking
                new_events = await self.query_events(run_id, after_sequence=cursor)
                if not new_events:
                    await condition.wait()

    # ------------------------------------------------------------------
    # Tick journal
    # ------------------------------------------------------------------

    async def _next_tick_sequence(self, run_id: str) -> int:
        async with self._seq_lock:
            seq = self._tick_sequences.get(run_id, -1) + 1
            self._tick_sequences[run_id] = seq
            return seq

    async def append_tick(self, run_id: str, tick_data: dict[str, Any]) -> None:
        seq = await self._next_tick_sequence(run_id)
        now = datetime.now(timezone.utc)
        stored = StoredTick(
            run_id=run_id,
            sequence=seq,
            timestamp=now,
            tick_data=tick_data,
        )
        await self._create(
            self._ticks_collection,
            stored.model_dump(mode="json"),
        )

    async def get_ticks(self, run_id: str) -> list[StoredTick]:
        items = await self._search(
            self._ticks_collection,
            {"run_id": {"eq": run_id}},
        )
        ticks = [StoredTick.model_validate(item["data"]) for item in items]
        ticks.sort(key=lambda t: t.sequence)
        return ticks

    # ------------------------------------------------------------------
    # State store
    # ------------------------------------------------------------------

    def create_state_store(
        self,
        run_id: str,
        state_type: type[Any] | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> StateStore[Any]:
        return AgentDataStateStore(  # type: ignore[return-value]
            base_url=self._base_url,
            api_key=self._api_key,
            project_id=self._project_id,
            deployment_name=self._deployment_name,
            run_id=run_id,
            state_type=state_type,
            collection=f"{self._collection}_state",
            client_factory=self._client,
        )
