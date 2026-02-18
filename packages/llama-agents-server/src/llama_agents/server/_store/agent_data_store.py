# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""AgentDataStore — AbstractWorkflowStore backed by the LlamaCloud Agent Data API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, List

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
from .agent_data_client import AgentDataClient
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
        poll_interval: float = 30.0,
    ) -> None:
        self._client = AgentDataClient(
            base_url=base_url,
            api_key=api_key,
            project_id=project_id,
            deployment_name=deployment_name,
        )
        self._collection = collection
        self.poll_interval = poll_interval

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
        self._conditions: weakref.WeakValueDictionary[str, asyncio.Condition] = (
            weakref.WeakValueDictionary()
        )

    # ------------------------------------------------------------------
    # Notification helpers
    # ------------------------------------------------------------------

    def _get_or_create_condition(self, run_id: str) -> asyncio.Condition:
        cond = self._conditions.get(run_id)
        if cond is None:
            cond = asyncio.Condition()
            self._conditions[run_id] = cond
        return cond

    async def _max_sequence(self, collection: str, run_id: str) -> int:
        """Query the API for the max sequence in a collection for a run_id.

        Returns -1 if no items exist.
        """
        items = await self._client.search(
            collection,
            {"run_id": {"eq": run_id}},
            page_size=1,
            order_by="sequence desc",
        )
        if items:
            return items[0]["data"].get("sequence", -1)
        return -1

    # ------------------------------------------------------------------
    # Handler CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _in_filter(field: str, values: list[Any] | None) -> tuple[bool, dict[str, Any]]:
        """Build an ``includes`` filter for *field*.

        Returns ``(ok, filter_fragment)`` where *ok* is ``False`` when the
        caller should short-circuit with "match nothing" (empty list).
        """
        if values is None:
            return True, {}
        if len(values) == 0:
            return False, {}
        return True, {field: {"includes": values}}

    @staticmethod
    def _build_handler_filters(query: HandlerQuery) -> dict[str, Any] | None:
        """Convert a HandlerQuery to Agent Data API filter format.

        Returns ``{}`` for "match everything" or ``None`` for "match nothing".
        """
        filters: dict[str, Any] = {}

        for field, values in [
            ("handler_id", query.handler_id_in),
            ("run_id", query.run_id_in),
            ("workflow_name", query.workflow_name_in),
            ("status", query.status_in),
        ]:
            ok, fragment = AgentDataStore._in_filter(field, values)
            if not ok:
                return None
            filters.update(fragment)

        if query.is_idle is not None:
            if query.is_idle:
                filters["idle_since"] = {"ne": None}
            else:
                filters["idle_since"] = {"eq": None}

        return filters

    @staticmethod
    def _item_to_handler(item: dict[str, Any]) -> PersistentHandler:
        """Convert an Agent Data API item to a PersistentHandler."""
        data = item["data"]
        return PersistentHandler.model_validate(data)

    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        filters = self._build_handler_filters(query)
        if filters is None:
            return []

        items = await self._client.search(self._collection, filters or None)
        handlers = [self._item_to_handler(item) for item in items]
        return handlers

    async def update(self, handler: PersistentHandler) -> None:
        data = handler.model_dump(mode="json")
        handler_id = handler.handler_id

        async with self._locks(handler_id):
            # Check cache for existing agent_data_id
            cached_id = self._id_cache.get(handler_id)
            if cached_id is not None:
                await self._client.update_item(cached_id, data)
                return

            # Search for existing item
            items = await self._client.search(
                self._collection,
                {"handler_id": {"eq": handler_id}},
            )
            if items:
                item_id = items[0]["id"]
                self._id_cache.put(handler_id, item_id)
                await self._client.update_item(item_id, data)
            else:
                result = await self._client.create(self._collection, data)
                self._id_cache.put(handler_id, result["id"])

    async def delete(self, query: HandlerQuery) -> int:
        filters = self._build_handler_filters(query)
        if filters is None:
            return 0

        # Invalidate cached IDs for matching handlers before bulk delete
        items = await self._client.search(self._collection, filters or None)
        for item in items:
            handler_id = item["data"].get("handler_id")
            if handler_id:
                self._id_cache.delete(handler_id)

        if not items:
            return 0

        return await self._client.delete_many(self._collection, filters or {})

    # ------------------------------------------------------------------
    # Event journal
    # ------------------------------------------------------------------

    async def _next_event_sequence(self, run_id: str) -> int:
        async with self._seq_lock:
            if run_id not in self._event_sequences:
                self._event_sequences[run_id] = await self._max_sequence(
                    self._events_collection, run_id
                )
            seq = self._event_sequences[run_id] + 1
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
        await self._client.create(
            self._events_collection,
            stored.model_dump(mode="json"),
        )
        if condition := self._conditions.get(run_id):
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

        items = await self._client.search(
            self._events_collection,
            filters,
            page_size=limit or 1000,
            order_by="sequence",
        )

        events = [StoredEvent.model_validate(item["data"]) for item in items]
        if limit is not None:
            events = events[:limit]
        return events

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    async def subscribe_events(
        self, run_id: str, after_sequence: int = -1
    ) -> AsyncIterator[StoredEvent]:
        """Condition-based subscription with timeout fallback."""
        condition = self._get_or_create_condition(run_id)
        cursor = after_sequence

        while True:
            async with condition:
                batch = await self.query_events(run_id, after_sequence=cursor)
                if not batch:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            condition.wait(), timeout=self.poll_interval
                        )
                    continue

            for event in batch:
                yield event
                cursor = event.sequence
                if self._is_terminal_event(event):
                    return

    # ------------------------------------------------------------------
    # Tick journal
    # ------------------------------------------------------------------

    async def _next_tick_sequence(self, run_id: str) -> int:
        async with self._seq_lock:
            if run_id not in self._tick_sequences:
                self._tick_sequences[run_id] = await self._max_sequence(
                    self._ticks_collection, run_id
                )
            seq = self._tick_sequences[run_id] + 1
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
        await self._client.create(
            self._ticks_collection,
            stored.model_dump(mode="json"),
        )

    async def get_ticks(self, run_id: str) -> list[StoredTick]:
        page_size = 100
        all_items: list[dict[str, Any]] = []
        last_sequence = -1

        while True:
            filters: dict[str, Any] = {
                "run_id": {"eq": run_id},
                "sequence": {"gt": last_sequence},
            }
            page = await self._client.search(
                self._ticks_collection,
                filters,
                page_size=page_size,
                order_by="sequence",
            )
            all_items.extend(page)
            if len(page) < page_size:
                break
            last_sequence = page[-1]["data"]["sequence"]

        return [StoredTick.model_validate(item["data"]) for item in all_items]

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
        return AgentDataStateStore(
            client=self._client,
            run_id=run_id,
            state_type=state_type,
            collection=f"{self._collection}_state",
        )
