# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""AgentDataStore — AbstractWorkflowStore backed by the LlamaCloud Agent Data API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dataclasses import field as dataclass_field
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


@dataclass
class _RunBuffer:
    """Per-run buffer for deferred event persistence."""

    events: list[StoredEvent] = dataclass_field(default_factory=list)
    flush_task: asyncio.Task[None] | None = None


class AgentDataStore(AbstractWorkflowStore):
    """Workflow store backed by the LlamaCloud Agent Data API.

    Optimized for streaming performance:
    - Same-process subscribers receive events via in-memory queues (no HTTP).
    - Event writes are batched/throttled to amortize API call costs.
    - Terminal events flush immediately for durability.
    - HTTP connections are reused across operations.

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
        flush_interval: float = 1.0,
    ) -> None:
        self._client = AgentDataClient(
            base_url=base_url,
            api_key=api_key,
            project_id=project_id,
            deployment_name=deployment_name,
        )
        self._collection = collection
        self.poll_interval = poll_interval
        self._flush_interval = flush_interval

        self._events_collection = f"{collection}_events"
        self._ticks_collection = f"{collection}_ticks"

        self._id_cache: LRUCache[str, str] = LRUCache(maxsize=256)
        self._locks = KeyedLock()

        self._event_sequences: dict[str, int] = {}
        self._tick_sequences: dict[str, int] = {}
        self._event_seq_lock: asyncio.Lock | None = None
        self._tick_seq_lock: asyncio.Lock | None = None

        self._subscriber_queues: dict[str, list[asyncio.Queue[StoredEvent | None]]] = {}
        self._buffers: dict[str, _RunBuffer] = {}
        self._state_stores: dict[str, AgentDataStateStore[Any]] = {}
        self._pending_ticks: dict[str, list[asyncio.Task[Any]]] = {}

    def _get_event_seq_lock(self) -> asyncio.Lock:
        if self._event_seq_lock is None:
            self._event_seq_lock = asyncio.Lock()
        return self._event_seq_lock

    def _get_tick_seq_lock(self) -> asyncio.Lock:
        if self._tick_seq_lock is None:
            self._tick_seq_lock = asyncio.Lock()
        return self._tick_seq_lock

    # ------------------------------------------------------------------
    # In-memory subscriber helpers
    # ------------------------------------------------------------------

    def _add_subscriber_queue(self, run_id: str) -> asyncio.Queue[StoredEvent | None]:
        """Create and register a new subscriber queue for a run."""
        queue: asyncio.Queue[StoredEvent | None] = asyncio.Queue()
        self._subscriber_queues.setdefault(run_id, []).append(queue)
        return queue

    def _remove_subscriber_queue(
        self, run_id: str, queue: asyncio.Queue[StoredEvent | None]
    ) -> None:
        """Unregister a subscriber queue. Cleans up the list if empty."""
        queues = self._subscriber_queues.get(run_id)
        if queues is not None:
            try:
                queues.remove(queue)
            except ValueError:
                pass
            if not queues:
                del self._subscriber_queues[run_id]

    def _broadcast_to_subscribers(self, run_id: str, event: StoredEvent) -> None:
        """Deliver an event to all in-memory subscriber queues for a run."""
        for queue in self._subscriber_queues.get(run_id, ()):
            queue.put_nowait(event)

    # ------------------------------------------------------------------
    # Buffered persistence helpers
    # ------------------------------------------------------------------

    def _get_or_create_buffer(self, run_id: str) -> _RunBuffer:
        if run_id not in self._buffers:
            self._buffers[run_id] = _RunBuffer()
        return self._buffers[run_id]

    async def _flush_buffer(self, run_id: str) -> None:
        """Persist all buffered events for a run via parallel API creates."""
        buf = self._buffers.get(run_id)
        if not buf or not buf.events:
            return
        events_to_write = buf.events[:]
        buf.events.clear()
        if buf.flush_task and not buf.flush_task.done():
            # Don't cancel if we're being called from within the flush task
            # itself (via _deferred_flush), as that would raise CancelledError
            # at the next await and abort the persist.
            if buf.flush_task is not asyncio.current_task():
                buf.flush_task.cancel()
            buf.flush_task = None
        results = await asyncio.gather(
            *[
                self._client.create(
                    self._events_collection,
                    e.model_dump(mode="json"),
                )
                for e in events_to_write
            ],
            return_exceptions=True,
        )
        failed = [
            events_to_write[i]
            for i, r in enumerate(results)
            if isinstance(r, BaseException)
        ]
        if failed:
            logger.warning(
                "Failed to flush %d/%d events for run %s",
                len(failed),
                len(events_to_write),
                run_id,
                exc_info=True,
            )
            # Re-queue failed events at the front for the next flush cycle
            if run_id in self._buffers:
                self._buffers[run_id].events[:0] = failed

    async def _deferred_flush(self, run_id: str) -> None:
        """Wait for the flush interval, then persist buffered events."""
        await asyncio.sleep(self._flush_interval)
        await self._flush_buffer(run_id)

    def _schedule_deferred_flush(self, run_id: str, buf: _RunBuffer) -> None:
        """Schedule a deferred flush if one isn't already pending."""
        if buf.flush_task is None or buf.flush_task.done():
            buf.flush_task = asyncio.create_task(self._deferred_flush(run_id))

    async def _cleanup_run(self, run_id: str) -> None:
        """Clean up buffer state and subscriber queues for a completed run."""
        # Await any in-flight tick writes before discarding tracking state
        await self._regroup_ticks(run_id)
        buf = self._buffers.pop(run_id, None)
        if buf and buf.flush_task and not buf.flush_task.done():
            buf.flush_task.cancel()
        # Signal subscribers that the run is done, then remove the key
        for queue in self._subscriber_queues.get(run_id, []):
            queue.put_nowait(None)
        self._subscriber_queues.pop(run_id, None)
        # Clean up sequence counters and cached state store
        self._event_sequences.pop(run_id, None)
        self._tick_sequences.pop(run_id, None)
        self._state_stores.pop(run_id, None)

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------

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
        async with self._get_event_seq_lock():
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

        self._broadcast_to_subscribers(run_id, stored)

        buf = self._get_or_create_buffer(run_id)
        buf.events.append(stored)

        if self._is_output_event(stored):
            await self._flush_buffer(run_id)
            if self._is_terminal_event(stored):
                await self._cleanup_run(run_id)
        else:
            self._schedule_deferred_flush(run_id, buf)

    async def query_events(
        self,
        run_id: str,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[StoredEvent]:
        await self._flush_buffer(run_id)

        filters: dict[str, Any] = {"run_id": {"eq": run_id}}
        if after_sequence is not None:
            filters["sequence"] = {"gte": after_sequence + 1}

        items = await self._client.search(
            self._events_collection,
            filters,
            page_size=limit or 1000,
            order_by="sequence",
        )

        return [StoredEvent.model_validate(item["data"]) for item in items]

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    async def subscribe_events(
        self, run_id: str, after_sequence: int = -1
    ) -> AsyncIterator[StoredEvent]:
        """In-memory queue-based subscription.

        Subscribes to the queue *before* running backfill to avoid losing
        events in the race window. Deduplicates by sequence number.
        """
        # Register queue before backfill to avoid race condition
        queue = self._add_subscriber_queue(run_id)
        try:
            cursor = after_sequence

            # Backfill: yield historical events already persisted
            backfill = await self.query_events(run_id, after_sequence=cursor)
            for event in backfill:
                yield event
                cursor = event.sequence
                if self._is_terminal_event(event):
                    return

            # Stream from in-memory queue
            while True:
                event = await queue.get()
                if event is None:
                    # Run completed, queue was signaled
                    return
                # Deduplicate: skip events already yielded in backfill
                if event.sequence <= cursor:
                    continue
                yield event
                cursor = event.sequence
                if self._is_terminal_event(event):
                    return
        finally:
            self._remove_subscriber_queue(run_id, queue)

    # ------------------------------------------------------------------
    # Tick journal
    # ------------------------------------------------------------------

    async def _next_tick_sequence(self, run_id: str) -> int:
        async with self._get_tick_seq_lock():
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

        # Fire-and-forget: tick creates run in the background so they don't
        # block the control loop.  Failures surface at _regroup_ticks time.
        task = asyncio.create_task(
            self._client.create(
                self._ticks_collection,
                stored.model_dump(mode="json"),
            )
        )
        pending = self._pending_ticks.setdefault(run_id, [])
        pending.append(task)
        # Prune completed tasks to avoid unbounded growth
        if len(pending) > 50:
            self._pending_ticks[run_id] = [t for t in pending if not t.done()]

    async def _regroup_ticks(self, run_id: str) -> None:
        """Wait for all in-flight tick creates to complete. Raises on failure."""
        tasks = self._pending_ticks.pop(run_id, [])
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]

    async def get_ticks(self, run_id: str) -> list[StoredTick]:
        await self._regroup_ticks(run_id)
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
        cached = self._state_stores.get(run_id)
        if cached is not None:
            return cached
        store = AgentDataStateStore(
            client=self._client,
            run_id=run_id,
            state_type=state_type,
            collection=f"{self._collection}_state",
        )
        self._state_stores[run_id] = store
        return store
