from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Dict, List

from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from workflows.context.serializers import BaseSerializer
from workflows.context.state_store import DictState, InMemoryStateStore

from .abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
    StoredEvent,
    StoredTick,
    is_terminal_status,
)

logger = logging.getLogger(__name__)


def _matches_query(handler: PersistentHandler, query: HandlerQuery) -> bool:
    # Empty lists should match nothing (short-circuit)
    if query.handler_id_in is not None:
        if len(query.handler_id_in) == 0:
            return False
        if handler.handler_id not in query.handler_id_in:
            return False

    if query.run_id_in is not None:
        if len(query.run_id_in) == 0:
            return False
        if handler.run_id not in query.run_id_in:
            return False

    if query.workflow_name_in is not None:
        if len(query.workflow_name_in) == 0:
            return False
        if handler.workflow_name not in query.workflow_name_in:
            return False

    if query.status_in is not None:
        if len(query.status_in) == 0:
            return False
        if handler.status not in query.status_in:
            return False

    if query.is_idle is not None:
        handler_is_idle = handler.idle_since is not None
        if query.is_idle != handler_is_idle:
            return False

    return True


_GC_BUFFER_RATIO = 0.1
_GC_BUFFER_MIN = 10


class MemoryWorkflowStore(AbstractWorkflowStore):
    def __init__(self, max_completed: int | None = 1000) -> None:
        self.handlers: Dict[str, PersistentHandler] = {}
        self.events: Dict[str, List[StoredEvent]] = {}
        self.ticks: Dict[str, List[StoredTick]] = {}
        self.state_stores: Dict[str, InMemoryStateStore[Any]] = {}
        self._conditions: weakref.WeakValueDictionary[str, asyncio.Condition] = (
            weakref.WeakValueDictionary()
        )
        self.max_completed = max_completed
        self._terminal_count = 0

    def create_state_store(
        self,
        run_id: str,
        state_type: type[Any] | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> InMemoryStateStore[Any]:
        if run_id not in self.state_stores:
            if serialized_state is not None and serializer is not None:
                try:
                    self.state_stores[run_id] = InMemoryStateStore.from_dict(
                        serialized_state, serializer
                    )
                except Exception:
                    logger.warning("Failed to seed InMemoryStateStore", exc_info=True)
                    self.state_stores[run_id] = InMemoryStateStore(
                        state_type() if state_type else DictState()
                    )
            else:
                self.state_stores[run_id] = InMemoryStateStore(
                    state_type() if state_type else DictState()
                )
        return self.state_stores[run_id]

    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        return [
            handler
            for handler in self.handlers.values()
            if _matches_query(handler, query)
        ]

    async def update(self, handler: PersistentHandler) -> None:
        old = self.handlers.get(handler.handler_id)
        # When update_handler_status mutates the handler in-place before
        # calling update(), old *is* handler — the old status is gone.
        # Treat same-identity updates as a fresh insertion for counting.
        was_terminal = (
            old is not None and old is not handler and is_terminal_status(old.status)
        )
        self.handlers[handler.handler_id] = handler

        if is_terminal_status(handler.status):
            if not was_terminal:
                self._terminal_count += 1
            self._maybe_evict()
        elif was_terminal:
            self._terminal_count -= 1

    async def delete(self, query: HandlerQuery) -> int:
        to_delete = [
            handler
            for handler in self.handlers.values()
            if _matches_query(handler, query)
        ]
        for handler in to_delete:
            del self.handlers[handler.handler_id]
            if is_terminal_status(handler.status):
                self._terminal_count -= 1
        return len(to_delete)

    def _gc_threshold(self) -> int:
        """Return the terminal count at which a GC sweep triggers."""
        if self.max_completed is None:
            return 0  # unreachable when None, but keeps typing happy
        buffer = max(int(self.max_completed * _GC_BUFFER_RATIO), _GC_BUFFER_MIN)
        return self.max_completed + buffer

    def _maybe_evict(self) -> None:
        """Trigger a GC sweep only when the terminal count exceeds the threshold."""
        if self.max_completed is None:
            return
        if self._terminal_count >= self._gc_threshold():
            self._evict_oldest_completed()

    def _evict_oldest_completed(self) -> None:
        """Batch-evict terminal handlers back down to max_completed.

        Also cleans up associated events, ticks, and state stores for evicted
        handlers.
        """
        if self.max_completed is None:
            return

        terminal = [h for h in self.handlers.values() if is_terminal_status(h.status)]
        overflow = len(terminal) - self.max_completed
        if overflow <= 0:
            return

        # Sort by completed_at (None sorts earliest) then by handler_id for
        # deterministic tie-breaking.
        terminal.sort(key=lambda h: (h.completed_at or datetime.min, h.handler_id))
        to_evict = terminal[:overflow]

        for handler in to_evict:
            del self.handlers[handler.handler_id]
            run_id = handler.run_id
            if run_id is not None:
                self.events.pop(run_id, None)
                self.ticks.pop(run_id, None)
                self.state_stores.pop(run_id, None)
        self._terminal_count -= len(to_evict)

    def _get_or_create_condition(self, run_id: str) -> asyncio.Condition:
        """Get or create a condition for a run_id.

        The caller is responsible for holding a strong reference to the
        returned Condition for as long as it needs notifications.
        """
        cond = self._conditions.get(run_id)
        if cond is None:
            cond = asyncio.Condition()
            self._conditions[run_id] = cond
        return cond

    async def append_event(self, run_id: str, event: EventEnvelopeWithMetadata) -> None:
        if run_id not in self.events:
            self.events[run_id] = []
        existing = self.events[run_id]
        next_seq = (existing[-1].sequence + 1) if existing else 0
        stored = StoredEvent(
            run_id=run_id,
            sequence=next_seq,
            timestamp=datetime.now(timezone.utc),
            event=event,
        )
        existing.append(stored)
        condition = self._conditions.get(run_id)
        if condition is not None:
            async with condition:
                condition.notify_all()

    async def query_events(
        self,
        run_id: str,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> List[StoredEvent]:
        events = self.events.get(run_id, [])
        if after_sequence is not None:
            events = [e for e in events if e.sequence > after_sequence]
        if limit is not None:
            events = events[:limit]
        return events

    async def append_tick(self, run_id: str, tick_data: dict[str, Any]) -> None:
        if run_id not in self.ticks:
            self.ticks[run_id] = []
        existing = self.ticks[run_id]
        next_seq = (existing[-1].sequence + 1) if existing else 0
        stored = StoredTick(
            run_id=run_id,
            sequence=next_seq,
            timestamp=datetime.now(timezone.utc),
            tick_data=tick_data,
        )
        existing.append(stored)

    async def get_ticks(self, run_id: str) -> list[StoredTick]:
        return list(self.ticks.get(run_id, []))

    async def subscribe_events(
        self, run_id: str, after_sequence: int = -1
    ) -> AsyncIterator[StoredEvent]:
        """Condition-based subscription — no polling.

        Uses list-index cursoring rather than sequence-field cursoring to
        handle duplicate sequence numbers (which occur when multiple internal
        adapters share the same run_id).
        """
        condition = self._get_or_create_condition(run_id)
        # Determine starting index: skip events with sequence <= after_sequence
        all_events = self.events.get(run_id, [])
        if after_sequence >= 0:
            cursor = 0
            for i, e in enumerate(all_events):
                if e.sequence <= after_sequence:
                    cursor = i + 1
            # cursor is now the index of the first event to yield
        else:
            cursor = 0

        while True:
            all_events = self.events.get(run_id, [])
            batch = all_events[cursor:]
            for event in batch:
                yield event
                cursor += 1
                if self._is_terminal_event(event):
                    return
            # Before waiting, check if the run already has a terminal event
            # that we've already passed (e.g. cursor is beyond all events but
            # the last event was terminal). This prevents hanging when a late
            # subscriber joins after the run is fully complete.
            if all_events and self._is_terminal_event(all_events[-1]):
                return
            # No new events — wait for the producer to notify
            async with condition:
                all_events = self.events.get(run_id, [])
                if len(all_events) <= cursor:
                    await condition.wait()
