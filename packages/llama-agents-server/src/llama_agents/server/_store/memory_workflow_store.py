from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Dict, List

from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from workflows.context.state_store import DictState, InMemoryStateStore

from .abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
    StoredEvent,
    StoredTick,
)


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


class MemoryWorkflowStore(AbstractWorkflowStore):
    def __init__(self) -> None:
        self.handlers: Dict[str, PersistentHandler] = {}
        self.events: Dict[str, List[StoredEvent]] = {}
        self.ticks: Dict[str, List[StoredTick]] = {}
        self.state_stores: Dict[str, InMemoryStateStore[Any]] = {}
        self._conditions: Dict[str, asyncio.Condition] = {}

    def create_state_store(
        self, run_id: str, state_type: type[Any] | None = None
    ) -> InMemoryStateStore[Any]:
        if run_id not in self.state_stores:
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
        self.handlers[handler.handler_id] = handler

    async def delete(self, query: HandlerQuery) -> int:
        to_delete = [
            handler_id
            for handler_id, handler in list(self.handlers.items())
            if _matches_query(handler, query)
        ]
        for handler_id in to_delete:
            del self.handlers[handler_id]
        return len(to_delete)

    def _get_condition(self, run_id: str) -> asyncio.Condition:
        if run_id not in self._conditions:
            self._conditions[run_id] = asyncio.Condition()
        return self._conditions[run_id]

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
        condition = self._get_condition(run_id)
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
        condition = self._get_condition(run_id)
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
                    self._conditions.pop(run_id, None)
                    return
            # Before waiting, check if the run already has a terminal event
            # that we've already passed (e.g. cursor is beyond all events but
            # the last event was terminal). This prevents hanging when a late
            # subscriber joins after the run is fully complete.
            if all_events and self._is_terminal_event(all_events[-1]):
                self._conditions.pop(run_id, None)
                return
            # No new events — wait for the producer to notify
            async with condition:
                all_events = self.events.get(run_id, [])
                if len(all_events) <= cursor:
                    await condition.wait()
