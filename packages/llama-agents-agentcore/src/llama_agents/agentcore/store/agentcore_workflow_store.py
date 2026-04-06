from __future__ import annotations

import asyncio
import base64
import functools
from datetime import datetime
from typing import Any, List, cast

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.models import MetadataValue, OperatorType, StringValue
from bedrock_agentcore.memory.models.filters import EventMetadataFilter
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from llama_agents.server._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
    StoredEvent,
    StoredTick,
)
from pydantic import BaseModel
from workflows.context.serializers import BaseSerializer, JsonSerializer

from ._utils import check_memory_activity, create_memory
from .agentcore_state_store import AgentCoreStateStore

DEFAULT_INTEGRATION_NAME = "llama-agents-server-agentcore"
HANDLERS_ACTOR_ID = "handlers-actor-llamaindex"
HANDLERS_SESSION_ID = "handlers-session-llamaindex"
METADATA_KEYS = ("handler_id", "run_id", "workflow_name", "status")
IS_IDLE_KEY = "is_idle"
IS_IDLE_CONVERSION = {True: "yes", False: "no"}
HANDLERS_MEMORY_NAME = "handlers_memory_llamaindex"
EVENTS_MEMORY_NAME = "events_memory_llamaindex"
TICKS_MEMORY_NAME = "ticks_memory_llamaindex"
AGENTCORE_API_CONCURRENCY_LIMIT = 100


def _model_to_b64_str(model: BaseModel) -> str:
    dumped = JsonSerializer().serialize(model).encode("utf-8")
    return base64.b64encode(dumped).decode("utf-8")


def _b64_str_to_model(b64_str: str) -> BaseModel:
    decoded = base64.b64decode(b64_str).decode("utf-8")
    return JsonSerializer().deserialize(decoded)


def _handler_to_metadata(handler: PersistentHandler) -> dict[str, MetadataValue]:
    meta: dict[str, MetadataValue] = {}
    dumped = handler.model_dump()
    for key in METADATA_KEYS:
        val = dumped.get(key) or ""
        meta[key] = StringValue(stringValue=val)
    meta[IS_IDLE_KEY] = StringValue(
        stringValue=IS_IDLE_CONVERSION[handler.idle_since is not None]
    )
    return meta


def _query_to_metadata_filters(query: HandlerQuery) -> list[EventMetadataFilter]:
    filters: list[EventMetadataFilter] = []
    if query.handler_id_in:
        for handler_id in query.handler_id_in:
            filters.append(
                {
                    "left": {"metadataKey": "handler_id"},
                    "operator": OperatorType.EQUALS_TO.value,  # type: ignore
                    "right": {"metadataValue": StringValue(stringValue=handler_id)},
                }  # type: ignore
            )
    if query.run_id_in:
        for run_id in query.run_id_in:
            filters.append(
                {
                    "left": {"metadataKey": "run_id"},
                    "operator": OperatorType.EQUALS_TO.value,  # type: ignore
                    "right": {"metadataValue": StringValue(stringValue=run_id)},
                }  # type: ignore
            )
    if query.workflow_name_in:
        for workflow_name in query.workflow_name_in:
            filters.append(
                {
                    "left": {"metadataKey": "workflow_name"},
                    "operator": OperatorType.EQUALS_TO.value,  # type: ignore
                    "right": {"metadataValue": StringValue(stringValue=workflow_name)},
                }  # type: ignore
            )
    if query.status_in:
        for status in query.status_in:
            filters.append(
                {
                    "left": {"metadataKey": "status"},
                    "operator": OperatorType.EQUALS_TO.value,  # type: ignore
                    "right": {"metadataValue": StringValue(stringValue=status)},
                }  # type: ignore
            )
    if query.is_idle is not None:
        filters.append(
            {
                "left": {"metadataKey": IS_IDLE_KEY},
                "operator": OperatorType.EQUALS_TO.value,  # type: ignore
                "right": {
                    "metadataValue": StringValue(
                        stringValue=IS_IDLE_CONVERSION[query.is_idle]
                    )
                },
            }  # type: ignore
        )
    return filters


class AgentCoreWorkflowStore(AbstractWorkflowStore):
    def __init__(
        self,
        region_name: str | None = None,
    ) -> None:
        self._client: MemoryClient = MemoryClient(
            region_name=region_name,
            integration_source=DEFAULT_INTEGRATION_NAME,
        )
        self._handlers_memory_id: str | None = None
        self._events_memory_id: str | None = None
        self._ticks_memory_id: str | None = None
        self._memories_inited = False
        self._events_counter_cache: dict[str, int] = {}
        self._ticks_counter_cache: dict[str, int] = {}
        self._ticks_tracker: dict[str, list[asyncio.Task[None]]] = {}
        self._events_tracker: dict[str, list[asyncio.Task[None]]] = {}
        self._run_locks: dict[str, asyncio.Lock] = {}

    @property
    def handlers_memory_id(self) -> str:
        if self._handlers_memory_id is None:
            raise ValueError("Cannot access a None property as string")
        return self._handlers_memory_id

    @functools.cached_property
    def _events_lock(self) -> asyncio.Lock:
        return asyncio.Lock()

    @functools.cached_property
    def _ticks_lock(self) -> asyncio.Lock:
        return asyncio.Lock()

    @functools.cached_property
    def _delete_semaphore(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(AGENTCORE_API_CONCURRENCY_LIMIT)

    @functools.cached_property
    def _append_semaphore(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(AGENTCORE_API_CONCURRENCY_LIMIT)

    @property
    def events_memory_id(self) -> str:
        if self._events_memory_id is None:
            raise ValueError("Cannot access a None property as string")
        return self._events_memory_id

    @property
    def ticks_memory_id(self) -> str:
        if self._ticks_memory_id is None:
            raise ValueError("Cannot access a None property as string")
        return self._ticks_memory_id

    async def _create_memories(self) -> None:
        if not self._memories_inited:
            create_tasks = [
                create_memory(self._client, HANDLERS_MEMORY_NAME, "handlers"),
                create_memory(self._client, EVENTS_MEMORY_NAME, "events"),
                create_memory(self._client, TICKS_MEMORY_NAME, "ticks"),
            ]
            results = await asyncio.gather(*create_tasks)
            self._handlers_memory_id = results[0]
            self._events_memory_id = results[1]
            self._ticks_memory_id = results[2]
            check_tasks = [
                check_memory_activity(self._client, self.handlers_memory_id),
                check_memory_activity(self._client, self.events_memory_id),
                check_memory_activity(self._client, self.ticks_memory_id),
            ]
            checks = await asyncio.gather(*check_tasks)
            if any(not c for c in checks):
                map: dict[int, str] = {0: "handlers", 1: "events", 2: "ticks"}
                failed: list[str] = []
                for i in range(len(checks)):
                    if not checks[i]:
                        failed.append(map[i])
                raise RuntimeError(
                    f"Failed to provision memory for {', '.join(failed)}"
                )
            self._memories_inited = True

    def create_state_store(
        self,
        run_id: str,
        state_type: type[Any] | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> AgentCoreStateStore[Any]:
        state = AgentCoreStateStore(self._client, run_id, state_type, serializer)
        if serialized_state:
            state._seed_from_serialized(serialized_state)
        else:
            state._init_state()
        return state

    def _get_run_lock(self, run_id: str) -> asyncio.Lock:
        if run_id not in self._run_locks:
            self._run_locks[run_id] = asyncio.Lock()
        return self._run_locks[run_id]

    async def _append_one_tick_or_event(
        self,
        run_id: str,
        data: str,
        memory_id: str,
    ) -> None:
        async with self._append_semaphore:
            await asyncio.to_thread(
                self._client.create_blob_event,
                memory_id=memory_id,
                actor_id=run_id,
                session_id=run_id,
                blob_data=data,
            )

    def _track_pending(
        self,
        run_id: str,
        pending: dict[str, list[asyncio.Task[None]]],
        data: str,
        memory_id: str,
    ) -> None:
        lock = self._get_run_lock(run_id)
        if lock.locked():  # cleanup in process, cannot add more tasks
            return
        task = asyncio.create_task(
            self._append_one_tick_or_event(run_id, data, memory_id)
        )
        tasks = pending.setdefault(run_id, [])
        tasks.append(task)
        if len(tasks) > 50:
            pending[run_id] = [t for t in tasks if not t.done()]

    @staticmethod
    async def _regroup(
        pending: dict[str, list[asyncio.Task[None]]], run_id: str
    ) -> None:
        """Await all in-flight tasks for a run. Raises the first error."""
        tasks = pending.pop(run_id, [])
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]

    async def _regroup_events(self, run_id: str) -> None:
        await self._regroup(self._events_tracker, run_id)

    async def _regroup_ticks(self, run_id: str) -> None:
        await self._regroup(self._ticks_tracker, run_id)

    async def after_tick(self, run_id: str) -> None:
        """Gather all in-flight tick and event writes for a run."""
        await self._regroup_ticks(run_id)
        await self._regroup_events(run_id)

    async def _cleanup_run(self, run_id: str) -> None:
        """Clean up pending writes and subscriber queues for a completed run."""
        async with self._get_run_lock(run_id):
            await self._regroup_ticks(run_id)
            await self._regroup_events(run_id)
        async with self._events_lock:
            self._events_counter_cache.pop(run_id, 0)
        async with self._ticks_lock:
            self._ticks_counter_cache.pop(run_id, 0)

    async def update(self, handler: PersistentHandler) -> None:
        await self._create_memories()
        await asyncio.to_thread(
            self._client.create_blob_event,
            memory_id=self.handlers_memory_id,
            actor_id=HANDLERS_ACTOR_ID,
            session_id=HANDLERS_SESSION_ID,
            blob_data=_model_to_b64_str(handler),
            metadata=_handler_to_metadata(handler),
        )

    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        await self._create_memories()
        filters = _query_to_metadata_filters(query)
        events = await asyncio.to_thread(
            self._client.list_events,
            memory_id=self.handlers_memory_id,
            actor_id=HANDLERS_ACTOR_ID,
            session_id=HANDLERS_SESSION_ID,
            event_metadata=filters,
        )
        handlers = []
        for event in events:
            payload = event.get("payload", [])
            data = payload[0].get("blob") if len(payload) > 0 else None
            if data is not None:
                handlers.append(cast(PersistentHandler, _b64_str_to_model(data)))
        return handlers

    async def delete(self, query: HandlerQuery) -> int:
        await self._create_memories()
        filters = _query_to_metadata_filters(query)
        events = await asyncio.to_thread(
            self._client.list_events,
            memory_id=self.handlers_memory_id,
            actor_id=HANDLERS_ACTOR_ID,
            session_id=HANDLERS_SESSION_ID,
            event_metadata=filters,
        )
        tasks = []
        for event in events:
            id_ = event.get("eventId")
            if id_ is not None:
                tasks.append(self._delete_one(id_))
        await asyncio.gather(*tasks)

        return len(tasks)

    async def _delete_one(self, event_id: str) -> None:
        async with self._delete_semaphore:
            await asyncio.to_thread(
                self._client.gmdp_client.delete_event,  # use data client directly as MemoryClient does not provide a delete_event method
                eventId=event_id,
                memoryId=self.handlers_memory_id,
                actorId=HANDLERS_ACTOR_ID,
                sessionId=HANDLERS_SESSION_ID,
            )

    async def _get_events_or_ticks_by_run_id(
        self,
        run_id: str,
        limit: int | None = None,
        get_ticks: bool = False,
    ) -> list[dict[str, Any]]:
        await self._create_memories()
        response = await asyncio.to_thread(
            self._client.list_events,
            memory_id=self.events_memory_id if not get_ticks else self.ticks_memory_id,
            actor_id=run_id,
            session_id=run_id,
            max_results=limit or 100,
        )

        return response

    async def append_event(self, run_id: str, event: EventEnvelopeWithMetadata) -> None:
        async with self._events_lock:
            if run_id not in self._events_counter_cache:
                sequence = await self._get_events_or_ticks_by_run_id(
                    run_id
                )  # create_memories is called here, no need to duplicate
                self._events_counter_cache[run_id] = len(sequence)
            self._events_counter_cache[run_id] += 1
            stored_event = StoredEvent(
                run_id=run_id,
                sequence=self._events_counter_cache[run_id],
                timestamp=datetime.now(),
                event=event,
            )
        self._track_pending(
            run_id,
            self._events_tracker,
            _model_to_b64_str(stored_event),
            self.events_memory_id,
        )
        if self._is_terminal_event(stored_event):
            await self._cleanup_run(run_id)

    async def query_events(
        self, run_id: str, after_sequence: int | None = None, limit: int | None = None
    ) -> list[StoredEvent]:
        await self._regroup_events(run_id)
        events = await self._get_events_or_ticks_by_run_id(
            run_id
        )  # create_memories is called here, no need to duplicate
        stored_events: list[StoredEvent] = []
        for event in events:
            payload = event.get("payload", [])
            data = payload[0].get("blob") if len(payload) > 0 else None
            if data is not None:
                stored_events.append(cast(StoredEvent, _b64_str_to_model(data)))
        sorted_stored_events = sorted(stored_events, key=lambda x: x.sequence)
        if after_sequence:
            sorted_stored_events = [
                e for e in sorted_stored_events if e.sequence > after_sequence
            ]
        if limit:
            sorted_stored_events = sorted_stored_events[:limit]
        return sorted_stored_events

    async def append_tick(self, run_id: str, tick_data: dict[str, Any]) -> None:
        async with self._ticks_lock:
            if run_id not in self._ticks_counter_cache:
                sequence = await self._get_events_or_ticks_by_run_id(
                    run_id, get_ticks=True
                )  # create_memories is called here, no need to duplicate
                self._ticks_counter_cache[run_id] = len(sequence)
            self._ticks_counter_cache[run_id] += 1
            tick = StoredTick(
                run_id=run_id,
                tick_data=tick_data,
                sequence=self._ticks_counter_cache[run_id],
                timestamp=datetime.now(),
            )
        self._track_pending(
            run_id,
            self._ticks_tracker,
            _model_to_b64_str(tick),
            self.ticks_memory_id,
        )

    async def get_ticks(self, run_id: str) -> list[StoredTick]:
        await self._regroup_ticks(run_id)
        events = await self._get_events_or_ticks_by_run_id(
            run_id, get_ticks=True
        )  # create_memories is called here, no need to duplicate
        stored_ticks: list[StoredTick] = []
        for event in events:
            payload = event.get("payload", [])
            data = payload[0].get("blob") if len(payload) > 0 else None
            if data is not None:
                stored_ticks.append(cast(StoredTick, _b64_str_to_model(data)))
        sorted_ticks = sorted(stored_ticks, key=lambda x: x.sequence)
        return sorted_ticks
