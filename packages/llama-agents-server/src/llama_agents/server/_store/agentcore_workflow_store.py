from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from typing import Any, List, cast

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.models import MetadataValue, OperatorType, StringValue
from bedrock_agentcore.memory.models.filters import EventMetadataFilter
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from pydantic import BaseModel
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.context.state_store import StateStore

from .abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
    StoredEvent,
    StoredTick,
)

DEFAULT_INTEGRATION_NAME = "llama-agents-server-agentcore"
HANDLERS_MEMORY_ID = "handlers"
EVENTS_MEMORY_ID = "events"
TICKS_MEMORY_ID = "ticks"
METADATA_KEYS = ("handler_id", "run_id", "workflow_name", "status")
IS_IDLE_KEY = "is_idle"
IS_IDLE_CONVERSION = {True: "yes", False: "no"}


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
        if dumped.get(key) is not None:
            val = cast(str, dumped.get(key))
            meta[key] = StringValue(stringValue=val)
    meta[IS_IDLE_KEY] = StringValue(
        stringValue=IS_IDLE_CONVERSION[handler.idle_since is not None]
    )
    return meta


def _query_to_metadata_filters(query: HandlerQuery) -> list[EventMetadataFilter]:
    filters: list[EventMetadataFilter] = []
    if query.run_id_in:
        for run_id in query.run_id_in:
            filters.append(
                {
                    "left": {"metadataKey": "run_id"},
                    "operator": OperatorType.EQUALS_TO,
                    "right": {"metadataValue": StringValue(stringValue=run_id)},
                }
            )
    if query.workflow_name_in:
        for workflow_name in query.workflow_name_in:
            filters.append(
                {
                    "left": {"metadataKey": "workflow_name"},
                    "operator": OperatorType.EQUALS_TO,
                    "right": {"metadataValue": StringValue(stringValue=workflow_name)},
                }
            )
    if query.status_in:
        for status in query.status_in:
            filters.append(
                {
                    "left": {"metadataKey": "status"},
                    "operator": OperatorType.EQUALS_TO,
                    "right": {"metadataValue": StringValue(stringValue=status)},
                }
            )
    if query.is_idle is not None:
        filters.append(
            {
                "left": {"metadataKey": IS_IDLE_KEY},
                "operator": OperatorType.EQUALS_TO,
                "right": {
                    "metadataValue": StringValue(
                        stringValue=IS_IDLE_CONVERSION[query.is_idle]
                    )
                },
            }
        )
    return filters


class AgentCoreWorkflowStore(AbstractWorkflowStore):
    def __init__(
        self,
        region_name: str | None = None,
        *,
        concurrent_operations: int = 5,
    ) -> None:
        self.region_name = region_name
        self.concurrent_operations = concurrent_operations
        self._client: MemoryClient | None = None
        self._current_handler_id: str | None = None

    def create_state_store(
        self,
        run_id: str,
        state_type: type[Any] | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> StateStore[Any]:
        raise NotImplementedError("no yet implemented")

    def _init_client(self) -> None:
        if self._client is None:
            self._client = MemoryClient(
                region_name=self.region_name,
                integration_source=DEFAULT_INTEGRATION_NAME,
            )

    async def update(self, handler: PersistentHandler) -> None:
        self._init_client()
        self._client = cast(MemoryClient, self._client)

        response = await asyncio.to_thread(
            self._client.create_blob_event,
            memory_id=HANDLERS_MEMORY_ID,
            actor_id=handler.handler_id,
            session_id=handler.handler_id,
            blob_data=_model_to_b64_str(handler),
            metadata=_handler_to_metadata(handler),
        )

        self._current_handler_id = handler.handler_id

        if response.get("eventId") is None:
            raise RuntimeError("AgentCore Memory did not produce an event ID")

    async def _get_handlers_by_handler_id(
        self, handler_id: str, filters: list[EventMetadataFilter]
    ) -> list[dict[str, Any]]:
        self._init_client()
        self._client = cast(MemoryClient, self._client)
        events = await asyncio.to_thread(
            self._client.list_events,
            memory_id=HANDLERS_MEMORY_ID,
            actor_id=handler_id,
            session_id=handler_id,
            event_metadata=filters,
        )
        return events

    async def _subquery(
        self, handler_id: str, filters: list[EventMetadataFilter]
    ) -> list[PersistentHandler]:
        events = await self._get_handlers_by_handler_id(handler_id, filters)
        handlers = []
        for event in events:
            payload = event.get("payload", [])
            data = payload[0].get("blob") if len(payload) > 0 else None
            if data is not None:
                handlers.append(cast(PersistentHandler, _b64_str_to_model(data)))
        return handlers

    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        filters = _query_to_metadata_filters(query)
        if query.handler_id_in:
            semaphore = asyncio.Semaphore(self.concurrent_operations)

            async def run_query(handler_id: str) -> list[PersistentHandler]:
                async with semaphore:
                    return await self._subquery(handler_id, filters)

            results = await asyncio.gather(
                *[run_query(handler_id) for handler_id in query.handler_id_in]
            )
            flattened: list[PersistentHandler] = []
            for result in results:
                flattened.extend(result)
            return flattened
        else:
            if not self._current_handler_id:
                raise RuntimeError(
                    "No current handler ID registered within the workflow store"
                )
            return await self._subquery(self._current_handler_id, filters)

    async def _subdelete(
        self, handler_id: str, filters: list[EventMetadataFilter]
    ) -> int:
        self._init_client()
        self._client = cast(MemoryClient, self._client)

        events = await self._get_handlers_by_handler_id(handler_id, filters)
        deleted = 0
        for event in events:
            id_ = event.get("eventId")
            payload = event.get("payload", [])
            data = payload[0].get("blob") if len(payload) > 0 else None
            if id_ is not None and data is not None:
                handler = cast(PersistentHandler, _b64_str_to_model(data))
                await asyncio.to_thread(
                    self._client.gmdp_client.delete_event,  # use data client directly as MemoryClient does not provide a delete_event method
                    eventId=id_,
                    memoryId=HANDLERS_MEMORY_ID,
                    actorId=handler.handler_id,
                    sessionId=handler.handler_id,
                )
                deleted += 1

        return deleted

    async def delete(self, query: HandlerQuery) -> int:
        filters = _query_to_metadata_filters(query)
        if query.handler_id_in:
            semaphore = asyncio.Semaphore(self.concurrent_operations)

            async def run_delete(handler_id: str) -> int:
                async with semaphore:
                    return await self._subdelete(handler_id, filters)

            results = await asyncio.gather(
                *[run_delete(handler_id) for handler_id in query.handler_id_in]
            )
            return sum(results)
        else:
            if not self._current_handler_id:
                raise RuntimeError(
                    "No current handler ID registered within the workflow store"
                )
            return await self._subdelete(self._current_handler_id, filters)

    async def _get_events_or_ticks_by_run_id(
        self,
        run_id: str,
        limit: int | None = None,
        get_ticks: bool = False,
    ) -> list[dict[str, Any]]:
        self._init_client()
        self._client = cast(MemoryClient, self._client)

        response = await asyncio.to_thread(
            self._client.list_events,
            memory_id=EVENTS_MEMORY_ID if not get_ticks else TICKS_MEMORY_ID,
            actor_id=run_id,
            session_id=run_id,
            max_results=limit or 100,
        )

        return response

    async def append_event(self, run_id: str, event: EventEnvelopeWithMetadata) -> None:
        self._init_client()
        self._client = cast(MemoryClient, self._client)
        sequence = await self._get_events_or_ticks_by_run_id(run_id)
        stored_event = StoredEvent(
            run_id=run_id,
            sequence=len(sequence) + 1,
            timestamp=datetime.now(),
            event=event,
        )
        response = await asyncio.to_thread(
            self._client.create_blob_event,
            memory_id=EVENTS_MEMORY_ID,
            actor_id=run_id,
            session_id=run_id,
            blob_data=_model_to_b64_str(stored_event),
        )

        if response.get("eventId") is None:
            raise RuntimeError("AgentCore Memory did not produce an event ID")

    async def query_events(
        self, run_id: str, after_sequence: int | None = None, limit: int | None = None
    ) -> list[StoredEvent]:
        events = await self._get_events_or_ticks_by_run_id(run_id, limit)
        stored_events: list[StoredEvent] = []
        for event in events:
            data = event.get("payload", {}).get("blob")
            if data is not None:
                stored_events.append(cast(StoredEvent, _b64_str_to_model(data)))
        if after_sequence:
            stored_events = [e for e in stored_events if e.sequence > after_sequence]
        return stored_events

    async def append_tick(self, run_id: str, tick_data: dict[str, Any]) -> None:
        self._init_client()
        self._client = cast(MemoryClient, self._client)
        sequence = await self._get_events_or_ticks_by_run_id(run_id, get_ticks=True)
        tick = StoredTick(
            run_id=run_id,
            tick_data=tick_data,
            sequence=len(sequence) + 1,
            timestamp=datetime.now(),
        )
        response = await asyncio.to_thread(
            self._client.create_blob_event,
            memory_id=TICKS_MEMORY_ID,
            actor_id=run_id,
            session_id=run_id,
            blob_data=_model_to_b64_str(tick),
        )
        if response.get("eventId") is None:
            raise RuntimeError("AgentCore Memory did not produce an event ID")

    async def get_ticks(self, run_id: str) -> list[StoredTick]:
        events = await self._get_events_or_ticks_by_run_id(run_id, get_ticks=True)
        stored_ticks: list[StoredTick] = []
        for event in events:
            payload = event.get("payload", [])
            data = payload[0].get("blob") if len(payload) > 0 else None
            if data is not None:
                stored_ticks.append(cast(StoredTick, _b64_str_to_model(data)))

        return stored_ticks
