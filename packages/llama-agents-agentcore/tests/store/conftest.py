import base64
import uuid
from datetime import datetime
from typing import Any, TypedDict, cast

from bedrock_agentcore.memory.models import MetadataValue
from bedrock_agentcore.memory.models.filters import EventMetadataFilter, OperatorType
from llama_agents.agentcore.store._utils import ActivityStatus
from llama_agents.agentcore.store.agentcore_state_store import STATE_STORE_ACTOR_ID
from pydantic import BaseModel
from workflows.context.serializers import BaseSerializer

OPERATOR_MAPPINGS = {
    OperatorType.EQUALS_TO.value: lambda x, y: (
        x == y
    ),  # field value is equal to target value
    OperatorType.EXISTS.value: lambda x, y: x in y,  # field exists in metadata
    OperatorType.NOT_EXISTS.value: lambda x, y: (
        x not in y
    ),  # field does not exist in metadata
}


def is_valid_uuid(uuid_to_test: str) -> bool:
    try:
        uuid.UUID(uuid_to_test, version=4)
    except ValueError:
        return False
    return True


class BlobPayload(TypedDict):
    blob: Any


class MockEvent(TypedDict):
    eventId: str
    payload: list[BlobPayload]
    eventTimestamp: datetime
    metadata: dict[str, MetadataValue] | None


class MockBotoClient:
    def __init__(self) -> None:
        self.events: list[MockEvent] = []
        self.deleted_events: list[str] = []
        self.gotten_events: list[str] = []
        self.without_data = "without-data"

    def get_event(
        self, eventId: str, memoryId: str, actorId: str, sessionId: str
    ) -> dict[str, MockEvent]:
        events = [event for event in self.events if event["eventId"] == eventId]
        if len(events) > 0:
            if eventId == self.without_data:
                ev = events[0].copy()
                ev["payload"] = []
                return {"event": ev}
            if not is_valid_uuid(eventId):
                self.gotten_events.append(eventId)
            return {"event": events[0]}
        return {}

    def delete_event(
        self, eventId: str, memoryId: str, actorId: str, sessionId: str
    ) -> None:
        self.events = [event for event in self.events if event["eventId"] != eventId]
        self.deleted_events.append(eventId)

    def _append_event(
        self,
        event: MockEvent,
    ) -> None:
        self.events.append(event)


class MockMemoryClient:
    def __init__(
        self, region_name: str | None = None, integration_source: str | None = None
    ):
        self.region_name = region_name
        self.integration_source = integration_source
        self.events_count = 0
        self.events: list[MockEvent] = []
        self._boto_client = MockBotoClient()
        self.created_memories: list[str] = []
        self.checked_state_store: bool = False
        self.memories_failed: bool = False
        self.memories_pending: int = 0

    def create_or_get_memory(
        self,
        name: str,
        strategies: list[dict[str, Any]] | None = None,
        description: str | None = None,
        event_expiry_days: int = 90,
        memory_execution_role_arn: str | None = None,
    ) -> dict[str, Any]:
        self.created_memories.append(name)
        return {
            "memoryId": name,
            "strategies": strategies,
            "description": description,
            "eventExpiryDays": event_expiry_days,
            "memoryExecutionRoleArn": memory_execution_role_arn,
        }

    def create_blob_event(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        blob_data: Any,
        event_timestamp: datetime | None = None,
        branch: dict[str, str] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> dict[str, Any]:
        self.events_count += 1
        event: MockEvent = {
            "eventId": f"{memory_id}-{actor_id}-{session_id}-{self.events_count}",
            "payload": [{"blob": blob_data}],
            "eventTimestamp": event_timestamp or datetime.now(),
            "metadata": metadata,
        }
        self.events.append(event)
        self._boto_client._append_event(event)
        return cast(dict[str, Any], event)

    def _filter_events(
        self, events: list[MockEvent], filters: list[EventMetadataFilter]
    ) -> list[MockEvent]:
        filtered_events: list[MockEvent] = []
        for event in events:
            truths = []
            for filter in filters:
                field = filter["left"]["metadataKey"]
                operator = str(filter["operator"])
                if operator == OperatorType.EQUALS_TO.value:
                    right_expression = filter["right"]
                    assert right_expression is not None
                    target_value = right_expression["metadataValue"]["stringValue"]
                    event_meta = event["metadata"]
                    if event_meta is not None:
                        value = event_meta.get(field)
                        if value is None:
                            truths.append(False)
                        else:
                            truths.append(
                                OPERATOR_MAPPINGS[operator](
                                    value["stringValue"], target_value
                                )
                            )
                    else:
                        truths.append(False)
                else:
                    event_meta = event.get("metadata")
                    if event_meta is not None:
                        truths.append(OPERATOR_MAPPINGS[operator](field, event_meta))
                    else:
                        truths.append(False)
            if any(t for t in truths):
                filtered_events.append(event)
        return filtered_events

    def get_memory_status(
        self,
        memory_id: str,
    ) -> ActivityStatus:
        if self.memories_failed:
            return "FAILED"
        if self.memories_pending > 0:
            self.memories_pending -= 1
            return "CREATING"
        return "ACTIVE"

    def list_events(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        branch_name: str | None = None,
        include_parent_branches: bool = False,
        event_metadata: list[EventMetadataFilter] | None = None,
        max_results: int = 100,
        include_payload: bool = True,
    ) -> list[MockEvent]:
        self._sync_with_gmdp_client()
        if actor_id == STATE_STORE_ACTOR_ID:
            self.checked_state_store = True
        events = [
            event
            for event in self.events
            if event["eventId"].startswith(f"{memory_id}-{actor_id}-{session_id}-")
        ]
        if event_metadata:
            events = self._filter_events(events, event_metadata)
        return events[:max_results]

    def _sync_with_gmdp_client(self) -> None:
        if self._boto_client.deleted_events:
            self.events = [
                event
                for event in self.events
                if event["eventId"] not in self._boto_client.deleted_events
            ]

    @property
    def gmdp_client(self) -> MockBotoClient:
        return self._boto_client


def serialize_state(serializer: BaseSerializer, state: Any) -> str:
    serialized = serializer.serialize(state).encode("utf-8")
    return base64.b64encode(serialized).decode("utf-8")


def make_data_event(
    state_data: BaseModel, serializer: BaseSerializer, event_id: str | None = None
) -> tuple[str, MockEvent]:
    event_id = event_id or str(uuid.uuid4())
    serialized = serializer.serialize(state_data).encode("utf-8")
    blob = base64.b64encode(serialized).decode("utf-8")
    return event_id, MockEvent(
        eventId=event_id,
        payload=[BlobPayload(blob=blob)],
        eventTimestamp=datetime.now(),
        metadata={},
    )
