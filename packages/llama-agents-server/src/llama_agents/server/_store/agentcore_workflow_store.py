from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import boto3
from typing_extensions import TypedDict

from .abstract_workflow_store import (
    AbstractWorkflowStore,
    PersistentHandler,
)

DEFAULT_SERVICE_NAME = "llama-agents-agentcore-workflow-store"


class BlobPayload(TypedDict):
    blob: dict[str, Any]


class AgentCoreWorkflowStore(AbstractWorkflowStore):
    def __init__(
        self,
        service_name: str = DEFAULT_SERVICE_NAME,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str | None = None,
    ) -> None:
        self._client: Any = None
        self.service_name = service_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self._event_id_to_handler: dict[str, PersistentHandler] = {}

    def _init_client(self) -> None:
        if self._client is None:
            self._client = boto3.client(
                self.service_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
            )

    async def update(self, handler: PersistentHandler) -> None:
        self._init_client()

        response = self._client.create_event(
            memoryId=handler.handler_id,
            actorId=handler.workflow_name,
            sessionId=handler.handler_id,
            payload=[BlobPayload(blob=handler.model_dump())],
            eventTimestamp=datetime.now(timezone.utc),
        )

        event_id = response.get("event", {}).get("eventId")

        if event_id is None:
            raise RuntimeError("AgentCore Memory did not produce an event ID")

        self._event_id_to_handler[event_id] = handler
