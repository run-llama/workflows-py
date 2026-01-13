from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Literal

from pydantic import (
    BaseModel,
    field_serializer,
    field_validator,
)

from workflows.context import JsonSerializer
from workflows.events import StopEvent

Status = Literal["running", "completed", "failed", "cancelled"]


@dataclass()
class HandlerQuery:
    # Matches if any of the handler_ids match
    handler_id_in: List[str] | None = None
    # Matches if any of the workflow_names match
    workflow_name_in: List[str] | None = None
    # Matches if the status flag matches
    status_in: List[Status] | None = None
    # True = only idle handlers, False = only non-idle handlers, None = all
    is_idle: bool | None = None


class PersistentHandler(BaseModel):
    handler_id: str
    workflow_name: str
    status: Status
    run_id: str | None = None
    error: str | None = None
    result: StopEvent | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    idle_since: datetime | None = None
    ctx: dict[str, Any] = {}

    @field_validator("result", mode="before")
    @classmethod
    def _parse_stop_event(cls, data: Any) -> StopEvent | None:
        if isinstance(data, StopEvent):
            return data
        elif isinstance(data, dict):
            deserialized = JsonSerializer().deserialize_value(data)
            if isinstance(deserialized, StopEvent):
                return deserialized
            else:
                return StopEvent(result=data)
        elif data is None:
            return None
        else:
            return StopEvent(result=data)

    @field_serializer("result", mode="plain")
    def _serialize_stop_event(self, data: StopEvent | None) -> Any:
        if data is None:
            return None
        result = JsonSerializer().serialize_value(data)
        return result


class AbstractWorkflowStore(ABC):
    @abstractmethod
    async def query(self, query: HandlerQuery) -> List[PersistentHandler]: ...

    @abstractmethod
    async def update(self, handler: PersistentHandler) -> None: ...

    @abstractmethod
    async def delete(self, query: HandlerQuery) -> int: ...
