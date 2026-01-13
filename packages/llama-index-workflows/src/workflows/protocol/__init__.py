from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from workflows.protocol.serializable_events import EventEnvelopeWithMetadata
from workflows.representation import WorkflowGraph

# Shared protocol types between client and server

# Mirrors server.store Status
Status = Literal["running", "completed", "failed", "cancelled"]


def is_status_completed(status: Status) -> bool:
    return status in {"completed", "failed", "cancelled"}


class HandlerData(BaseModel):
    handler_id: str
    workflow_name: str
    run_id: str | None
    error: str | None
    result: EventEnvelopeWithMetadata | None
    status: Status
    started_at: str
    updated_at: str | None
    completed_at: str | None


class HandlersListResponse(BaseModel):
    handlers: list[HandlerData]


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    loaded_workflows: int = Field(
        description="Number of workflow handlers currently loaded in memory"
    )
    active_workflows: int = Field(
        description="Number of workflow handlers that are active (not idle)"
    )
    idle_workflows: int = Field(description="Number of workflow handlers that are idle")


class WorkflowsListResponse(BaseModel):
    workflows: list[str]


class SendEventResponse(BaseModel):
    status: Literal["sent"]


class CancelHandlerResponse(BaseModel):
    status: Literal["deleted", "cancelled"]


class WorkflowSchemaResponse(BaseModel):
    start: dict[str, Any]
    stop: dict[str, Any]


class WorkflowEventsListResponse(BaseModel):
    events: list[dict[str, Any]]


class WorkflowGraphResponse(BaseModel):
    graph: WorkflowGraph


__all__ = [
    "Status",
    "is_status_completed",
    "HandlerData",
    "HandlersListResponse",
    "HealthResponse",
    "WorkflowsListResponse",
    "SendEventResponse",
    "CancelHandlerResponse",
    "WorkflowSchemaResponse",
    "WorkflowEventsListResponse",
    "WorkflowGraphResponse",
]
