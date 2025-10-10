from __future__ import annotations

from typing import Any, Literal
from typing_extensions import TypedDict
from pydantic import TypeAdapter

# Shared protocol types between client and server

# Mirrors server.store Status
Status = Literal["running", "completed", "failed", "cancelled"]


class HandlerDict(TypedDict):
    handler_id: str
    workflow_name: str
    run_id: str | None
    error: str | None
    # result is workflow-defined; None if not completed
    result: Any | None
    status: Status
    started_at: str
    updated_at: str | None
    completed_at: str | None


class HandlersListResponse(TypedDict):
    handlers: list[HandlerDict]


class HealthResponse(TypedDict):
    status: Literal["healthy"]


class WorkflowsListResponse(TypedDict):
    workflows: list[str]


class SendEventResponse(TypedDict):
    status: Literal["sent"]


class CancelHandlerResponse(TypedDict):
    status: Literal["deleted", "cancelled"]


class WorkflowSchemaResponse(TypedDict):
    start: dict[str, Any]
    stop: dict[str, Any]


class WorkflowEventsListResponse(TypedDict):
    events: list[dict[str, Any]]


class WorkflowGraphResponse(TypedDict):
    graph: WorkflowGraphNodeEdges


class WorkflowGraphNode(TypedDict):
    id: str
    label: str
    node_type: str
    title: str | None
    event_type: str | None


class WorkflowGraphEdge(TypedDict):
    source: str
    target: str


class WorkflowGraphNodeEdges(TypedDict):
    nodes: list[WorkflowGraphNode]
    edges: list[WorkflowGraphEdge]


# Pydantic TypeAdapter validators for lightweight runtime validation/casting
HandlerDictValidator = TypeAdapter(HandlerDict)
HandlersListResponseValidator = TypeAdapter(HandlersListResponse)
HealthResponseValidator = TypeAdapter(HealthResponse)
WorkflowsListResponseValidator = TypeAdapter(WorkflowsListResponse)
SendEventResponseValidator = TypeAdapter(SendEventResponse)
CancelHandlerResponseValidator = TypeAdapter(CancelHandlerResponse)
WorkflowSchemaResponseValidator = TypeAdapter(WorkflowSchemaResponse)
WorkflowEventsListResponseValidator = TypeAdapter(WorkflowEventsListResponse)
WorkflowGraphResponseValidator = TypeAdapter(WorkflowGraphResponse)
