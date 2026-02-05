# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import AsyncGenerator, cast

from llama_agents.client.protocol import (
    CancelHandlerResponse,
    HandlerData,
    HandlersListResponse,
    HealthResponse,
    SendEventResponse,
    WorkflowEventsListResponse,
    WorkflowGraphResponse,
    WorkflowSchemaResponse,
    is_status_completed,
)
from llama_agents.client.protocol.serializable_events import (
    EventEnvelope,
    EventEnvelopeWithMetadata,
    EventValidationError,
)
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.schemas import SchemaGenerator
from starlette.staticfiles import StaticFiles
from workflows import Context, Workflow
from workflows.events import InternalDispatchEvent, StartEvent
from workflows.representation import get_workflow_representation
from workflows.runtime.types.ticks import TickAddEvent
from workflows.utils import _nanoid as nanoid

from ._server_runtime import ServerExternalAdapter
from ._service import _WorkflowService
from .abstract_workflow_store import (
    HandlerQuery,
    PersistentHandler,
    Status,
)

logger = logging.getLogger()


_DEFAULT_ASSETS_PATH = Path(__file__).parent / "static"


def _handler_data_from_persistent(persistent: PersistentHandler) -> HandlerData:
    """Build HandlerData from a PersistentHandler record."""
    return HandlerData(
        handler_id=persistent.handler_id,
        workflow_name=persistent.workflow_name,
        run_id=persistent.run_id,
        status=persistent.status,
        started_at=persistent.started_at.isoformat()
        if persistent.started_at is not None
        else datetime.now(timezone.utc).isoformat(),
        updated_at=persistent.updated_at.isoformat()
        if persistent.updated_at is not None
        else None,
        completed_at=persistent.completed_at.isoformat()
        if persistent.completed_at is not None
        else None,
        error=persistent.error,
        result=EventEnvelopeWithMetadata.from_event(persistent.result)
        if persistent.result is not None
        else None,
    )


def _build_handler_data(
    handler_id: str,
    adapter: ServerExternalAdapter,
    service: _WorkflowService,
) -> HandlerData:
    """Build HandlerData from an active adapter."""
    workflow_name = service.get_workflow_name_for_handler(handler_id) or ""
    for decorator in service._decorators.values():
        meta = decorator.get_run_metadata(handler_id)
        if meta is not None:
            status: Status = "running"
            error: str | None = None
            result = None

            try:
                if adapter._inner_done():
                    try:
                        result = adapter._inner_result()
                        status = "completed"
                        meta.completed_at = meta.updated_at
                    except asyncio.CancelledError:
                        status = "cancelled"
                        meta.completed_at = meta.updated_at
                    except Exception as e:
                        status = "failed"
                        error = str(e)
                        meta.completed_at = meta.updated_at
            except Exception:
                pass

            return HandlerData(
                handler_id=handler_id,
                workflow_name=meta.workflow_name,
                run_id=adapter.run_id,
                status=status,
                started_at=meta.started_at.isoformat(),
                updated_at=meta.updated_at.isoformat(),
                completed_at=meta.completed_at.isoformat()
                if meta.completed_at is not None
                else None,
                error=error,
                result=EventEnvelopeWithMetadata.from_event(result)
                if result is not None
                else None,
            )

    return HandlerData(
        handler_id=handler_id,
        workflow_name=workflow_name,
        run_id=adapter.run_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )


class _WorkflowAPI:
    def __init__(
        self,
        service: _WorkflowService,
        *,
        middleware: list[Middleware] | None = None,
        assets_path: Path = _DEFAULT_ASSETS_PATH,
    ) -> None:
        self._service = service

        middleware = middleware or [
            Middleware(
                CORSMiddleware,  # type: ignore[arg-type]
                # regex echoes the origin header back, which some browsers require (rather than "*") when credentials are required
                allow_origin_regex=".*",
                allow_methods=["*"],
                allow_headers=["*"],
                allow_credentials=True,
            )
        ]

        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
            await self._service.start()
            try:
                yield
            finally:
                await self._service.stop()

        self.app = Starlette(
            routes=self._routes(),
            middleware=middleware,
            lifespan=lifespan,
        )
        self.app.mount(
            "/", app=StaticFiles(directory=assets_path, html=True), name="ui"
        )

    def _routes(self) -> list[Route]:
        return [
            Route("/workflows", self._list_workflows, methods=["GET"]),
            Route("/workflows/{name}/run", self._run_workflow, methods=["POST"]),
            Route(
                "/workflows/{name}/run-nowait",
                self._run_workflow_nowait,
                methods=["POST"],
            ),
            Route(
                "/workflows/{name}/schema",
                self._get_events_schema,
                methods=["GET"],
            ),
            Route(
                "/results/{handler_id}",
                self._get_workflow_result,
                methods=["GET"],
            ),
            Route("/events/{handler_id}", self._stream_events, methods=["GET"]),
            Route("/events/{handler_id}", self._post_event, methods=["POST"]),
            Route("/health", self._health_check, methods=["GET"]),
            Route("/handlers", self._get_handlers, methods=["GET"]),
            Route(
                "/handlers/{handler_id}",
                self._get_workflow_handler,
                methods=["GET"],
            ),
            Route(
                "/handlers/{handler_id}/cancel",
                self._cancel_handler,
                methods=["POST"],
            ),
            Route(
                "/workflows/{name}/representation",
                self._get_workflow_representation,
                methods=["GET"],
            ),
            Route(
                "/workflows/{name}/events",
                self._list_workflow_events,
                methods=["GET"],
            ),
        ]

    def openapi_schema(self) -> dict:
        gen = SchemaGenerator(
            {
                "openapi": "3.0.0",
                "info": {
                    "title": "Workflows API",
                    "version": version("llama-index-workflows"),
                },
                "components": {
                    "schemas": {
                        "EventEnvelopeWithMetadata": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "object"},
                                "types": {"type": "array", "items": {"type": "string"}},
                                "type": {"type": "string"},
                                "qualified_name": {"type": "string"},
                            },
                            "required": ["value", "type"],
                        },
                        "Handler": {
                            "type": "object",
                            "properties": {
                                "handler_id": {"type": "string"},
                                "workflow_name": {"type": "string"},
                                "run_id": {"type": "string", "nullable": True},
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "running",
                                        "completed",
                                        "failed",
                                        "cancelled",
                                    ],
                                },
                                "started_at": {"type": "string", "format": "date-time"},
                                "updated_at": {
                                    "type": "string",
                                    "format": "date-time",
                                    "nullable": True,
                                },
                                "completed_at": {
                                    "type": "string",
                                    "format": "date-time",
                                    "nullable": True,
                                },
                                "error": {"type": "string", "nullable": True},
                                "result": {
                                    "description": "Workflow result value",
                                    "oneOf": [
                                        {
                                            "$ref": "#/components/schemas/EventEnvelopeWithMetadata"
                                        },
                                        {"type": "null"},
                                    ],
                                },
                            },
                            "required": [
                                "handler_id",
                                "workflow_name",
                                "status",
                                "started_at",
                            ],
                        },
                        "HandlersList": {
                            "type": "object",
                            "properties": {
                                "handlers": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Handler"},
                                }
                            },
                            "required": ["handlers"],
                        },
                    }
                },
            }
        )

        return gen.get_schema(self.app.routes)

    #
    # HTTP endpoints
    #

    async def _health_check(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Health check
        description: Returns the server health status and workflow counts.
        responses:
          200:
            description: Successful health check
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    status:
                      type: string
                      example: healthy
                    loaded_workflows:
                      type: integer
                      description: Number of workflow handlers currently loaded in memory
                    active_workflows:
                      type: integer
                      description: Number of workflow handlers that are active (not idle)
                    idle_workflows:
                      type: integer
                      description: Number of workflow handlers that are idle
                  required: [status, loaded_workflows, active_workflows, idle_workflows]
        """
        loaded = 0
        idle = 0
        for decorator in self._service._decorators.values():
            loaded += len(decorator.active_runs)
            for hid in decorator.active_runs:
                meta = decorator.get_run_metadata(hid)
                if meta is not None and meta.idle_since is not None:
                    idle += 1
        active = loaded - idle
        return JSONResponse(
            HealthResponse(
                status="healthy",
                loaded_workflows=loaded,
                active_workflows=active,
                idle_workflows=idle,
            ).model_dump()
        )

    async def _list_workflows(self, request: Request) -> JSONResponse:
        """
        ---
        summary: List workflows
        description: Returns the list of registered workflow names.
        responses:
          200:
            description: List of workflows
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    workflows:
                      type: array
                      items:
                        type: string
                  required: [workflows]
        """
        workflow_names = list(self._service._workflows.keys())
        return JSONResponse({"workflows": workflow_names})

    async def _list_workflow_events(self, request: Request) -> JSONResponse:
        """
        ---
        summary: List workflow events
        description: Returns the list of registered workflow event schemas.
        parameters:
          - in: path
            name: name
            required: true
            schema:
              type: string
            description: Registered workflow name.
        responses:
          200:
            description: List of workflow event schemas
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    events:
                      type: array
                      description: List of workflow event JSON schemas
                      items:
                        type: object
                  required: [events]
        """
        if "name" not in request.path_params:
            raise HTTPException(status_code=400, detail="name param is required")

        name = request.path_params["name"]
        if name not in self._service._workflows:
            raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")

        events = self._service._workflows[name].events + (
            self._service._additional_events.get(name, []) or []
        )

        return JSONResponse(
            WorkflowEventsListResponse(
                events=[event.model_json_schema() for event in events]
            ).model_dump()
        )

    async def _run_workflow(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Run workflow (wait)
        description: |
          Runs the specified workflow synchronously and returns the final result.
          The request body may include an optional serialized start event, an optional
          context object, and optional keyword arguments passed to the workflow run.
        parameters:
          - in: path
            name: name
            required: true
            schema:
              type: string
            description: Registered workflow name.
        requestBody:
          required: false
          content:
            application/json:
              schema:
                type: object
                properties:
                  start_event:
                    type: object
                    description: 'Plain JSON object representing the start event (e.g., {"message": "..."}).'
                  context:
                    type: object
                    description: Serialized workflow Context.
                  handler_id:
                    type: string
                    description: Workflow handler identifier to continue from a previous completed run.
                  kwargs:
                    type: object
                    description: Additional keyword arguments for the workflow.
        responses:
          200:
            description: Workflow completed successfully
            content:
              application/json:
                schema:
                  $ref: '#/components/schemas/Handler'
          400:
            description: Invalid start_event payload
          404:
            description: Workflow or handler identifier not found
          500:
            description: Error running workflow or invalid request body
        """
        name, workflow = self._extract_workflow(request)
        context, start_event, handler_id = await self._extract_run_params(
            request, workflow, name
        )

        if start_event is not None:
            input_ev = workflow.start_event_class.model_validate(start_event)
        else:
            input_ev = None

        try:
            adapter = await self._service.start_workflow(
                workflow_name=name,
                workflow=workflow,
                handler_id=handler_id,
                context=context,
                start_event=input_ev,
            )
            try:
                await adapter.get_result()
                status = 200
            except Exception as e:
                status = 500
                logger.error(f"Error running workflow: {e}", exc_info=True)

            # Wait briefly for the _on_complete task to persist final state
            await asyncio.sleep(0)

            handler_data = _build_handler_data(handler_id, adapter, self._service)

            return JSONResponse(handler_data.model_dump(), status_code=status)
        except Exception as e:
            status = 500
            logger.error(f"Error running workflow: {e}", exc_info=True)
            raise HTTPException(
                detail=f"Error running workflow: {e}", status_code=status
            )

    async def _get_events_schema(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Get JSON schema for start event
        description: |
          Gets the JSON schema of the start and stop events from the specified workflow and returns it under "start" (start event) and "stop" (stop event)
        parameters:
          - in: path
            name: name
            required: true
            schema:
              type: string
            description: Registered workflow name.
        requestBody:
          required: false
        responses:
          200:
            description: JSON schema successfully retrieved for start event
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    start:
                      description: JSON schema for the start event
                    stop:
                      description: JSON schema for the stop event
                  required: [start, stop]
          404:
            description: Workflow not found
          500:
            description: Error while getting the JSON schema for the start or stop event
        """
        name, workflow = self._extract_workflow(request)
        try:
            start_event_schema = workflow.start_event_class.model_json_schema()
        except Exception as e:
            raise HTTPException(
                detail=f"Error getting schema of start event for workflow: {e}",
                status_code=500,
            )
        try:
            stop_event_schema = workflow.stop_event_class.model_json_schema()
        except Exception as e:
            raise HTTPException(
                detail=f"Error getting schema of stop event for workflow: {e}",
                status_code=500,
            )

        return JSONResponse(
            WorkflowSchemaResponse(
                start=start_event_schema, stop=stop_event_schema
            ).model_dump()
        )

    async def _get_workflow_representation(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Get the representation of the workflow
        description: |
          Get the representation of the workflow as a directed graph in JSON format
        parameters:
          - in: path
            name: name
            required: true
            schema:
              type: string
            description: Registered workflow name.
        requestBody:
          required: false
        responses:
          200:
            description: JSON representation successfully retrieved
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    graph:
                      description: the elements of the JSON representation of the workflow
                  required: [graph]
          404:
            description: Workflow not found
          500:
            description: Error while getting JSON workflow representation
        """
        name, workflow = self._extract_workflow(request)
        try:
            workflow_graph = get_workflow_representation(workflow)
        except Exception as e:
            raise HTTPException(
                detail=f"Error while getting JSON workflow representation: {e}",
                status_code=500,
            )
        return JSONResponse(WorkflowGraphResponse(graph=workflow_graph).model_dump())

    async def _run_workflow_nowait(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Run workflow (no-wait)
        description: |
          Starts the specified workflow asynchronously and returns a handler identifier
          which can be used to query results or stream events.
        parameters:
          - in: path
            name: name
            required: true
            schema:
              type: string
            description: Registered workflow name.
        requestBody:
          required: false
          content:
            application/json:
              schema:
                type: object
                properties:
                  start_event:
                    type: object
                    description: 'Plain JSON object representing the start event (e.g., {"message": "..."}).'
                  context:
                    type: object
                    description: Serialized workflow Context.
                  handler_id:
                    type: string
                    description: Workflow handler identifier to continue from a previous completed run.
                  kwargs:
                    type: object
                    description: Additional keyword arguments for the workflow.
        responses:
          200:
            description: Workflow started
            content:
              application/json:
                schema:
                  $ref: '#/components/schemas/Handler'
          400:
            description: Invalid start_event payload
          404:
            description: Workflow or handler identifier not found
        """
        name, workflow = self._extract_workflow(request)
        context, start_event, handler_id = await self._extract_run_params(
            request, workflow, name
        )

        if start_event is not None:
            input_ev = workflow.start_event_class.model_validate(start_event)
        else:
            input_ev = None

        try:
            adapter = await self._service.start_workflow(
                workflow_name=name,
                workflow=workflow,
                handler_id=handler_id,
                context=context,
                start_event=input_ev,
            )

        except Exception as e:
            raise HTTPException(
                detail=f"Initial persistence failed: {e}", status_code=500
            )
        handler_data = _build_handler_data(handler_id, adapter, self._service)
        return JSONResponse(handler_data.model_dump())

    async def _load_handler(self, handler_id: str) -> HandlerData:
        """Load handler data from active decorators or persistence."""
        adapter = self._service.get_adapter_for_handler(handler_id)
        if adapter is not None:
            return _build_handler_data(handler_id, adapter, self._service)

        # Not active - check persistence
        found = await self._service._workflow_store.query(
            HandlerQuery(handler_id_in=[handler_id])
        )
        if not found:
            raise HTTPException(detail="Handler not found", status_code=404)
        return _handler_data_from_persistent(found[0])

    async def _get_workflow_result(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Get workflow result (deprecated)
        description: |
          Deprecated. Use GET /handlers/{handler_id} instead. Returns the final result of an asynchronously started workflow, if available.
        parameters:
          - in: path
            name: handler_id
            required: true
            schema:
              type: string
            description: Workflow run identifier returned from the no-wait run endpoint.
        deprecated: true
        responses:
          200:
            description: Result is available
            content:
              application/json:
                schema:
                  type: object
          202:
            description: Result not ready yet
            content:
              application/json:
                schema:
                  type: object
          404:
            description: Handler not found
          500:
            description: Error computing result
            content:
              text/plain:
                schema:
                  type: string
        """
        handler_id = request.path_params["handler_id"]
        if not handler_id:
            raise HTTPException(detail="Handler ID is required", status_code=400)

        handler_data = await self._load_handler(handler_id)
        status = (
            202
            if handler_data.status in "running"
            else 200
            if handler_data.status == "completed"
            else 500
        )
        response_model = handler_data.model_dump()

        # compatibility. Use handler api instead
        if not handler_data.result:
            response_model["result"] = None
        else:
            type = handler_data.result.qualified_name
            response_model["result"] = (
                handler_data.result.value.get("result")
                if type == "workflows.events.StopEvent"
                else handler_data.result.value
            )
        return JSONResponse(response_model, status_code=status)

    async def _get_workflow_handler(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Get workflow handler
        description: Returns the final result of an asynchronously started workflow, if available
        parameters:
          - in: path
            name: handler_id
            required: true
            schema:
              type: string
            description: Workflow run identifier returned from the no-wait run endpoint.
        responses:
          200:
            description: Result is available
            content:
              application/json:
                schema:
                  $ref: '#/components/schemas/Handler'
          202:
            description: Result not ready yet
            content:
              application/json:
                schema:
                  $ref: '#/components/schemas/Handler'
          404:
            description: Handler not found
          500:
            description: Error computing result
            content:
              text/plain:
                schema:
                  type: string
        """
        handler_id = request.path_params["handler_id"]
        if not handler_id:
            raise HTTPException(detail="Handler ID is required", status_code=400)

        handler_data = await self._load_handler(handler_id)
        status = (
            202
            if handler_data.status in "running"
            else 200
            if handler_data.status == "completed"
            else 500
        )
        return JSONResponse(handler_data.model_dump(), status_code=status)

    async def _stream_events(self, request: Request) -> StreamingResponse:
        """
        ---
        summary: Stream workflow events
        description: |
          Streams events produced by a workflow execution. Events are emitted as
          newline-delimited JSON by default, or as Server-Sent Events when `sse=true`.
        parameters:
          - in: path
            name: handler_id
            required: true
            schema:
              type: string
            description: Identifier returned from the no-wait run endpoint.
          - in: query
            name: sse
            required: false
            schema:
              type: boolean
              default: true
            description: If false, as NDJSON instead of Server-Sent Events.
          - in: query
            name: include_internal
            required: false
            schema:
              type: boolean
              default: false
            description: If true, include internal workflow events (e.g., step state changes).
          - in: query
            name: include_qualified_name
            required: false
            schema:
              type: boolean
              default: true
            description: If true, include the qualified name of the event in the response body.
        responses:
          200:
            description: Streaming started
            content:
              text/event-stream:
                schema:
                  type: object
          404:
            description: Handler not found
        """
        handler_id = request.path_params["handler_id"]
        include_internal = (
            request.query_params.get("include_internal", "false").lower() == "true"
        )
        include_qualified_name = (
            request.query_params.get("include_qualified_name", "true").lower() == "true"
        )
        sse = request.query_params.get("sse", "true").lower() == "true"

        # Find active adapter
        adapter = self._service.get_adapter_for_handler(handler_id)
        if adapter is None:
            # Try to reload from persistence (for released idle workflows)
            try:
                adapter, persisted = await self._service.try_reload_handler(handler_id)
            except Exception as e:
                raise HTTPException(
                    detail=f"Failed to reload handler: {e}", status_code=500
                )
            if adapter is None:
                if persisted:
                    status = persisted.status
                    if status in {"completed", "failed", "cancelled"}:
                        raise HTTPException(
                            detail="Handler is completed", status_code=204
                        )
                raise HTTPException(detail="Handler not found", status_code=404)

        # Check if already done
        try:
            if adapter._inner_done():
                raise HTTPException(detail="Handler is completed", status_code=204)
        except HTTPException:
            raise
        except Exception:
            pass

        media_type = "text/event-stream" if sse else "application/x-ndjson"

        async def event_stream() -> AsyncGenerator[str, None]:
            async for event in adapter.stream_published_events():
                if not include_internal and isinstance(event, InternalDispatchEvent):
                    continue
                envelope = EventEnvelopeWithMetadata.from_event(
                    event, include_qualified_name=include_qualified_name
                )
                payload = envelope.model_dump_json()
                if sse:
                    yield f"data: {payload}\n\n"
                else:
                    yield f"{payload}\n"

                await asyncio.sleep(0)

        return StreamingResponse(event_stream(), media_type=media_type)

    async def _get_handlers(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Get handlers
        description: Returns workflow handlers, optionally filtered by query parameters.
        parameters:
          - in: query
            name: status
            required: false
            schema:
              type: array
              items:
                type: string
                enum: [running, completed, failed, cancelled]
            style: form
            explode: true
            description: |
              Filter by handler status. Can be provided multiple times (e.g., status=running&status=failed)
          - in: query
            name: workflow_name
            required: false
            schema:
              type: array
              items:
                type: string
            style: form
            explode: true
            description: |
              Filter by workflow name. Can be provided multiple times (e.g., workflow_name=test&workflow_name=other)
        responses:
          200:
            description: List of handlers
            content:
              application/json:
                schema:
                  $ref: '#/components/schemas/HandlersList'
        """

        def _parse_list_param(param_name: str) -> list[str] | None:
            # parse repeated params
            values = list(request.query_params.getlist(param_name))
            if not values:
                single = request.query_params.get(param_name) or ""
                values = [single]
            values = [value.strip() for value in values if value.strip()]
            if not values:
                return None
            return values

        # Parse filters
        status_values = _parse_list_param("status")
        workflow_name_in = _parse_list_param("workflow_name")

        # Narrow types for status to match HandlerQuery expectations
        allowed_status_values: set[Status] = {
            "running",
            "completed",
            "failed",
            "cancelled",
        }

        status_in: list[Status] | None = (
            cast(
                list[Status],
                list(set(allowed_status_values).intersection(status_values)),
            )
            if status_values is not None
            else None
        )
        persistent_handlers = await self._service._workflow_store.query(
            HandlerQuery(status_in=status_in, workflow_name_in=workflow_name_in)
        )
        items = [
            HandlerData(
                handler_id=h.handler_id,
                workflow_name=h.workflow_name,
                run_id=h.run_id,
                status=h.status,
                started_at=h.started_at.isoformat() if h.started_at else "",
                updated_at=h.updated_at.isoformat() if h.updated_at else None,
                completed_at=h.completed_at.isoformat() if h.completed_at else None,
                error=h.error,
                result=EventEnvelopeWithMetadata.from_event(h.result)
                if h.result
                else None,
            )
            for h in persistent_handlers
        ]
        return JSONResponse(HandlersListResponse(handlers=items).model_dump())

    async def _post_event(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Send event to workflow
        description: Sends an event to a running workflow's context.
        parameters:
          - in: path
            name: handler_id
            required: true
            schema:
              type: string
            description: Workflow handler identifier.
        requestBody:
          required: true
          content:
            application/json:
              schema:
                type: object
                properties:
                  event:
                    description: Serialized event. Accepts object or JSON-encoded string for backward compatibility.
                  step:
                    type: string
                    description: Optional target step name.
                required: [event]
        responses:
          200:
            description: Event sent successfully
          400:
            description: Invalid event data
          404:
            description: Handler not found
          409:
            description: Workflow already completed
        """
        handler_id = request.path_params["handler_id"]

        # Find active adapter
        adapter = self._service.get_adapter_for_handler(handler_id)

        # Check if active and completed
        if adapter is not None:
            try:
                if adapter._inner_done():
                    raise HTTPException(
                        detail="Workflow already completed", status_code=409
                    )
            except HTTPException:
                raise
            except Exception:
                pass
        else:
            # Try to reload from persistence
            try:
                adapter, persisted = await self._service.try_reload_handler(handler_id)
            except Exception as e:
                raise HTTPException(
                    detail=f"Failed to reload handler: {e}", status_code=500
                )
            if adapter is None:
                if persisted and is_status_completed(persisted.status):
                    raise HTTPException(
                        detail="Workflow already completed", status_code=409
                    )
                elif persisted is None:
                    raise HTTPException(detail="Handler not found", status_code=404)
                else:
                    raise HTTPException(
                        detail=f"Failed to resume incomplete handler with status {persisted.status}",
                        status_code=500,
                    )

        # Get workflow name for event registry
        workflow_name = self._service.get_workflow_name_for_handler(handler_id)
        if workflow_name is None:
            raise HTTPException(
                detail="Workflow name not found for handler", status_code=500
            )

        # Parse request body
        try:
            body = await request.json()
            event_str = body.get("event")
            step = body.get("step")

            if not event_str:
                raise HTTPException(detail="Event data is required", status_code=400)

            # Deserialize the event
            try:
                event = EventEnvelope.parse(
                    event_str, self._service.event_registry(workflow_name)
                )
            except EventValidationError as e:
                raise HTTPException(detail=str(e), status_code=400)
            except Exception as e:
                raise HTTPException(
                    detail=f"Failed to deserialize event: {e}", status_code=400
                )

            # Send the event via the adapter using TickAddEvent
            try:
                tick = TickAddEvent(event=event, step_name=step)
                await adapter.send_event(tick)
            except Exception as e:
                raise HTTPException(
                    detail=f"Failed to send event: {e}", status_code=400
                )

            return JSONResponse(SendEventResponse(status="sent").model_dump())

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                detail=f"Error processing request: {e}", status_code=500
            )

    async def _cancel_handler(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Stop and delete handler
        description: |
          Stops a running workflow handler by cancelling its tasks. Optionally removes the
          handler from the persistence store if purge=true.
        parameters:
          - in: path
            name: handler_id
            required: true
            schema:
              type: string
            description: Workflow handler identifier.
          - in: query
            name: purge
            required: false
            schema:
              type: boolean
              default: false
            description: If true, also deletes the handler from the store.
        responses:
          200:
            description: Handler cancelled and deleted or cancelled only
          404:
            description: Handler not found
        """
        handler_id = request.path_params["handler_id"]
        purge = request.query_params.get("purge", "false").lower() == "true"

        adapter = self._service.get_adapter_for_handler(handler_id)
        if adapter is None and not purge:
            raise HTTPException(detail="Handler not found", status_code=404)

        # Cancel the adapter if active
        if adapter is not None:
            adapter._cancel_idle_release_timer()
            try:
                await adapter.cancel()
            except Exception:
                pass
            # Persist cancelled status
            workflow_name = (
                self._service.get_workflow_name_for_handler(handler_id) or ""
            )
            for decorator in self._service._decorators.values():
                meta = decorator.get_run_metadata(handler_id)
                if meta is not None:
                    now = datetime.now(timezone.utc)
                    meta.completed_at = now
                    meta.updated_at = now
                    persistent = PersistentHandler(
                        handler_id=handler_id,
                        workflow_name=workflow_name,
                        status="cancelled",
                        run_id=adapter.run_id,
                        started_at=meta.started_at,
                        updated_at=now,
                        completed_at=now,
                        ctx=self._service._get_context_dict(handler_id) or {},
                    )
                    try:
                        await self._service._workflow_store.update(persistent)
                    except Exception:
                        pass
                    break
            # Unregister from all decorators
            for decorator in self._service._decorators.values():
                decorator.unregister_run(handler_id)

        # Handle persistence
        if purge:
            n_deleted = await self._service._workflow_store.delete(
                HandlerQuery(handler_id_in=[handler_id])
            )
            if n_deleted == 0:
                raise HTTPException(detail="Handler not found", status_code=404)

        return JSONResponse(
            CancelHandlerResponse(
                status="deleted" if purge else "cancelled"
            ).model_dump()
        )

    #
    # Private methods
    #
    def _extract_workflow(self, request: Request) -> tuple[str, Workflow]:
        if "name" not in request.path_params:
            raise HTTPException(detail="'name' parameter missing", status_code=400)
        name = request.path_params["name"]

        if name not in self._service._workflows:
            raise HTTPException(detail="Workflow not found", status_code=404)

        return (name, self._service._workflows[name])

    async def _extract_run_params(
        self, request: Request, workflow: Workflow, workflow_name: str
    ) -> tuple[Context | None, StartEvent | None, str]:
        try:
            try:
                body = await request.json()
            except Exception as e:
                raise HTTPException(detail=f"Invalid JSON body: {e}", status_code=400)
            context_data = body.get("context")
            run_kwargs = body.get("kwargs", {})
            start_event_data = body.get("start_event", run_kwargs)
            handler_id = body.get("handler_id")

            # Extract custom StartEvent if present
            start_event = None
            if start_event_data is not None:
                try:
                    start_event = EventEnvelope.parse(
                        start_event_data,
                        self._service.event_registry(workflow_name),
                        explicit_event=workflow.start_event_class,
                    )

                except Exception as e:
                    raise HTTPException(
                        detail=f"Validation error for 'start_event': {e}",
                        status_code=400,
                    )
                if start_event is not None and not isinstance(
                    start_event, workflow.start_event_class
                ):
                    raise HTTPException(
                        detail=f"Start event must be an instance of {workflow.start_event_class}",
                        status_code=400,
                    )

            # Extract custom Context if present
            context = None
            if context_data:
                context = Context.from_dict(workflow=workflow, data=context_data)
            elif handler_id:
                persisted_handlers = await self._service._workflow_store.query(
                    HandlerQuery(
                        handler_id_in=[handler_id],
                        workflow_name_in=[workflow_name],
                        status_in=["completed"],
                    )
                )
                if len(persisted_handlers) == 0:
                    raise HTTPException(detail="Handler not found", status_code=404)

                context = Context.from_dict(workflow, persisted_handlers[0].ctx)

            handler_id = handler_id or nanoid()
            return (context, start_event, handler_id)

        except HTTPException:
            # Re-raise HTTPExceptions as-is (like start_event validation errors)
            raise
        except Exception as e:
            raise HTTPException(
                detail=f"Error processing request body: {e}", status_code=500
            )
