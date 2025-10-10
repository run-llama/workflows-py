# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any, AsyncGenerator
from datetime import datetime, timezone

from pydantic import BaseModel
import uvicorn
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
from workflows.context.serializers import JsonSerializer
from workflows.events import (
    Event,
    InternalDispatchEvent,
    StartEvent,
    StepState,
    StepStateChanged,
    StopEvent,
)
from workflows.handler import WorkflowHandler


from workflows.protocol import (
    CancelHandlerResponse,
    HandlerData,
    HandlersListResponse,
    HealthResponse,
    SendEventResponse,
    WorkflowEventsListResponse,
    WorkflowGraphResponse,
    WorkflowSchemaResponse,
)
from workflows.server.abstract_workflow_store import (
    AbstractWorkflowStore,
    EmptyWorkflowStore,
    HandlerQuery,
    PersistentHandler,
    Status,
)
from workflows.types import RunResultT

# Protocol models are used on the client side; server responds with plain dicts
from .utils import nanoid
from .representation_utils import _extract_workflow_structure
from .serialization import build_event_envelope

logger = logging.getLogger()


class WorkflowServer:
    def __init__(
        self,
        *,
        middleware: list[Middleware] | None = None,
        workflow_store: AbstractWorkflowStore = EmptyWorkflowStore(),
        # retry/backoff seconds for persisting the handler state in the store after failures. Configurable mainly for testing.
        persistence_backoff: list[float] = [0.5, 3],
    ):
        self._workflows: dict[str, Workflow] = {}
        self._additional_events: dict[str, list[type[Event]] | None] = {}
        self._contexts: dict[str, Context] = {}
        self._handlers: dict[str, _WorkflowHandler] = {}
        self._results: dict[str, StopEvent] = {}
        self._workflow_store = workflow_store
        self._assets_path = Path(__file__).parent / "static"
        self._persistence_backoff = persistence_backoff

        self._middleware = middleware or [
            Middleware(
                CORSMiddleware,
                # regex echoes the origin header back, which some browsers require (rather than "*") when credentials are required
                allow_origin_regex=".*",
                allow_methods=["*"],
                allow_headers=["*"],
                allow_credentials=True,
            )
        ]

        self._routes = [
            Route(
                "/workflows",
                self._list_workflows,
                methods=["GET"],
            ),
            Route(
                "/workflows/{name}/run",
                self._run_workflow,
                methods=["POST"],
            ),
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
            Route(
                "/events/{handler_id}",
                self._stream_events,
                methods=["GET"],
            ),
            Route(
                "/events/{handler_id}",
                self._post_event,
                methods=["POST"],
            ),
            Route(
                "/health",
                self._health_check,
                methods=["GET"],
            ),
            Route(
                "/handlers",
                self._get_handlers,
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

        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
            async with self.contextmanager():
                yield

        self.app = Starlette(
            routes=self._routes,
            middleware=self._middleware,
            lifespan=lifespan,
        )
        # Serve the UI as static files
        self.app.mount(
            "/", app=StaticFiles(directory=self._assets_path, html=True), name="ui"
        )

    def add_workflow(
        self,
        name: str,
        workflow: Workflow,
        additional_events: list[type[Event]] | None = None,
    ) -> None:
        self._workflows[name] = workflow
        if additional_events is not None:
            self._additional_events[name] = additional_events

    async def start(self) -> "WorkflowServer":
        """Resumes previously running workflows, if they were not complete at last shutdown"""
        handlers = await self._workflow_store.query(
            HandlerQuery(
                status_in=["running"], workflow_name_in=list(self._workflows.keys())
            )
        )
        for persistent in handlers:
            workflow = self._workflows[persistent.workflow_name]
            try:
                ctx = Context.from_dict(workflow=workflow, data=persistent.ctx)
                handler = workflow.run(ctx=ctx)
            except Exception as e:
                logger.error(
                    f"Failed to resume handler {persistent.handler_id} for workflow {persistent.workflow_name}: {e}"
                )
                try:
                    await self._workflow_store.update(
                        PersistentHandler(
                            handler_id=persistent.handler_id,
                            workflow_name=persistent.workflow_name,
                            status="failed",
                            ctx=persistent.ctx,
                        )
                    )
                except Exception:
                    pass
                continue

            self._run_workflow_handler(
                persistent.handler_id, persistent.workflow_name, handler
            )
        return self

    @asynccontextmanager
    async def contextmanager(self) -> AsyncGenerator["WorkflowServer", None]:
        """Use this server as a context manager to start and stop it"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def stop(self) -> None:
        logger.info(
            f"Shutting down Workflow server. Cancelling {len(self._handlers)} handlers."
        )
        for handler in list(self._handlers.values()):
            await self._close_handler(handler)
        self._handlers.clear()
        self._results.clear()

    async def serve(
        self,
        host: str = "localhost",
        port: int = 80,
        uvicorn_config: dict[str, Any] | None = None,
    ) -> None:
        """Run the server."""
        uvicorn_config = uvicorn_config or {}

        config = uvicorn.Config(self.app, host=host, port=port, **uvicorn_config)
        server = uvicorn.Server(config)
        logger.info(
            f"Starting Workflow server at http://{host}:{port}{uvicorn_config.get('root_path', '/')}"
        )

        await server.serve()

    def openapi_schema(self) -> dict:
        app = self.app
        gen = SchemaGenerator(
            {
                "openapi": "3.0.0",
                "info": {
                    "title": "Workflows API",
                    "version": version("llama-index-workflows"),
                },
                "components": {
                    "schemas": {
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
                                "result": {"description": "Workflow result value"},
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

        return gen.get_schema(app.routes)

    #
    # HTTP endpoints
    #

    async def _health_check(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Health check
        description: Returns the server health status.
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
                  required: [status]
        """
        return JSONResponse(HealthResponse(status="healthy").model_dump())

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
        workflow_names = list(self._workflows.keys())
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
        if name not in self._workflows:
            raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")

        events = self._workflows[name].events
        additional_events = self._additional_events.get(name, [])
        if additional_events:
            events.extend(additional_events)

        event_objs = []
        for event in events:
            event_objs.append(event.model_json_schema())

        return JSONResponse(WorkflowEventsListResponse(events=event_objs).model_dump())

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
        workflow = self._extract_workflow(request)
        context, start_event, handler_id = await self._extract_run_params(
            request, workflow.workflow, workflow.name
        )

        if start_event is not None:
            input_ev = workflow.workflow.start_event_class.model_validate(start_event)
        else:
            input_ev = None

        try:
            handler = workflow.workflow.run(
                ctx=context,
                start_event=input_ev,
            )
            wrapper = self._run_workflow_handler(handler_id, workflow.name, handler)
            await handler
            return JSONResponse(wrapper.to_response_model().model_dump())
        except Exception as e:
            raise HTTPException(detail=f"Error running workflow: {e}", status_code=500)

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
        workflow = self._extract_workflow(request)
        try:
            start_event_schema = workflow.workflow.start_event_class.model_json_schema()
        except Exception as e:
            raise HTTPException(
                detail=f"Error getting schema of start event for workflow: {e}",
                status_code=500,
            )
        try:
            stop_event_schema = workflow.workflow.stop_event_class.model_json_schema()
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
        workflow = self._extract_workflow(request)
        try:
            workflow_graph = _extract_workflow_structure(workflow.workflow)
        except Exception as e:
            raise HTTPException(
                detail=f"Error while getting JSON workflow representation: {e}",
                status_code=500,
            )
        return JSONResponse(
            WorkflowGraphResponse(graph=workflow_graph.to_response_model()).model_dump()
        )

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
        workflow = self._extract_workflow(request)
        context, start_event, handler_id = await self._extract_run_params(
            request, workflow.workflow, workflow.name
        )

        if start_event is not None:
            input_ev = workflow.workflow.start_event_class.model_validate(start_event)
        else:
            input_ev = None

        handler = workflow.workflow.run(
            ctx=context,
            start_event=input_ev,
        )
        wrapper = self._run_workflow_handler(
            handler_id,
            workflow.name,
            handler,
        )
        return JSONResponse(wrapper.to_response_model().model_dump())

    async def _get_workflow_result(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Get workflow result
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

        wrapper = self._handlers.get(handler_id)
        if wrapper is None:
            raise HTTPException(detail="Handler not found", status_code=404)

        handler = wrapper.run_handler
        if not handler.done():
            return JSONResponse(
                wrapper.to_response_model().model_dump(), status_code=202
            )

        try:
            result = await handler
            self._results[handler_id] = result

            return JSONResponse(wrapper.to_response_model().model_dump())
        except Exception as e:
            raise HTTPException(
                detail=f"Error getting workflow result: {e}", status_code=500
            )

    async def _stream_events(self, request: Request) -> StreamingResponse:
        """
        ---
        summary: Stream workflow events
        description: |
          Streams events produced by a workflow execution. Events are emitted as
          newline-delimited JSON by default, or as Server-Sent Events when `sse=true`.
          Event data is returned as an envelope that preserves backward-compatible fields
          and adds metadata for type-safety on the client:
          {
            "__is_pydantic": true,
            "value": <pydantic serialized value>,
            "qualified_name": <python path to pydantic class>,  # deprecated, prefer `mro`
            "mro": [<qualified class names from most to least specific>],
            "origin": "builtin" | "user"
          }

          Event queue is mutable. Elements are added to the queue by the workflow handler, and removed by any consumer of the queue.
          The queue is protected by a lock that is acquired by the consumer, so only one consumer of the queue at a time is allowed.

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
            name: acquire_timeout
            required: false
            schema:
              type: number
              default: 1
            description: Timeout for acquiring the lock to iterate over the events.
        responses:
          200:
            description: Streaming started
            content:
              text/event-stream:
                schema:
                  type: object
                  description: Server-Sent Events stream of event data.
                  properties:
                    value:
                      type: object
                      description: The event value.
                    qualified_name:
                      type: string
                      description: The qualified name of the event.
                  required: [value, qualified_name]
          404:
            description: Handler not found
        """
        handler_id = request.path_params["handler_id"]
        timeout = request.query_params.get("acquire_timeout", "1").lower()
        include_internal = (
            request.query_params.get("include_internal", "false").lower() == "true"
        )
        sse = request.query_params.get("sse", "true").lower() == "true"
        try:
            timeout = float(timeout)
        except ValueError:
            raise HTTPException(
                detail=f"Invalid acquire_timeout: '{timeout}'", status_code=400
            )

        handler = self._handlers.get(handler_id)
        if handler is None:
            raise HTTPException(detail="Handler not found", status_code=404)
        if handler.queue.empty() and handler.task.done():
            # https://html.spec.whatwg.org/multipage/server-sent-events.html
            # Clients will reconnect if the connection is closed; a client can
            # be told to stop reconnecting using the HTTP 204 No Content response code.
            raise HTTPException(detail="Handler is completed", status_code=204)

        # Get raw_event query parameter
        media_type = "text/event-stream" if sse else "application/x-ndjson"

        try:
            generator = await handler.acquire_events_stream(timeout=timeout)
        except NoLockAvailable as e:
            raise HTTPException(
                detail=f"No lock available to acquire after {timeout}s timeout",
                status_code=409,
            ) from e

        async def event_stream(handler: _WorkflowHandler) -> AsyncGenerator[str, None]:
            serializer = JsonSerializer()

            async for event in generator:
                if not include_internal and isinstance(event, InternalDispatchEvent):
                    continue
                envelope = build_event_envelope(event, serializer)
                payload = json.dumps(envelope)
                if sse:
                    # emit as untyped data. Difficult to subscribe to dynamic event types with SSE.
                    yield f"data: {payload}\n\n"
                else:
                    yield f"{payload}\n"

                await asyncio.sleep(0)

        return StreamingResponse(event_stream(handler), media_type=media_type)

    async def _get_handlers(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Get handlers
        description: Returns all workflow handlers.
        responses:
          200:
            description: List of handlers
            content:
              application/json:
                schema:
                  $ref: '#/components/schemas/HandlersList'
        """
        items = [wrapper.to_response_model() for wrapper in self._handlers.values()]
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
                    type: string
                    description: Serialized event in JSON format.
                    examples:
                        {"type": "event_name", "data": {"key": "value"}}
                  step:
                    type: string
                    description: Optional target step name. If not provided, event is sent to all steps.
                required: [event]
        responses:
          200:
            description: Event sent successfully
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    status:
                      type: string
                      enum: [sent]
                  required: [status]
          400:
            description: Invalid event data
          404:
            description: Handler not found
          409:
            description: Workflow already completed
        """
        handler_id = request.path_params["handler_id"]

        # Check if handler exists
        wrapper = self._handlers.get(handler_id)
        if wrapper is None:
            raise HTTPException(detail="Handler not found", status_code=404)

        handler = wrapper.run_handler
        events_by_title = {
            e.__name__: e for e in self._workflows[wrapper.workflow_name].events
        }

        # Check if workflow is still running
        if handler.done():
            raise HTTPException(detail="Workflow already completed", status_code=409)

        # Get the context
        ctx = handler.ctx
        if ctx is None:
            raise HTTPException(detail="Context not available", status_code=500)

        # Parse request body
        try:
            body = await request.json()
            event_str = body.get("event")
            step = body.get("step")

            if not event_str:
                raise HTTPException(detail="Event data is required", status_code=400)

            # Deserialize the event
            serializer = JsonSerializer()
            try:
                event_data = serializer.deserialize(event_str)
                if (
                    isinstance(event_data, dict)
                    and "type" in event_data
                    and "data" in event_data
                ):
                    event_title = event_data["type"]
                    event = events_by_title[event_title].model_validate(
                        event_data["data"]
                    )
                elif isinstance(event_data, Event):
                    event = event_data
                else:
                    raise ValueError(
                        "Invalid event data. Should be a dictionary of {'type': 'event_type', 'data': {...}} or a serialized event"
                    )
            except Exception as e:
                raise HTTPException(
                    detail=f"Failed to deserialize event: {e}", status_code=400
                )

            # Send the event to the context
            try:
                ctx.send_event(event, step=step)
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
            description: If true, also deletes the handler from the store, otherwise updates the status to cancelled.
        responses:
          200:
            description: Handler cancelled and deleted or cancelled only
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    status:
                      type: string
                      enum: [deleted, cancelled]
                  required: [status]
          404:
            description: Handler not found
        """
        handler_id = request.path_params["handler_id"]
        # Simple boolean parsing aligned with other APIs (e.g., `sse`): only "true" enables
        purge = request.query_params.get("purge", "false").lower() == "true"

        wrapper = self._handlers.get(handler_id)
        if wrapper is None and not purge:
            raise HTTPException(detail="Handler not found", status_code=404)

        if wrapper is not None:
            await self._close_handler(wrapper)

        # Single persistence delete path
        if purge:
            n_deleted = await self._workflow_store.delete(
                HandlerQuery(handler_id_in=[handler_id])
            )
            if n_deleted == 0:
                raise HTTPException(detail="Handler not found", status_code=404)
        else:
            # mark it as cancelled if it's not already completed
            existing = await self._workflow_store.query(
                HandlerQuery(handler_id_in=[handler_id])
            )
            if existing:
                ctx = (
                    wrapper.run_handler.ctx.to_dict()
                    if wrapper and wrapper.run_handler.ctx
                    else existing[0].ctx
                )
                await self._workflow_store.update(
                    PersistentHandler(
                        handler_id=handler_id,
                        workflow_name=existing[0].workflow_name,
                        status="cancelled",
                        ctx=ctx,
                    )
                )

        return JSONResponse(
            CancelHandlerResponse(
                status="deleted" if purge else "cancelled"
            ).model_dump()
        )

    #
    # Private methods
    #
    def _extract_workflow(self, request: Request) -> _NamedWorkflow:
        if "name" not in request.path_params:
            raise HTTPException(detail="'name' parameter missing", status_code=400)
        name = request.path_params["name"]

        if name not in self._workflows:
            raise HTTPException(detail="Workflow not found", status_code=404)

        return _NamedWorkflow(name=name, workflow=self._workflows[name])

    async def _extract_run_params(
        self, request: Request, workflow: Workflow, workflow_name: str
    ) -> tuple[Context | None, StartEvent | None, str]:
        try:
            body = await request.json()
            context_data = body.get("context")
            run_kwargs = body.get("kwargs", {})
            start_event_data = body.get("start_event", run_kwargs)
            handler_id = body.get("handler_id")

            # Extract custom StartEvent if present
            start_event = None
            if start_event_data is not None:
                serializer = JsonSerializer()
                try:
                    start_event = (
                        serializer.deserialize(start_event_data)
                        if isinstance(start_event_data, str)
                        else serializer.deserialize_value(start_event_data)
                    )
                    if isinstance(start_event, dict):
                        start_event = workflow.start_event_class.model_validate(
                            start_event
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
                persisted_handlers = await self._workflow_store.query(
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

    def _run_workflow_handler(
        self, handler_id: str, workflow_name: str, handler: WorkflowHandler
    ) -> _WorkflowHandler:
        """
        Streams events from the handler, persisting them, and pushing them to a queue.
        Stores a _WorkflowHandler helper that wraps the handler with it's queue and streaming task.
        """
        queue: asyncio.Queue[Event] = asyncio.Queue()

        async def _stream_events(handler: WorkflowHandler) -> None:
            async def checkpoint(status: Status) -> None:
                if not handler.ctx:
                    return
                ctx = handler.ctx.to_dict()
                backoffs = list(self._persistence_backoff)
                while True:
                    try:
                        await self._workflow_store.update(
                            PersistentHandler(
                                handler_id=handler_id,
                                workflow_name=workflow_name,
                                status=status,
                                ctx=ctx,
                            )
                        )
                        return
                    except Exception as e:
                        backoff = backoffs.pop(0) if backoffs else None
                        if backoff is None:
                            logger.error(
                                f"Failed to checkpoint handler {handler_id} after final attempt. Failing the handler.",
                                exc_info=True,
                            )
                            handler.cancel()
                            raise
                        logger.error(
                            f"Failed to checkpoint handler {handler_id}. Retrying in {backoff} seconds: {e}"
                        )
                        await asyncio.sleep(backoff)

            await checkpoint("running")
            async for event in handler.stream_events(expose_internal=True):
                if (  # Watch for a specific internal event that signals the step is complete
                    isinstance(event, StepStateChanged)
                    and event.step_state == StepState.NOT_RUNNING
                ):
                    state = handler.ctx.to_dict() if handler.ctx else None
                    if state is None:
                        logger.warning(
                            f"Context state is None for handler {handler_id}. This is not expected."
                        )
                        continue
                    await checkpoint("running")

                wrapper.updated_at = datetime.now(timezone.utc)
                queue.put_nowait(event)
            # done when stream events are complete
            status: Status = "completed"
            wrapper.completed_at = datetime.now(timezone.utc)
            try:
                await handler
            except Exception as e:
                status = "failed"
                logger.error(f"Workflow run {handler_id} failed! {e}", exc_info=True)

            if handler.ctx is None:
                logger.warning(
                    f"Context is None for handler {handler_id}. This is not expected."
                )
                return

            await checkpoint(status)

        task = asyncio.create_task(_stream_events(handler))
        wrapper = _WorkflowHandler(
            run_handler=handler,
            queue=queue,
            task=task,
            consumer_mutex=asyncio.Lock(),
            handler_id=handler_id,
            workflow_name=workflow_name,
            started_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            completed_at=None,
        )
        self._handlers[handler_id] = wrapper
        return wrapper

    async def _close_handler(
        self, handler: _WorkflowHandler, *, persist_status: Status | None = None
    ) -> None:
        if not handler.run_handler.done():
            try:
                handler.run_handler.cancel()
            except Exception:
                pass
            try:
                await handler.run_handler.cancel_run()
            except Exception:
                pass
            if persist_status is not None:
                if handler.run_handler.ctx is not None:
                    await self._workflow_store.update(
                        PersistentHandler(
                            handler_id=handler.handler_id,
                            workflow_name=handler.workflow_name,
                            status=persist_status,
                            ctx=handler.run_handler.ctx.to_dict(),
                        )
                    )
                else:
                    await self._workflow_store.delete(
                        HandlerQuery(handler_id_in=[handler.handler_id])
                    )
        if handler.task is not None and not handler.task.done():
            try:
                handler.task.cancel()
            except Exception:
                pass
        self._handlers.pop(handler.handler_id, None)
        self._results.pop(handler.handler_id, None)


@dataclass
class _WorkflowHandler:
    """A wrapper around a handler: WorkflowHandler. Necessary to monitor and dispatch events from the handler's stream_events"""

    run_handler: WorkflowHandler
    queue: asyncio.Queue[Event]
    task: asyncio.Task[None]
    # only one consumer of the queue at a time allowed
    consumer_mutex: asyncio.Lock

    # metadata
    handler_id: str
    workflow_name: str
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    def to_response_model(self) -> HandlerData:
        return HandlerData(
            handler_id=self.handler_id,
            workflow_name=self.workflow_name,
            run_id=self.run_handler.run_id,
            status=self.status,
            started_at=self.started_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            completed_at=self.completed_at.isoformat()
            if self.completed_at is not None
            else None,
            error=self.error,
            result=self.result.model_dump()
            if self.result is not None and isinstance(self.result, BaseModel)
            else self.result,
        )

    @property
    def status(self) -> Status:
        if not self.run_handler.done():
            return "running"
        # done
        exc = self.run_handler.exception()
        if exc is not None:
            return "failed"
        return "completed"

    @property
    def error(self) -> str | None:
        if not self.run_handler.done():
            return None
        exc = self.run_handler.exception()
        return str(exc) if exc is not None else None

    @property
    def result(self) -> RunResultT | None:
        if not self.run_handler.done():
            return None
        try:
            return self.run_handler.result()
        except Exception:
            return None

    async def acquire_events_stream(
        self, timeout: float = 1
    ) -> AsyncGenerator[Event, None]:
        """
        Acquires the lock to iterate over the events, and returns generator of events.
        """
        try:
            await asyncio.wait_for(self.consumer_mutex.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise NoLockAvailable(
                f"No lock available to acquire after {timeout}s timeout"
            )
        return self._iter_events(timeout=timeout)

    async def _iter_events(self, timeout: float = 1) -> AsyncGenerator[Event, None]:
        """
        Converts the queue to an async generator while the workflow is still running, and there are still events.
        For better or worse, multiple consumers will compete for events
        """

        try:
            while not self.queue.empty() or not self.task.done():
                available_events = []
                while not self.queue.empty():
                    available_events.append(self.queue.get_nowait())
                for event in available_events:
                    yield event
                queue_get_task: asyncio.Task[Event] = asyncio.create_task(
                    self.queue.get()
                )
                task_waitable = self.task
                done, pending = await asyncio.wait(
                    {queue_get_task, task_waitable}, return_when=asyncio.FIRST_COMPLETED
                )
                if queue_get_task in done:
                    yield await queue_get_task
                else:  # otherwise task completed, so nothing else will be published to the queue
                    queue_get_task.cancel()
                    break
        finally:
            self.consumer_mutex.release()


class NoLockAvailable(Exception):
    """Raised when no lock is available to acquire after a timeout"""

    pass


@dataclass
class _NamedWorkflow:
    name: str
    workflow: Workflow


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate OpenAPI schema")
    parser.add_argument(
        "--output", type=str, default="openapi.json", help="Output file path"
    )
    args = parser.parse_args()

    server = WorkflowServer()
    dict_schema = server.openapi_schema()
    with open(args.output, "w") as f:
        json.dump(dict_schema, indent=2, fp=f)
    print(f"OpenAPI schema written to {args.output}")
