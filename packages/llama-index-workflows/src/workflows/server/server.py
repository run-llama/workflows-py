# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, cast

import uvicorn
from llama_index_instrumentation.dispatcher import instrument_tags
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
from workflows.events import (
    Event,
    InternalDispatchEvent,
    StartEvent,
    StepState,
    StepStateChanged,
    StopEvent,
    UnhandledEvent,
    WorkflowIdleEvent,
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
    is_status_completed,
)
from workflows.protocol.serializable_events import (
    EventEnvelope,
    EventEnvelopeWithMetadata,
    EventValidationError,
)
from workflows.representation import get_workflow_representation
from workflows.server.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
    Status,
)
from workflows.server.keyed_lock import KeyedLock
from workflows.server.memory_workflow_store import MemoryWorkflowStore

# Protocol models are used on the client side; server responds with plain dicts
from workflows.utils import _nanoid as nanoid

logger = logging.getLogger()


class WorkflowServer:
    def __init__(
        self,
        *,
        middleware: list[Middleware] | None = None,
        workflow_store: AbstractWorkflowStore | None = None,
        # retry/backoff seconds for persisting the handler state in the store after failures. Configurable mainly for testing.
        persistence_backoff: list[float] = [0.5, 3],
        # Release idle workflows from memory after this timeout (None = disabled)
        idle_release_timeout: timedelta | None = timedelta(seconds=10),
    ):
        # Workflows that this server will run
        self._workflows: dict[str, Workflow] = {}
        # Additional event schemas supported. Used for serdes lookups. Should be used sparingly if ever.
        self._additional_events: dict[str, list[type[Event]] | None] = {}
        self._handlers: dict[str, _WorkflowHandler] = {}
        self._workflow_store = (
            workflow_store if workflow_store is not None else MemoryWorkflowStore()
        )
        self._assets_path = Path(__file__).parent / "static"
        self._persistence_backoff = list(persistence_backoff)
        self._idle_release_timeout = idle_release_timeout

        self._middleware = middleware or [
            Middleware(
                CORSMiddleware,  # type: ignore[arg-type]
                # regex echoes the origin header back, which some browsers require (rather than "*") when credentials are required
                allow_origin_regex=".*",
                allow_methods=["*"],
                allow_headers=["*"],
                allow_credentials=True,
            )
        ]

        # Lock to prevent concurrent reloads of the same handler
        self._reload_lock = KeyedLock()

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
        """Resumes previously running workflows, if they were not complete at last shutdown.

        Idle workflows are not resumed - they remain released and will be
        loaded on-demand when events arrive for them.
        """
        handlers = await self._workflow_store.query(
            HandlerQuery(
                status_in=["running"],
                workflow_name_in=list(self._workflows.keys()),
                is_idle=False,
            )
        )
        for persistent in handlers:
            workflow = self._workflows[persistent.workflow_name]
            try:
                await self._start_workflow(
                    workflow=_NamedWorkflow(
                        name=persistent.workflow_name, workflow=workflow
                    ),
                    handler_id=persistent.handler_id,
                    context=Context.from_dict(workflow=workflow, data=persistent.ctx),
                )
            except Exception as e:
                logger.error(
                    f"Failed to resume handler {persistent.handler_id} for workflow {persistent.workflow_name}: {e}"
                )

                try:
                    now = datetime.now(timezone.utc)
                    await self._workflow_store.update(
                        PersistentHandler(
                            handler_id=persistent.handler_id,
                            workflow_name=persistent.workflow_name,
                            status="failed",
                            run_id=persistent.run_id,
                            error=str(e),
                            result=None,
                            started_at=persistent.started_at,
                            updated_at=now,
                            completed_at=now,
                            ctx=persistent.ctx,
                        )
                    )
                except Exception:
                    pass
                continue

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
        await asyncio.gather(
            *[self._close_handler(handler) for handler in list(self._handlers.values())]
        )
        self._handlers.clear()

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

        return gen.get_schema(app.routes)

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
        loaded = len(self._handlers)
        idle = sum(1 for h in self._handlers.values() if h.idle_since is not None)
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

        events = self._workflows[name].events + (
            self._additional_events.get(name, []) or []
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
        workflow = self._extract_workflow(request)
        context, start_event, handler_id = await self._extract_run_params(
            request, workflow.workflow, workflow.name
        )

        if start_event is not None:
            input_ev = workflow.workflow.start_event_class.model_validate(start_event)
        else:
            input_ev = None

        try:
            wrapper = await self._start_workflow(
                workflow=_NamedWorkflow(name=workflow.name, workflow=workflow.workflow),
                handler_id=handler_id,
                context=context,
                start_event=input_ev,
            )
            handler = wrapper.run_handler
            try:
                await handler
                status = 200
            except Exception as e:
                status = 500
                logger.error(f"Error running workflow: {e}", exc_info=True)
            if wrapper.task is not None:
                try:
                    await wrapper.task
                except Exception:
                    pass
            # explicitly close handlers from this synchronous api so they don't linger with events
            # that no-one is listening for
            await self._close_handler(wrapper)

            return JSONResponse(
                wrapper.to_response_model().model_dump(), status_code=status
            )
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
            workflow_graph = get_workflow_representation(workflow.workflow)
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
        workflow = self._extract_workflow(request)
        context, start_event, handler_id = await self._extract_run_params(
            request, workflow.workflow, workflow.name
        )

        if start_event is not None:
            input_ev = workflow.workflow.start_event_class.model_validate(start_event)
        else:
            input_ev = None

        try:
            wrapper = await self._start_workflow(
                workflow=_NamedWorkflow(name=workflow.name, workflow=workflow.workflow),
                handler_id=handler_id,
                context=context,
                start_event=input_ev,
            )

        except Exception as e:
            raise HTTPException(
                detail=f"Initial persistence failed: {e}", status_code=500
            )
        return JSONResponse(wrapper.to_response_model().model_dump())

    async def _load_handler(self, handler_id: str) -> HandlerData:
        wrapper = self._handlers.get(handler_id)
        if wrapper is None:
            found = await self._workflow_store.query(
                HandlerQuery(handler_id_in=[handler_id])
            )
            if not found:
                raise HTTPException(detail="Handler not found", status_code=404)
            existing = found[0]
            return _WorkflowHandler.handler_data_from_persistent(existing)
        else:
            if wrapper.run_handler.done() and wrapper.task is not None:
                try:
                    await wrapper.task  # make sure its fully done
                except Exception:
                    # failed workflows raise their exception here
                    pass  # failed workflows raise their exception here

            return wrapper.to_response_model()

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
          Event data is returned as an envelope that preserves backward-compatible fields
          and adds metadata for type-safety on the client:
          {
            "value": <pydantic serialized value>,
            "types": [<class names from MRO excluding the event class and base Event>],
            "type": <class name>,
            "qualified_name": <python module path + class name>,
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
                  description: Server-Sent Events stream of event data.
                  properties:
                    value:
                      type: object
                      description: The event value.
                    type:
                      type: string
                      description: The class name of the event.
                    types:
                      type: array
                      description: Superclass names from MRO (excluding the event class and base Event).
                      items:
                        type: string
                    qualified_name:
                      type: string
                      description: The qualified name of the event.
                  required: [value, type]
          404:
            description: Handler not found
        """
        handler_id = request.path_params["handler_id"]
        timeout = request.query_params.get("acquire_timeout", "1").lower()
        include_internal = (
            request.query_params.get("include_internal", "false").lower() == "true"
        )
        include_qualified_name = (
            request.query_params.get("include_qualified_name", "true").lower() == "true"
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
            # Try to reload from persistence (for released idle workflows)
            handler, persisted = await self._try_reload_handler(handler_id)
            if handler is None:
                if persisted:
                    status = persisted.status
                    if status in {"completed", "failed", "cancelled"}:
                        raise HTTPException(
                            detail="Handler is completed", status_code=204
                        )
                raise HTTPException(detail="Handler not found", status_code=404)
        if handler.queue.empty() and handler.task is not None and handler.task.done():
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
            async for event in generator:
                if not include_internal and isinstance(event, InternalDispatchEvent):
                    continue
                envelope = EventEnvelopeWithMetadata.from_event(
                    event, include_qualified_name=include_qualified_name
                )
                payload = envelope.model_dump_json()
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
        persistent_handlers = await self._workflow_store.query(
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
                    oneOf:
                      - type: string
                        description: JSON string of the event envelope or value.
                        examples:
                          - '{"type": "ExternalEvent", "value": {"response": "hi"}}'
                      - type: object
                        properties:
                          type:
                            type: string
                            description: The class name of the event.
                          value:
                            type: object
                            description: The event value object (preferred over data).
                        additionalProperties: true
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
        if wrapper is not None and is_status_completed(wrapper.status):
            raise HTTPException(detail="Workflow already completed", status_code=409)
        if wrapper is None:
            # Try to reload from persistence (for released idle workflows)
            wrapper, persisted = await self._try_reload_handler(handler_id)
            if wrapper is None:
                # Check if it exists but is completed
                if persisted and is_status_completed(persisted.status):
                    raise HTTPException(
                        detail="Workflow already completed", status_code=409
                    )
                elif persisted is None:
                    raise HTTPException(detail="Handler not found", status_code=404)
                else:
                    # Shouldn't really happen
                    raise HTTPException(
                        detail=f"Failed to resume incomplete handler with status {persisted.status}",
                        status_code=500,
                    )

        # Immediately mark active to cancel the idle timer before it can fire.
        # This prevents a race where the timer releases the handler before we
        # finish processing the event.
        wrapper.mark_active()

        handler = wrapper.run_handler

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

            try:
                event = EventEnvelope.parse(
                    event_str, self._event_registry(wrapper.workflow_name)
                )
            except EventValidationError as e:
                raise HTTPException(detail=str(e), status_code=400)
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

        # Close the handler if it exists (this will cancel and trigger auto-checkpoint)
        if wrapper is not None:
            await self._close_handler(wrapper)

        # Handle persistence
        if purge:
            n_deleted = await self._workflow_store.delete(
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
                        self._event_registry(workflow_name),
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

    async def _start_workflow(
        self,
        workflow: _NamedWorkflow,
        handler_id: str,
        start_event: StartEvent | None = None,
        context: Context | None = None,
        idle_since: datetime | None = None,
    ) -> _WorkflowHandler:
        """Start a workflow and return a wrapper for the handler."""
        with instrument_tags({"handler_id": handler_id}):
            handler = workflow.workflow.run(
                ctx=context,
                start_event=start_event,
            )
            wrapper = await self._run_workflow_handler(
                handler_id, workflow.name, handler, idle_since=idle_since
            )
            return wrapper

    async def _run_workflow_handler(
        self,
        handler_id: str,
        workflow_name: str,
        handler: WorkflowHandler,
        idle_since: datetime | None = None,
    ) -> _WorkflowHandler:
        """
        Creates a wrapper for the handler and starts streaming events.
        """
        queue: asyncio.Queue[Event] = asyncio.Queue()
        started_at = datetime.now(timezone.utc)

        wrapper = _WorkflowHandler(
            run_handler=handler,
            queue=queue,
            task=None,  # Will be set by start_streaming()
            consumer_mutex=asyncio.Lock(),
            handler_id=handler_id,
            workflow_name=workflow_name,
            started_at=started_at,
            updated_at=started_at,
            completed_at=None,
            _workflow_store=self._workflow_store,
            _persistence_backoff=self._persistence_backoff,
            _idle_release_timeout=self._idle_release_timeout,
            _on_idle_release=self._release_handler,
        )
        wrapper.idle_since = idle_since
        # Initial checkpoint before registration; fail fast if persistence is unavailable
        await wrapper.checkpoint()
        # Now register and start streaming
        self._handlers[handler_id] = wrapper

        async def on_finish() -> None:
            self._handlers.pop(handler_id, None)

        wrapper.start_streaming(on_finish=on_finish)

        return wrapper

    async def _close_handler(self, handler: _WorkflowHandler) -> None:
        """Close and cleanup a handler."""
        # Cancel the run_handler if not done
        await handler.cancel_handlers_and_tasks()

        self._handlers.pop(handler.handler_id, None)

    async def _release_handler(self, wrapper: _WorkflowHandler) -> None:
        """Release an idle handler from memory, keeping it in persistence."""
        handler_id = wrapper.handler_id

        # Use lock to coordinate with _try_reload_handler and prevent race
        # where release checkpoint overwrites newer state from reload
        async with self._reload_lock(handler_id):
            # Check if this handler was already replaced by a reload.
            # If a different handler instance is now in _handlers, skip
            # checkpointing to avoid overwriting newer state.
            current = self._handlers.get(handler_id)
            if current is not None and current is not wrapper:
                logger.debug(
                    f"Skipping release checkpoint for {handler_id}: "
                    "handler was already reloaded"
                )
                # Mark to skip any further checkpoints (including auto-checkpoint
                # in _stream_events when we cancel the task)
                # Still need to cancel the old runtime
                wrapper._cancel_idle_release_timer(skip_checkpoint=True)
                await wrapper.cancel_handlers_and_tasks()
                return

            # Remove from memory FIRST to prevent race conditions where cancelling
            # the handler triggers _stream_events to cancel the timer, which would
            # interrupt this method before the pop happens.
            self._handlers.pop(handler_id, None)

            # Cancel the idle release timer to prevent re-entry
            wrapper._cancel_idle_release_timer()

            try:
                # Final checkpoint to ensure latest state is persisted
                await wrapper.checkpoint()
            finally:
                # Always stop the workflow runtime to prevent memory leaks,
                # even if checkpoint fails
                await wrapper.cancel_handlers_and_tasks()

        logger.info(f"Released idle workflow {handler_id} from memory")

    async def _try_reload_handler(
        self, handler_id: str
    ) -> tuple[_WorkflowHandler | None, PersistentHandler | None]:
        """Attempt to reload a released handler from persistence.

        Uses per-handler locking to prevent concurrent reloads from creating
        duplicate workflow instances.

        Returns the persistent handler data if it was fetched from the store, so callers can avoid fetching it again if needed.
        """
        async with self._reload_lock(handler_id):
            # Check if handler was already reloaded by another concurrent request
            if handler_id in self._handlers:
                return self._handlers[handler_id], None

            found = await self._workflow_store.query(
                HandlerQuery(handler_id_in=[handler_id])
            )
            if not found:
                return None, None

            handler_data = found[0]

            if handler_data.status != "running":
                return None, handler_data  # Can't reload completed/failed/cancelled

            workflow = self._workflows.get(handler_data.workflow_name)
            if workflow is None:
                logger.warning(
                    f"Cannot reload {handler_id}: workflow {handler_data.workflow_name} not registered"
                )
                return None, handler_data

            try:
                context = Context.from_dict(workflow=workflow, data=handler_data.ctx)
                wrapper = await self._start_workflow(
                    workflow=_NamedWorkflow(
                        name=handler_data.workflow_name, workflow=workflow
                    ),
                    handler_id=handler_id,
                    context=context,
                    idle_since=handler_data.idle_since,
                )

                # If workflow was idle when released, start the release timer
                # (it will be cancelled if an event wakes the workflow)
                if wrapper.idle_since is not None:
                    wrapper._start_idle_release_timer()

                logger.info(f"Reloaded workflow {handler_id} from persistence")
                return wrapper, handler_data
            except Exception as e:
                logger.error(f"Failed to reload handler {handler_id}: {e}")
                raise HTTPException(
                    detail=f"Failed to reload handler: {e}", status_code=500
                )

    def _event_registry(self, workflow_name: str) -> dict[str, type[Event]]:
        items = {e.__name__: e for e in self._workflows[workflow_name].events}
        items.update(
            {
                e.__name__: e
                for e in self._additional_events.get(workflow_name, None) or []
            }
        )
        return items


@dataclass
class _WorkflowHandler:
    """A wrapper around a handler: WorkflowHandler. Necessary to monitor and dispatch events from the handler's stream_events"""

    run_handler: WorkflowHandler
    queue: asyncio.Queue[Event]
    task: asyncio.Task[None] | None
    # only one consumer of the queue at a time allowed
    consumer_mutex: asyncio.Lock

    # metadata
    handler_id: str
    workflow_name: str
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    # Dependencies for persistence
    _workflow_store: AbstractWorkflowStore
    _persistence_backoff: list[float]
    _on_finish: Callable[[], Awaitable[None]] | None = None
    idle_since: datetime | None = None

    # Idle release support
    _idle_release_timeout: timedelta | None = None
    _on_idle_release: Callable[["_WorkflowHandler"], Awaitable[None]] | None = None
    _idle_release_timer: asyncio.Task[None] | None = None
    _skip_checkpoint: bool = False  # Set to prevent checkpointing stale handlers

    def _as_persistent(self) -> PersistentHandler:
        """Persist the current handler state immediately to the workflow store."""
        self.updated_at = datetime.now(timezone.utc)
        if self.status in ("completed", "failed", "cancelled"):
            self.completed_at = self.updated_at

        persistent = PersistentHandler(
            handler_id=self.handler_id,
            workflow_name=self.workflow_name,
            status=self.status,
            run_id=self.run_handler.run_id,
            error=self.error,
            result=self.result,
            started_at=self.started_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
            idle_since=self.idle_since,
            ctx=self.run_handler.ctx.to_dict() if self.run_handler.ctx else {},
        )
        return persistent

    async def persist(self, persistent: PersistentHandler) -> None:
        await self._workflow_store.update(persistent)

    async def checkpoint(self) -> None:
        """Persist with retry/backoff; cancel handler when retries exhausted."""
        if self._skip_checkpoint:
            logger.debug(f"Skipping checkpoint for handler {self.handler_id}")
            return
        backoffs = list(self._persistence_backoff)
        try:
            persistent = self._as_persistent()
        except Exception as e:
            logger.error(
                f"Failed to checkpoint handler {self.handler_id} to persistent state. Is there non-serializable state in an event or the state store? {e}",
                exc_info=True,
            )
            raise
        while True:
            try:
                await self.persist(persistent)
                return
            except Exception as e:
                backoff = backoffs.pop(0) if backoffs else None
                if backoff is None:
                    logger.error(
                        f"Failed to checkpoint handler {self.handler_id} after final attempt. Failing the handler.",
                        exc_info=True,
                    )
                    # Cancel the underlying workflow; do not re-raise here to allow callers to decide behavior
                    try:
                        self.run_handler.cancel()
                    except Exception:
                        pass
                    raise
                logger.error(
                    f"Failed to checkpoint handler {self.handler_id}. Retrying in {backoff} seconds: {e}"
                )
                await asyncio.sleep(backoff)

    def to_response_model(self) -> HandlerData:
        """Convert runtime handler to API response model."""
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
            result=EventEnvelopeWithMetadata.from_event(self.result)
            if self.result is not None
            else None,
        )

    @staticmethod
    def handler_data_from_persistent(persistent: PersistentHandler) -> HandlerData:
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

    @property
    def status(self) -> Status:
        """Get the current status by inspecting the handler state."""
        if not self.run_handler.done():
            return "running"
        # done - check if cancelled first
        if self.run_handler.cancelled():
            return "cancelled"
        # then check for exception
        exc = self.run_handler.exception()
        if exc is not None:
            return "failed"
        return "completed"

    @property
    def error(self) -> str | None:
        if not self.run_handler.done():
            return None
        try:
            exc = self.run_handler.exception()
        except asyncio.CancelledError:
            return None
        return str(exc) if exc is not None else None

    @property
    def result(self) -> StopEvent | None:
        if not self.run_handler.done():
            return None
        try:
            return self.run_handler.get_stop_event()
        except asyncio.CancelledError:
            return None
        except Exception:
            return None

    def _start_idle_release_timer(self) -> None:
        """Start a timer to release this handler after the idle timeout."""
        if self._idle_release_timeout is None or self._on_idle_release is None:
            return

        # Cancel any existing timer first
        self._cancel_idle_release_timer()

        timeout_seconds = self._idle_release_timeout.total_seconds()

        async def release_after_timeout() -> None:
            try:
                await asyncio.sleep(timeout_seconds)
                # Only release if still idle and no active stream consumers
                if self.idle_since is not None:
                    if self.consumer_mutex.locked():
                        # Mutex is locked - reschedule to try again later
                        self._start_idle_release_timer()
                        return
                    if self._on_idle_release is not None:
                        await self._on_idle_release(self)
            except asyncio.CancelledError:
                pass  # Timer was cancelled, nothing to do

        self._idle_release_timer = asyncio.create_task(release_after_timeout())

    def _cancel_idle_release_timer(self, skip_checkpoint: bool = False) -> None:
        """Cancel any pending idle release timer."""
        if skip_checkpoint:
            self._skip_checkpoint = True
        if self._idle_release_timer is not None:
            self._idle_release_timer.cancel()
            self._idle_release_timer = None

    def mark_idle(self, idle_since: datetime | None = None) -> None:
        self.idle_since = idle_since or datetime.now(timezone.utc)
        self._start_idle_release_timer()

    def mark_active(self) -> None:
        """Mark this handler as active (not idle).

        Call this when an event is being sent to prevent premature release.
        """
        if self.idle_since is not None:
            self.idle_since = None
            self._cancel_idle_release_timer()

    def start_streaming(self, on_finish: Callable[[], Awaitable[None]]) -> None:
        """Start streaming events from the handler and managing state."""
        self.task = asyncio.create_task(self._stream_events(on_finish=on_finish))

    async def _stream_events(self, on_finish: Callable[[], Awaitable[None]]) -> None:
        """Internal method that streams events, updates status, and persists state."""
        with instrument_tags({"handler_id": self.handler_id}):
            await self.checkpoint()
            self._on_finish = on_finish
            async for event in self.run_handler.stream_events(expose_internal=True):
                # Track idle state transitions and manage release timer
                if isinstance(event, WorkflowIdleEvent):
                    self.mark_idle()
                elif isinstance(event, UnhandledEvent):
                    self.mark_idle()
                elif (
                    isinstance(event, StepStateChanged)
                    and event.step_state == StepState.RUNNING
                ):
                    self.mark_active()

                if (  # Watch for a specific internal event that signals the step is complete
                    isinstance(event, StepStateChanged)
                    and event.step_state == StepState.NOT_RUNNING
                ):
                    state = (
                        self.run_handler.ctx.to_dict() if self.run_handler.ctx else None
                    )
                    if state is None:
                        logger.warning(
                            f"Context state is None for handler {self.handler_id}. This is not expected."
                        )
                        continue
                    await self.checkpoint()

                self.queue.put_nowait(event)

            # Workflow is completing - cancel any pending release timer
            self._cancel_idle_release_timer()

            # done when stream events are complete
            try:
                await self.run_handler
            except asyncio.CancelledError:
                # Handler was cancelled - status will be automatically detected via handler.cancelled()
                logger.info(f"Workflow run {self.handler_id} was cancelled")
                # Don't re-raise, just let the task complete
            except Exception as e:
                logger.error(
                    f"Workflow run {self.handler_id} failed! {e}", exc_info=True
                )

            await self.checkpoint()

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
            while not self.queue.empty() or (
                self.task is not None and not self.task.done()
            ):
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
                    {queue_get_task, task_waitable}
                    if task_waitable is not None
                    else {queue_get_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queue_get_task in done:
                    yield await queue_get_task
                else:  # otherwise task completed, so nothing else will be published to the queue
                    queue_get_task.cancel()
                    break
        finally:
            if self._on_finish is not None and self.run_handler.done():
                # clean up the resources if the stream has been consumed
                await self._on_finish()
            self.consumer_mutex.release()

    async def cancel_handlers_and_tasks(self) -> None:
        """Cancel the handler and release it from the store."""
        if not self.run_handler.done():
            try:
                self.run_handler.cancel()
            except Exception:
                pass
            try:
                await self.run_handler.cancel_run()
            except Exception:
                pass
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


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
    print(f"OpenAPI schema written to {args.output}")  # noqa: T201
