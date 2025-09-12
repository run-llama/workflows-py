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
    StepState,
    StepStateChanged,
    StopEvent,
)
from workflows.handler import WorkflowHandler


from workflows.server.abstract_workflow_store import (
    AbstractWorkflowStore,
    EmptyWorkflowStore,
    HandlerQuery,
    PersistentHandler,
)
from .utils import nanoid

logger = logging.getLogger()


class WorkflowServer:
    def __init__(
        self,
        middleware: list[Middleware] | None = None,
        workflow_store: AbstractWorkflowStore = EmptyWorkflowStore(),
    ):
        self._workflows: dict[str, Workflow] = {}
        self._contexts: dict[str, Context] = {}
        self._handlers: dict[str, _WorkflowHandler] = {}
        self._results: dict[str, StopEvent] = {}
        self._workflow_store = workflow_store
        self._assets_path = Path(__file__).parent / "static"

        self._middleware = middleware or [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
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
        ]

        self.app = Starlette(
            routes=self._routes, middleware=self._middleware, lifespan=self._lifespan
        )
        # Serve the UI as static files
        self.app.mount(
            "/", app=StaticFiles(directory=self._assets_path, html=True), name="ui"
        )

    def add_workflow(self, name: str, workflow: Workflow) -> None:
        self._workflows[name] = workflow

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
        return JSONResponse({"status": "healthy"})

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
                  kwargs:
                    type: object
                    description: Additional keyword arguments for the workflow.
        responses:
          200:
            description: Workflow completed successfully
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    result:
                      description: Workflow result value
                  required: [result]
          400:
            description: Invalid start_event payload
          404:
            description: Workflow not found
          500:
            description: Error running workflow or invalid request body
        """
        workflow = self._extract_workflow(request)
        context, start_event, run_kwargs = await self._extract_run_params(
            request, workflow.workflow
        )

        if start_event is not None:
            input_ev = workflow.workflow.start_event_class.model_validate(start_event)
        else:
            input_ev = None

        try:
            handler_id = nanoid()
            handler = workflow.workflow.run(
                ctx=context, start_event=input_ev, **run_kwargs
            )
            self._run_workflow_handler(handler_id, workflow.name, handler)
            result = await handler
            return JSONResponse({"result": result})
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

        return JSONResponse({"start": start_event_schema, "stop": stop_event_schema})

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
                  kwargs:
                    type: object
                    description: Additional keyword arguments for the workflow.
        responses:
          200:
            description: Workflow started
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    handler_id:
                      type: string
                    status:
                      type: string
                      enum: [started]
                  required: [handler_id, status]
          400:
            description: Invalid start_event payload
          404:
            description: Workflow not found
        """
        workflow = self._extract_workflow(request)
        context, start_event, run_kwargs = await self._extract_run_params(
            request, workflow.workflow
        )
        handler_id = nanoid()

        if start_event is not None:
            input_ev = workflow.workflow.start_event_class.model_validate(start_event)
        else:
            input_ev = None

        self._run_workflow_handler(
            handler_id,
            workflow.name,
            workflow.workflow.run(
                ctx=context,
                start_event=input_ev,
                **run_kwargs,
            ),
        )
        return JSONResponse({"handler_id": handler_id, "status": "started"})

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
                  type: object
                  properties:
                    result:
                      description: Workflow result value
                  required: [result]
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
        """
        handler_id = request.path_params["handler_id"]

        # Immediately return the result if available
        if handler_id in self._results:
            return JSONResponse({"result": self._results[handler_id]})

        wrapper = self._handlers.get(handler_id)
        if wrapper is None:
            raise HTTPException(detail="Handler not found", status_code=404)

        handler = wrapper.run_handler
        if not handler.done():
            return JSONResponse({}, status_code=202)

        try:
            result = await handler
            self._results[handler_id] = result

            if isinstance(result, StopEvent):
                result = result.model_dump()

            return JSONResponse({"result": result})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def _stream_events(self, request: Request) -> StreamingResponse:
        """
        ---
        summary: Stream workflow events
        description: |
          Streams events produced by a workflow execution. Events are emitted as
          newline-delimited JSON by default, or as Server-Sent Events when `sse=true`.
          Event data is formatted according to llama-index's json serializer. For
          pydantic serializable python types, it returns:
          {
            "__is_pydantic": True,
            "value": <pydantic serialized value>,
            "qualified_name": <python path to pydantic class>
          }
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
              default: false
            description: If true, stream as text/event-stream instead of NDJSON.
        responses:
          200:
            description: Streaming started
            content:
              application/x-ndjson:
                schema:
                  type: string
                  description: Newline-delimited JSON stream of events.
              text/event-stream:
                schema:
                  type: string
                  description: Server-Sent Events stream of event data.
          404:
            description: Handler not found
        """
        handler_id = request.path_params["handler_id"]
        handler = self._handlers.get(handler_id)
        if handler is None:
            raise HTTPException(detail="Handler not found", status_code=404)

        # Get raw_event query parameter
        sse = request.query_params.get("sse", "false").lower() == "true"
        media_type = "text/event-stream" if sse else "application/x-ndjson"

        async def event_stream(handler: _WorkflowHandler) -> AsyncGenerator[str, None]:
            serializer = JsonSerializer()

            async for event in handler.iter_events():
                serialized_event = serializer.serialize(event)
                if sse:
                    # need to convert back to str to use SSE
                    event_dict = json.loads(serialized_event)
                    yield f"event: {event_dict.get('qualified_name')}\ndata: {json.dumps(event_dict.get('value'))}\n"
                else:
                    yield f"{serialized_event}\n"

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
                  type: object
                  properties:
                    handlers:
                      type: array
                      items:
                        type: object
                        properties:
                          handler_id:
                            type: string
                          result:
                            type: object
                          error:
                            type: object
                          status:
                            type: string
                            enum: [running, completed, failed]
                  required: [handlers]
        """
        handlers = []
        for handler_id in self._handlers.keys():
            handler = self._handlers[handler_id].run_handler
            status = "running"
            result = None
            error = None

            if handler.done():
                try:
                    result = handler.result()
                    status = "completed"
                except Exception as e:
                    error = str(e)
                    status = "failed"

            handler_json = {
                "handler_id": handler_id,
                "status": status,
                "result": result,
                "error": error,
            }
            handlers.append(handler_json)

        return JSONResponse({"handlers": handlers})

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
                event = serializer.deserialize(event_str)
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

            return JSONResponse({"status": "sent"})

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                detail=f"Error processing request: {e}", status_code=500
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

    async def _extract_run_params(self, request: Request, workflow: Workflow) -> tuple:
        try:
            body = await request.json()
            context_data = body.get("context")
            run_kwargs = body.get("kwargs", {})
            start_event_data = body.get("start_event")

            # Extract custom StartEvent if present
            start_event = None
            if start_event_data:
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

            return (context, start_event, run_kwargs)

        except HTTPException:
            # Re-raise HTTPExceptions as-is (like start_event validation errors)
            raise
        except Exception as e:
            raise HTTPException(
                detail=f"Error processing request body: {e}", status_code=500
            )

    @asynccontextmanager
    async def _lifespan(self, _: Starlette) -> AsyncGenerator[None, None]:
        # checking the store for any incomplete runs and restart them
        await self._initialize_active_handlers()
        yield
        # cancel running workflows
        await self._close()

    async def _close(self) -> None:
        for handler in self._handlers.values():
            if not handler.run_handler.done():
                try:
                    handler.run_handler.cancel()
                except Exception:
                    pass
                try:
                    await handler.run_handler.cancel_run()
                except Exception:
                    pass
            if not handler.task.done():
                try:
                    handler.task.cancel()
                except Exception:
                    pass

    async def _initialize_active_handlers(self) -> None:
        """Resumes previously running workflows, if they were not complete at last shutdown"""
        handlers = await self._workflow_store.query(
            HandlerQuery(
                status_in=["running"], workflow_name_in=list(self._workflows.keys())
            )
        )
        for persistent in handlers:
            workflow = self._workflows[persistent.workflow_name]
            ctx = Context.from_dict(workflow=workflow, data=persistent.ctx)
            handler = workflow.run(ctx=ctx)
            self._run_workflow_handler(
                persistent.handler_id, persistent.workflow_name, handler
            )

    def _run_workflow_handler(
        self, handler_id: str, workflow_name: str, handler: WorkflowHandler
    ) -> None:
        """
        Streams events from the handler, persisting them, and pushing them to a queue.
        Stores a _WorkflowHandler helper that wraps the handler with it's queue and streaming task.
        """
        queue: asyncio.Queue[Event] = asyncio.Queue()

        async def _stream_events() -> None:
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
                    await self._workflow_store.update(
                        PersistentHandler(
                            handler_id=handler_id,
                            workflow_name=workflow_name,
                            status="running",
                            ctx=state,
                        )
                    )
                queue.put_nowait(event)
            # done when stream events are complete
            try:
                await handler
                status = "completed"
            except Exception:
                status = "failed"

            if handler.ctx is None:
                logger.warning(
                    f"Context is None for handler {handler_id}. This is not expected."
                )
                return
            await self._workflow_store.update(
                PersistentHandler(
                    handler_id=handler_id,
                    workflow_name=workflow_name,
                    status=status,
                    ctx=handler.ctx.to_dict(),
                )
            )

        task = asyncio.create_task(_stream_events())
        self._handlers[handler_id] = _WorkflowHandler(handler, queue, task)


@dataclass
class _WorkflowHandler:
    """A wrapper around a handler: WorkflowHandler. Necessary to monitor and dispatch events from the handler's stream_events"""

    run_handler: WorkflowHandler
    queue: asyncio.Queue[Event]
    task: asyncio.Task[None]

    async def iter_events(self) -> AsyncGenerator[Event, None]:
        """
        Converts the queue to an async generator while the workflow is still running, and there are still events.
        For better or worse, multiple consumers will compete for events
        """
        while not self.queue.empty() or not self.task.done():
            available_events = []
            while not self.queue.empty():
                available_events.append(self.queue.get_nowait())
            for event in available_events:
                yield event
            queue_get_task: asyncio.Task[Event] = asyncio.create_task(self.queue.get())
            task_waitable = self.task
            done, pending = await asyncio.wait(
                {queue_get_task, task_waitable}, return_when=asyncio.FIRST_COMPLETED
            )
            if queue_get_task in done:
                yield await queue_get_task
            else:  # otherwise task completed, so nothing else will be published to the queue
                queue_get_task.cancel()
                break


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
