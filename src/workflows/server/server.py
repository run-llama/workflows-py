# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
from __future__ import annotations

import asyncio
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
from workflows.events import StopEvent
from workflows.handler import WorkflowHandler

from .utils import nanoid

logger = logging.getLogger()


class WorkflowServer:
    def __init__(self, middleware: list[Middleware] | None = None):
        self._workflows: dict[str, Workflow] = {}
        self._contexts: dict[str, Context] = {}
        self._handlers: dict[str, WorkflowHandler] = {}
        self._results: dict[str, StopEvent] = {}
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
                "/workflows/{name}/start-event",
                self._get_start_event_schema,
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
                "/health",
                self._health_check,
                methods=["GET"],
            ),
        ]

        self.app = Starlette(routes=self._routes, middleware=self._middleware)
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
                    type: string
                    description: Serialized StartEvent in JSON.
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
            request, workflow
        )

        try:
            result = await workflow.run(
                ctx=context, start_event=start_event, **run_kwargs
            )
            return JSONResponse({"result": result})
        except Exception as e:
            raise HTTPException(detail=f"Error running workflow: {e}", status_code=500)

    async def _get_start_event_schema(self, request: Request) -> JSONResponse:
        """
        ---
        summary: Get JSON schema for start event
        description: |
          Gets the JSON schema of the start event from the specified workflow and returns it under "result"
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
                    result:
                      description: JSON schema for the start event
                  required: [result]
          404:
            description: Workflow not found
          500:
            description: Error while getting the JSON schema for start event
        """
        workflow = self._extract_workflow(request)
        try:
            start_event_class = workflow.start_event_class
            return JSONResponse({"result": start_event_class.model_json_schema()})
        except Exception as e:
            raise HTTPException(
                detail=f"Error getting schema of start event for workflow: {e}",
                status_code=500,
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
                    type: string
                    description: Serialized StartEvent in JSON.
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
            request, workflow
        )
        handler_id = nanoid()

        self._handlers[handler_id] = workflow.run(
            ctx=context,
            start_event=workflow.start_event_class.model_validate(start_event),
            **run_kwargs,
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

        handler = self._handlers.get(handler_id)
        if handler is None:
            raise HTTPException(detail="Handler not found", status_code=404)

        if not handler.done():
            return JSONResponse({}, status_code=202)

        try:
            result = await handler
            self._results[handler_id] = result

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

        async def event_stream(handler: WorkflowHandler) -> AsyncGenerator[str, None]:
            serializer = JsonSerializer()

            async for event in handler.stream_events(expose_internal=True):
                serialized_event = serializer.serialize(event)
                if sse:
                    # need to convert back to str to use SSE
                    event_dict = json.loads(serialized_event)
                    yield f"event: {event_dict.get('qualified_name')}\ndata: {json.dumps(event_dict.get('value'))}\n"
                else:
                    yield f"{serialized_event}\n"

                await asyncio.sleep(0)

        return StreamingResponse(event_stream(handler), media_type=media_type)

    #
    # Private methods
    #

    def _extract_workflow(self, request: Request) -> Workflow:
        if "name" not in request.path_params:
            raise HTTPException(detail="'name' parameter missing", status_code=400)
        name = request.path_params["name"]

        if name not in self._workflows:
            raise HTTPException(detail="Workflow not found", status_code=404)

        return self._workflows[name]

    async def _extract_run_params(self, request: Request, workflow: Workflow) -> tuple:
        try:
            body = await request.json()
            context_data = body.get("context")
            run_kwargs = body.get("kwargs", {})
            start_event_str = body.get("start_event")

            # Extract custom StartEvent if present
            start_event = None
            if start_event_str:
                serializer = JsonSerializer()
                try:
                    start_event = serializer.deserialize(start_event_str)
                except Exception as e:
                    raise HTTPException(
                        detail=f"Validation error for 'start_event': {e}",
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
