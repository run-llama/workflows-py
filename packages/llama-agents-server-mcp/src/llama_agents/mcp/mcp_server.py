# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import inspect
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.server.context import Context
from fastmcp.tools import Tool
from llama_agents.server._handler import _NamedWorkflow, _WorkflowHandler
from llama_agents.server._service import _WorkflowService
from llama_agents.server.server import WorkflowServer
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send
from workflows.events import HumanResponseEvent, InternalDispatchEvent, StopEvent

from .tool_config import MCPToolConfig

logger = logging.getLogger(__name__)


class MCPWorkflowServer:
    """Exposes workflows registered on a WorkflowServer as MCP tools via FastMCP."""

    def __init__(
        self,
        server: WorkflowServer,
        *,
        name: str = "Workflow Tools",
        tools: dict[str, MCPToolConfig] | None = None,
        expose_all: bool = False,
        default_config: MCPToolConfig | None = None,
        path: str = "/mcp",
    ) -> None:
        self._server = server
        self._service: _WorkflowService = server._service
        self._path = path
        self._mcp = FastMCP(name)

        # Determine which workflows to expose
        if tools is not None:
            workflow_configs = dict(tools)
        elif expose_all:
            cfg = default_config or MCPToolConfig()
            workflow_configs = {name: cfg for name in self._service._workflows}
        else:
            raise ValueError(
                "Either 'tools' must be provided or 'expose_all' must be True."
            )

        # Validate all named workflows exist
        registered = set(self._service._workflows.keys())
        for workflow_name in workflow_configs:
            if workflow_name not in registered:
                raise ValueError(
                    f"Workflow '{workflow_name}' is not registered on the server. "
                    f"Available workflows: {sorted(registered)}"
                )

        # Register each workflow as an MCP tool
        for workflow_name, config in workflow_configs.items():
            self._register_tool(workflow_name, config)

            # For async HITL workflows, auto-register respond tools
            if config.mode == "async":
                self._register_respond_tools(workflow_name, config)

    @property
    def mcp(self) -> FastMCP:
        return self._mcp

    @property
    def app(self) -> Starlette:
        """The FastMCP ASGI app for standalone usage."""
        return self._mcp.http_app()

    def mount(self) -> None:
        """Mount the FastMCP app into the WorkflowServer's Starlette app.

        Intercepts requests whose path matches ``self._path`` (with or
        without a trailing slash) and dispatches them to the FastMCP ASGI
        app.  All other requests pass through to the original WorkflowServer
        middleware stack.  Also wraps the parent app's lifespan so the
        FastMCP session manager is initialised alongside the
        WorkflowService.
        """
        path = self._path.rstrip("/")
        path_slash = path + "/"
        mcp_asgi_app = self._mcp.http_app(path=path)
        parent_app = self._server.app

        # --- lifespan --------------------------------------------------
        # Wrap the existing lifespan so that FastMCP's session-manager
        # lifespan runs alongside the WorkflowService start/stop.
        original_lifespan = parent_app.router.lifespan_context
        mcp_lifespan = mcp_asgi_app.router.lifespan_context

        @asynccontextmanager
        async def combined_lifespan(
            app: Starlette,
        ) -> AsyncGenerator[None, None]:
            async with original_lifespan(app):
                async with mcp_lifespan(app):
                    yield

        parent_app.router.lifespan_context = combined_lifespan

        # --- request dispatch ------------------------------------------
        # Build the original middleware stack so we can delegate non-MCP
        # requests to it.  Then replace the parent app's middleware stack
        # with a thin dispatcher that routes MCP paths to the FastMCP app.
        original_stack: ASGIApp = parent_app.build_middleware_stack()

        async def _mcp_dispatch(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] in ("http", "websocket"):
                req_path: str = scope.get("path", "")
                if req_path == path or req_path.startswith(path_slash):
                    # Normalize trailing-slash so the FastMCP Route at
                    # ``path`` always matches (avoids a redirect round-trip
                    # that some MCP clients don't follow).
                    if req_path == path_slash:
                        scope = dict(scope, path=path)
                    await mcp_asgi_app(scope, receive, send)
                    return
            await original_stack(scope, receive, send)

        parent_app.middleware_stack = _mcp_dispatch

    def _register_tool(self, workflow_name: str, config: MCPToolConfig) -> None:
        """Register a single workflow as an MCP tool on the FastMCP instance."""
        workflow = self._service._workflows[workflow_name]
        start_event_class = workflow.start_event_class

        # Build parameter list from the start event's model fields
        parameters: list[inspect.Parameter] = []
        for field_name, field_info in start_event_class.model_fields.items():
            # Skip internal/private fields defensively
            if field_name.startswith("_"):
                continue

            if field_info.is_required():
                default = inspect.Parameter.empty
            else:
                default = field_info.default

            param = inspect.Parameter(
                name=field_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=field_info.annotation,
            )
            parameters.append(param)

        # Capture references for the closure
        service = self._service
        render_fn = config.render_result or self._render_result

        if config.mode == "async":
            fn = self._make_async_wrapper(
                workflow_name, workflow, start_event_class, service, render_fn
            )
        else:
            fn = self._make_sync_wrapper(
                workflow_name, workflow, start_event_class, service, render_fn
            )

        # Set function metadata for FastMCP introspection
        tool_name = config.tool_name or workflow_name
        description = config.description
        tags = config.tags

        # For async mode, include the Context parameter so FastMCP can detect
        # and inject it. without_injected_parameters() will strip it from the
        # tool's input schema.
        sig_params = list(parameters)
        annotations: dict[str, Any] = {
            p.name: p.annotation
            for p in parameters
            if p.annotation is not inspect.Parameter.empty
        }
        if config.mode == "async":
            ctx_param = inspect.Parameter(
                name="ctx",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Context,
            )
            sig_params.insert(0, ctx_param)
            annotations["ctx"] = Context

        sig = inspect.Signature(parameters=sig_params, return_annotation=str)
        fn.__signature__ = sig  # type: ignore[attr-defined]
        fn.__name__ = tool_name
        fn.__qualname__ = tool_name
        fn.__doc__ = description
        # FastMCP uses get_type_hints() which reads __annotations__; set them
        # explicitly so the hints match the custom signature.
        annotations["return"] = str
        fn.__annotations__ = annotations

        tool = Tool.from_function(
            fn=fn,
            name=tool_name,
            description=description,
            tags=tags,
        )
        self._mcp.add_tool(tool)

    @staticmethod
    def _make_sync_wrapper(
        workflow_name: str,
        workflow: Any,
        start_event_class: type,
        service: _WorkflowService,
        render_fn: Any,
    ) -> Any:
        """Create an async tool wrapper for sync (blocking) execution."""

        async def _tool_wrapper(**kwargs: Any) -> str:
            start_event = start_event_class(**kwargs)
            handler_id = str(uuid4())
            named_workflow = _NamedWorkflow(name=workflow_name, workflow=workflow)
            wrapper = await service.start_workflow(
                workflow=named_workflow,
                handler_id=handler_id,
                start_event=start_event,
            )
            try:
                # Wait for the workflow handler to complete
                try:
                    await wrapper.run_handler
                except Exception:
                    pass  # Error is captured via wrapper.error below
                # Also await the streaming task if present
                if wrapper.task is not None:
                    try:
                        await wrapper.task
                    except Exception:
                        pass
                # Check for errors
                error = wrapper.error
                if error is not None:
                    raise RuntimeError(error)
                # Get the result
                stop_event = wrapper.result
                if stop_event is None:
                    return "Workflow completed with no result."
                return render_fn(stop_event)
            finally:
                await service.close_handler(wrapper)

        return _tool_wrapper

    @staticmethod
    def _make_async_wrapper(
        workflow_name: str,
        workflow: Any,
        start_event_class: type,
        service: _WorkflowService,
        render_fn: Any,
    ) -> Any:
        """Create an async tool wrapper that streams events as progress notifications."""

        async def _tool_wrapper(ctx: Context, **kwargs: Any) -> str:
            start_event = start_event_class(**kwargs)
            handler_id = str(uuid4())
            named_workflow = _NamedWorkflow(name=workflow_name, workflow=workflow)
            wrapper = await service.start_workflow(
                workflow=named_workflow,
                handler_id=handler_id,
                start_event=start_event,
            )
            try:
                # Report handler_id as first progress notification
                await ctx.report_progress(
                    0,
                    message=json.dumps({"handler_id": handler_id}),
                )

                # Acquire the event stream and iterate events
                progress_counter = 1
                event_stream = await wrapper.acquire_events_stream(timeout=10)
                async for event in event_stream:
                    # Filter out internal dispatch events (StepStateChanged, etc.)
                    if isinstance(event, InternalDispatchEvent):
                        continue
                    # Skip StopEvent; we handle it via wrapper.result
                    if isinstance(event, StopEvent):
                        continue

                    # Format event as an LLM-readable progress message
                    event_type = type(event).__name__
                    try:
                        event_data = event.model_dump()
                    except Exception:
                        event_data = str(event)
                    message = json.dumps({"event": event_type, "data": event_data})
                    await ctx.report_progress(progress_counter, message=message)
                    progress_counter += 1

                # Check for errors
                error = wrapper.error
                if error is not None:
                    raise RuntimeError(error)

                # Get the result
                stop_event = wrapper.result
                if stop_event is None:
                    return "Workflow completed with no result."
                return render_fn(stop_event)
            finally:
                await service.close_handler(wrapper)

        return _tool_wrapper

    def _register_respond_tools(
        self, workflow_name: str, config: MCPToolConfig
    ) -> None:
        """Auto-register respond tools for HITL workflows.

        Detects HumanResponseEvent subclasses consumed by the workflow's steps
        and creates a typed respond tool for each.
        """
        workflow = self._service._workflows[workflow_name]

        # Find HumanResponseEvent subclasses in the workflow's events
        response_event_classes: list[type[HumanResponseEvent]] = []
        for event_type in workflow.events:
            if (
                issubclass(event_type, HumanResponseEvent)
                and event_type is not HumanResponseEvent
            ):
                response_event_classes.append(event_type)

        # If no subclass found, check if bare HumanResponseEvent is used
        if not response_event_classes:
            for event_type in workflow.events:
                if event_type is HumanResponseEvent:
                    response_event_classes.append(HumanResponseEvent)
                    break

        if not response_event_classes:
            return

        service = self._service
        tool_name_base = config.tool_name or workflow_name

        for response_event_class in response_event_classes:
            # Build tool name
            if len(response_event_classes) == 1:
                respond_tool_name = f"{tool_name_base}_respond"
            else:
                # Multiple response event types: include event class name
                respond_tool_name = (
                    f"{tool_name_base}_respond_{response_event_class.__name__}"
                )

            # Build parameters: handler_id + response event fields
            parameters: list[inspect.Parameter] = [
                inspect.Parameter(
                    name="handler_id",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                ),
            ]

            has_typed_fields = False
            for field_name, field_info in response_event_class.model_fields.items():
                if field_name.startswith("_"):
                    continue
                has_typed_fields = True
                if field_info.is_required():
                    default = inspect.Parameter.empty
                else:
                    default = field_info.default
                param = inspect.Parameter(
                    name=field_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=field_info.annotation,
                )
                parameters.append(param)

            # For bare HumanResponseEvent with no typed fields, add a
            # convenience `response` parameter since that's the common pattern.
            if not has_typed_fields:
                parameters.append(
                    inspect.Parameter(
                        name="response",
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=str,
                    )
                )

            fn = self._make_respond_wrapper(
                workflow_name, response_event_class, service
            )

            # Set function metadata
            sig = inspect.Signature(parameters=parameters, return_annotation=str)
            fn.__signature__ = sig  # type: ignore[attr-defined]
            fn.__name__ = respond_tool_name
            fn.__qualname__ = respond_tool_name
            fn.__doc__ = (
                f"Send a response to a running '{workflow_name}' workflow that "
                f"is waiting for human input."
            )
            annotations: dict[str, Any] = {
                p.name: p.annotation
                for p in parameters
                if p.annotation is not inspect.Parameter.empty
            }
            annotations["return"] = str
            fn.__annotations__ = annotations

            tool = Tool.from_function(
                fn=fn,
                name=respond_tool_name,
                description=fn.__doc__,
                tags=config.tags,
            )
            self._mcp.add_tool(tool)

    @staticmethod
    def _make_respond_wrapper(
        workflow_name: str,
        response_event_class: type[HumanResponseEvent],
        service: _WorkflowService,
    ) -> Any:
        """Create an async wrapper for the respond tool."""

        async def _respond_wrapper(handler_id: str, **kwargs: Any) -> str:
            # Look up handler in service
            wrapper: _WorkflowHandler | None = service._handlers.get(handler_id)
            if wrapper is None:
                # Try to reload from persistence
                wrapper, _ = await service.try_reload_handler(handler_id)
            if wrapper is None:
                raise RuntimeError(
                    f"Handler '{handler_id}' not found. "
                    "The workflow may have completed or been released."
                )

            # Mark active to prevent idle release during event send
            wrapper.mark_active()

            # Construct the response event
            event = response_event_class(**kwargs)

            # Send the event into the workflow
            await wrapper.run_handler.send_event(event)

            return f"Response sent to workflow '{workflow_name}'."

        return _respond_wrapper

    @staticmethod
    def _render_result(result: Any) -> str:
        """Default renderer that converts a StopEvent or value to a string."""
        if isinstance(result, StopEvent):
            if type(result) is StopEvent:
                # Base StopEvent: the meaningful value is .result
                inner = result.result
                if isinstance(inner, BaseModel):
                    return inner.model_dump_json()
                return str(inner)
            else:
                # Subclass of StopEvent: the event itself is the result
                return result.model_dump_json()
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return str(result)
