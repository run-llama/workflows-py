# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from llama_agents.mcp import MCPToolConfig, MCPWorkflowServer
from llama_agents.server import WorkflowServer
from workflows import Context, Workflow, step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)

# ---------------------------------------------------------------------------
# Workflow definitions
# ---------------------------------------------------------------------------


class SearchStartEvent(StartEvent):
    query: str
    limit: int = 10


class SearchWorkflow(Workflow):
    @step
    async def search(self, ev: SearchStartEvent) -> StopEvent:
        return StopEvent(result=f"results for '{ev.query}' (limit={ev.limit})")


class NoParamWorkflow(Workflow):
    @step
    async def do_work(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


class SearchResult(StopEvent):
    answer: str
    score: float


class CustomStopWorkflow(Workflow):
    @step
    async def search(self, ev: StartEvent) -> SearchResult:
        return SearchResult(answer="hello", score=0.95)


class ErrorWorkflow(Workflow):
    @step
    async def fail(self, ev: StartEvent) -> StopEvent:
        raise ValueError("something went wrong")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_and_mcp(
    workflows: dict[str, Workflow],
    tools: dict[str, MCPToolConfig] | None = None,
    expose_all: bool = False,
    default_config: MCPToolConfig | None = None,
) -> tuple[WorkflowServer, MCPWorkflowServer]:
    """Build a WorkflowServer, register workflows, and wrap with MCPWorkflowServer."""
    server = WorkflowServer()
    for name, wf in workflows.items():
        server.add_workflow(name, wf)
    mcp = MCPWorkflowServer(
        server,
        name="Test",
        tools=tools,
        expose_all=expose_all,
        default_config=default_config,
    )
    return server, mcp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_sync_tool_executes_workflow() -> None:
    """Call the tool with parameters and verify the result string matches expected output."""
    server, mcp = _make_server_and_mcp(
        workflows={"search": SearchWorkflow()},
        tools={"search": MCPToolConfig()},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            result = await client.call_tool("search", {"query": "hello", "limit": 5})
            assert result.content[0].text == "results for 'hello' (limit=5)"  # type: ignore[union-attr]
    finally:
        await server._service.stop()


async def test_tool_schema_matches_start_event_fields() -> None:
    """List tools and verify the tool has the right input schema."""
    server, mcp = _make_server_and_mcp(
        workflows={"search": SearchWorkflow()},
        tools={"search": MCPToolConfig()},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            tools = await client.list_tools()
            assert len(tools) == 1
            tool = tools[0]
            assert tool.name == "search"

            schema = tool.inputSchema
            props = schema["properties"]

            # query: required string
            assert props["query"]["type"] == "string"

            # limit: optional integer with default 10
            assert props["limit"]["type"] == "integer"
            assert props["limit"]["default"] == 10

            # query should be required, limit should not
            required = schema.get("required", [])
            assert "query" in required
            assert "limit" not in required
    finally:
        await server._service.stop()


async def test_no_param_workflow_tool() -> None:
    """Workflow with bare StartEvent; tool takes no parameters and returns result."""
    server, mcp = _make_server_and_mcp(
        workflows={"noop": NoParamWorkflow()},
        tools={"noop": MCPToolConfig()},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            result = await client.call_tool("noop", {})
            assert result.content[0].text == "done"  # type: ignore[union-attr]
    finally:
        await server._service.stop()


async def test_custom_stop_event_result() -> None:
    """Workflow with custom StopEvent subclass; result should be JSON-serialized."""
    server, mcp = _make_server_and_mcp(
        workflows={"custom_stop": CustomStopWorkflow()},
        tools={"custom_stop": MCPToolConfig()},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            result = await client.call_tool("custom_stop", {})
            parsed = json.loads(result.content[0].text)  # type: ignore[union-attr]
            assert parsed["answer"] == "hello"
            assert parsed["score"] == 0.95
    finally:
        await server._service.stop()


async def test_custom_render_result() -> None:
    """Use MCPToolConfig(render_result=...) and verify the custom renderer output."""
    server, mcp = _make_server_and_mcp(
        workflows={"search": SearchWorkflow()},
        tools={"search": MCPToolConfig(render_result=lambda r: "custom")},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            result = await client.call_tool("search", {"query": "test"})
            assert result.content[0].text == "custom"  # type: ignore[union-attr]
    finally:
        await server._service.stop()


async def test_expose_all_workflows() -> None:
    """Use expose_all=True with default_config and verify all workflows get tools."""
    server, mcp = _make_server_and_mcp(
        workflows={
            "search": SearchWorkflow(),
            "noop": NoParamWorkflow(),
        },
        expose_all=True,
        default_config=MCPToolConfig(),
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools}
            assert tool_names == {"search", "noop"}
    finally:
        await server._service.stop()


async def test_tool_name_override() -> None:
    """Use MCPToolConfig(tool_name='custom_name') and verify tool is registered with that name."""
    server, mcp = _make_server_and_mcp(
        workflows={"search": SearchWorkflow()},
        tools={"search": MCPToolConfig(tool_name="custom_name")},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            tools = await client.list_tools()
            assert len(tools) == 1
            assert tools[0].name == "custom_name"

            result = await client.call_tool("custom_name", {"query": "test"})
            assert result.content[0].text == "results for 'test' (limit=10)"  # type: ignore[union-attr]
    finally:
        await server._service.stop()


def test_invalid_workflow_name_raises() -> None:
    """Config references a nonexistent workflow; should raise ValueError at construction."""
    server = WorkflowServer()
    server.add_workflow("real", NoParamWorkflow())
    with pytest.raises(ValueError, match="nonexistent"):
        MCPWorkflowServer(
            server,
            name="Test",
            tools={"nonexistent": MCPToolConfig()},
        )


async def test_error_workflow_raises() -> None:
    """Call tool for a workflow that raises; verify the error propagates."""
    server, mcp = _make_server_and_mcp(
        workflows={"fail": ErrorWorkflow()},
        tools={"fail": MCPToolConfig()},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            with pytest.raises(ToolError, match="something went wrong"):
                await client.call_tool("fail", {})
    finally:
        await server._service.stop()


# ---------------------------------------------------------------------------
# Async/streaming workflow definitions
# ---------------------------------------------------------------------------


class ProgressEvent(Event):
    message: str


class StreamStartEvent(StartEvent):
    topic: str


class StreamingWorkflow(Workflow):
    """Emits multiple ProgressEvents then completes."""

    @step
    async def process(self, ctx: Context, ev: StreamStartEvent) -> StopEvent:
        for i in range(3):
            ctx.write_event_to_stream(ProgressEvent(message=f"step {i}"))
        return StopEvent(result=f"done: {ev.topic}")


class AsyncErrorWorkflow(Workflow):
    """Emits one event then fails."""

    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(message="starting"))
        raise ValueError("async failure")


# ---------------------------------------------------------------------------
# Async mode tests
# ---------------------------------------------------------------------------


async def test_async_tool_executes_and_returns_result() -> None:
    """Async mode tool runs workflow to completion and returns rendered result."""
    server, mcp = _make_server_and_mcp(
        workflows={"stream": StreamingWorkflow()},
        tools={"stream": MCPToolConfig(mode="async")},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            result = await client.call_tool("stream", {"topic": "weather"})
            assert result.content[0].text == "done: weather"  # type: ignore[union-attr]
    finally:
        await server._service.stop()


async def test_async_tool_streams_progress_notifications() -> None:
    """Async mode tool reports workflow events as progress notifications."""
    server, mcp = _make_server_and_mcp(
        workflows={"stream": StreamingWorkflow()},
        tools={"stream": MCPToolConfig(mode="async")},
    )
    await server._service.start()
    try:
        progress_messages: list[str | None] = []

        async def on_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            progress_messages.append(message)

        async with Client(mcp.mcp) as client:
            result = await client.call_tool(
                "stream",
                {"topic": "weather"},
                progress_handler=on_progress,
            )
            assert result.content[0].text == "done: weather"  # type: ignore[union-attr]

        # Should have received handler_id notification + workflow events
        assert len(progress_messages) >= 1
        # First progress notification should contain the handler_id
        assert progress_messages[0] is not None
        assert "handler_id" in progress_messages[0]
        # Should have ProgressEvent messages in the notifications
        event_messages = [m for m in progress_messages if m and "ProgressEvent" in m]
        assert len(event_messages) == 3
    finally:
        await server._service.stop()


async def test_async_tool_handler_id_in_progress() -> None:
    """The handler_id is included in progress messages so the LLM can use it for respond tools."""
    server, mcp = _make_server_and_mcp(
        workflows={"stream": StreamingWorkflow()},
        tools={"stream": MCPToolConfig(mode="async")},
    )
    await server._service.start()
    try:
        handler_ids: list[str] = []

        async def on_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            if message and "handler_id" in message:
                # Extract handler_id from the message
                parsed = json.loads(message)
                handler_ids.append(parsed["handler_id"])

        async with Client(mcp.mcp) as client:
            await client.call_tool(
                "stream",
                {"topic": "test"},
                progress_handler=on_progress,
            )

        assert len(handler_ids) == 1
        # handler_id should be a UUID string
        assert len(handler_ids[0]) == 36  # UUID format
    finally:
        await server._service.stop()


async def test_async_tool_error_propagates() -> None:
    """Async mode tool propagates workflow errors."""
    server, mcp = _make_server_and_mcp(
        workflows={"fail": AsyncErrorWorkflow()},
        tools={"fail": MCPToolConfig(mode="async")},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            with pytest.raises(ToolError, match="async failure"):
                await client.call_tool("fail", {})
    finally:
        await server._service.stop()


async def test_async_tool_filters_internal_events() -> None:
    """Internal dispatch events (StepStateChanged etc.) are not reported as progress."""
    server, mcp = _make_server_and_mcp(
        workflows={"stream": StreamingWorkflow()},
        tools={"stream": MCPToolConfig(mode="async")},
    )
    await server._service.start()
    try:
        progress_messages: list[str | None] = []

        async def on_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            progress_messages.append(message)

        async with Client(mcp.mcp) as client:
            await client.call_tool(
                "stream",
                {"topic": "test"},
                progress_handler=on_progress,
            )

        # None of the progress messages should contain StepStateChanged
        for msg in progress_messages:
            if msg:
                assert "StepStateChanged" not in msg
    finally:
        await server._service.stop()


# ---------------------------------------------------------------------------
# HITL (Human-in-the-loop) workflow definitions
# ---------------------------------------------------------------------------


class HITLStartEvent(StartEvent):
    topic: str


class NameResponse(HumanResponseEvent):
    """Typed human response with a specific field."""

    name: str


class TypedHITLWorkflow(Workflow):
    """HITL workflow using a typed HumanResponseEvent subclass."""

    @step
    async def ask_name(self, ctx: Context, ev: HITLStartEvent) -> InputRequiredEvent:
        await ctx.store.set("topic", ev.topic)
        return InputRequiredEvent(prefix=f"What is your name for topic '{ev.topic}'?")  # type: ignore[call-arg]

    @step
    async def greet(self, ctx: Context, ev: NameResponse) -> StopEvent:
        topic = await ctx.store.get("topic")
        return StopEvent(result=f"Hello {ev.name}, topic={topic}")


class BareHITLWorkflow(Workflow):
    """HITL workflow using bare HumanResponseEvent (no subclass)."""

    @step
    async def ask_input(self, ctx: Context, ev: HITLStartEvent) -> InputRequiredEvent:
        await ctx.store.set("topic", ev.topic)
        return InputRequiredEvent(prefix="Please provide your response.")  # type: ignore[call-arg]

    @step
    async def handle_response(self, ctx: Context, ev: HumanResponseEvent) -> StopEvent:
        topic = await ctx.store.get("topic")
        # With bare HumanResponseEvent, fields are dynamic via DictLikeModel
        response_text = ev.get("response", "no response")
        return StopEvent(result=f"Got '{response_text}' for topic={topic}")


# ---------------------------------------------------------------------------
# HITL tests
# ---------------------------------------------------------------------------


async def test_hitl_workflow_auto_gets_respond_tool() -> None:
    """Register a HITL workflow in async mode. Verify both the main tool and
    a respond tool are auto-registered."""
    server, mcp = _make_server_and_mcp(
        workflows={"chat": TypedHITLWorkflow()},
        tools={"chat": MCPToolConfig(mode="async")},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools}
            assert "chat" in tool_names, f"Main tool missing. Found: {tool_names}"
            assert "chat_respond" in tool_names, (
                f"Respond tool missing. Found: {tool_names}"
            )
    finally:
        await server._service.stop()


async def test_hitl_respond_tool_has_typed_parameters() -> None:
    """Verify the respond tool's schema includes handler_id and the typed
    HumanResponseEvent subclass fields (e.g. 'name')."""
    server, mcp = _make_server_and_mcp(
        workflows={"chat": TypedHITLWorkflow()},
        tools={"chat": MCPToolConfig(mode="async")},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            tools = await client.list_tools()
            respond_tools = [t for t in tools if t.name == "chat_respond"]
            assert len(respond_tools) == 1, "Expected exactly one respond tool"

            schema = respond_tools[0].inputSchema
            props = schema["properties"]
            required = schema.get("required", [])

            # handler_id is required
            assert "handler_id" in props, (
                f"handler_id missing from respond tool schema. Props: {list(props)}"
            )
            assert "handler_id" in required

            # The typed response field 'name' should be present
            assert "name" in props, (
                f"'name' field from NameResponse missing. Props: {list(props)}"
            )
            assert "name" in required
    finally:
        await server._service.stop()


async def test_hitl_full_cycle() -> None:
    """Full integration test: start HITL workflow, receive InputRequiredEvent
    via progress, call the respond tool, and verify workflow completion."""
    server, mcp = _make_server_and_mcp(
        workflows={"chat": TypedHITLWorkflow()},
        tools={"chat": MCPToolConfig(mode="async")},
    )
    await server._service.start()
    try:
        progress_messages: list[str | None] = []
        handler_id_holder: list[str] = []
        got_input_required = asyncio.Event()

        async def on_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            progress_messages.append(message)
            if message and "handler_id" in message:
                parsed = json.loads(message)
                handler_id_holder.append(parsed["handler_id"])
            if message and "InputRequiredEvent" in message:
                got_input_required.set()

        async with Client(mcp.mcp) as client:
            # Start the main tool call in background -- it blocks on the event stream
            task = asyncio.create_task(
                client.call_tool(
                    "chat",
                    {"topic": "weather"},
                    progress_handler=on_progress,
                )
            )
            # Wait for the InputRequiredEvent to appear in progress
            await asyncio.wait_for(got_input_required.wait(), timeout=10)

            assert len(handler_id_holder) >= 1, "handler_id not received"

            # Call the respond tool with the handler_id and typed field
            respond_result = await client.call_tool(
                "chat_respond",
                {
                    "handler_id": handler_id_holder[0],
                    "name": "San Francisco",
                },
            )
            # The respond tool should acknowledge the event was sent
            assert respond_result is not None

            # Wait for the main task to complete
            result = await asyncio.wait_for(task, timeout=10)
            assert result.content[0].text == "Hello San Francisco, topic=weather"  # type: ignore[union-attr]
    finally:
        await server._service.stop()


async def test_hitl_bare_response_event() -> None:
    """Full cycle test with bare HumanResponseEvent (no subclass).
    The respond tool should accept handler_id + arbitrary kwargs."""
    server, mcp = _make_server_and_mcp(
        workflows={"chat": BareHITLWorkflow()},
        tools={"chat": MCPToolConfig(mode="async")},
    )
    await server._service.start()
    try:
        progress_messages: list[str | None] = []
        handler_id_holder: list[str] = []
        got_input_required = asyncio.Event()

        async def on_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            progress_messages.append(message)
            if message and "handler_id" in message:
                parsed = json.loads(message)
                handler_id_holder.append(parsed["handler_id"])
            if message and "InputRequiredEvent" in message:
                got_input_required.set()

        async with Client(mcp.mcp) as client:
            task = asyncio.create_task(
                client.call_tool(
                    "chat",
                    {"topic": "greetings"},
                    progress_handler=on_progress,
                )
            )
            await asyncio.wait_for(got_input_required.wait(), timeout=10)

            assert len(handler_id_holder) >= 1, "handler_id not received"

            # Respond with dynamic kwargs
            respond_result = await client.call_tool(
                "chat_respond",
                {
                    "handler_id": handler_id_holder[0],
                    "response": "hello world",
                },
            )
            assert respond_result is not None

            result = await asyncio.wait_for(task, timeout=10)
            assert result.content[0].text == "Got 'hello world' for topic=greetings"  # type: ignore[union-attr]
    finally:
        await server._service.stop()


async def test_sync_hitl_workflow_no_respond_tool() -> None:
    """Register a HITL workflow in sync mode. Verify only the main tool exists
    (no respond tool), since sync mode cannot handle HITL interactivity."""
    server, mcp = _make_server_and_mcp(
        workflows={"chat": TypedHITLWorkflow()},
        tools={"chat": MCPToolConfig(mode="sync")},
    )
    await server._service.start()
    try:
        async with Client(mcp.mcp) as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools}
            assert "chat" in tool_names, f"Main tool missing. Found: {tool_names}"
            assert "chat_respond" not in tool_names, (
                f"Respond tool should NOT exist in sync mode. Found: {tool_names}"
            )
    finally:
        await server._service.stop()


# ---------------------------------------------------------------------------
# ASGI mounting / integration tests
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _mounted_server(
    workflows: dict[str, Workflow],
    tools: dict[str, MCPToolConfig] | None = None,
    expose_all: bool = False,
    default_config: MCPToolConfig | None = None,
    path: str = "/mcp",
) -> AsyncGenerator[tuple[WorkflowServer, MCPWorkflowServer], None]:
    """Build a WorkflowServer with MCPWorkflowServer mounted, managing lifespan."""
    server = WorkflowServer()
    for name, wf in workflows.items():
        server.add_workflow(name, wf)
    mcp_server = MCPWorkflowServer(
        server,
        name="Test",
        tools=tools,
        expose_all=expose_all,
        default_config=default_config,
        path=path,
    )
    mcp_server.mount()
    # Run the combined lifespan (starts WorkflowService + FastMCP session manager)
    async with server.app.router.lifespan_context(server.app):
        yield server, mcp_server


def _make_asgi_mcp_client(app: Any, mcp_path: str = "/mcp/") -> tuple[Any, Client]:
    """Create a FastMCP Client that connects via ASGI transport to the mounted app."""
    import httpx
    from fastmcp.client.transports import StreamableHttpTransport

    def asgi_client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        **kwargs: Any,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers=headers or {},
            timeout=timeout or httpx.Timeout(30.0),
            auth=auth,
        )

    transport = StreamableHttpTransport(
        url=f"http://testserver{mcp_path}",
        httpx_client_factory=asgi_client_factory,
    )
    return transport, Client(transport)


async def test_mounted_mcp_tool_via_asgi() -> None:
    """MCP endpoint is accessible at the configured path on the WorkflowServer's app."""
    async with _mounted_server(
        workflows={"search": SearchWorkflow()},
        tools={"search": MCPToolConfig()},
    ) as (server, mcp):
        _, client = _make_asgi_mcp_client(server.app)
        async with client:
            result = await client.call_tool("search", {"query": "hello", "limit": 5})
            assert result.content[0].text == "results for 'hello' (limit=5)"  # type: ignore[union-attr]


async def test_mounted_http_and_mcp_coexist() -> None:
    """Both HTTP API and MCP work on the same server."""
    import httpx

    async with _mounted_server(
        workflows={"search": SearchWorkflow()},
        tools={"search": MCPToolConfig()},
    ) as (server, mcp):
        # HTTP API calls
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as http_client:
            # List workflows
            resp = await http_client.get("/workflows")
            assert resp.status_code == 200
            data = resp.json()
            assert "search" in data["workflows"]

            # Health check
            resp = await http_client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"

        # MCP: call tool via mounted endpoint
        _, mcp_client = _make_asgi_mcp_client(server.app)
        async with mcp_client:
            tools = await mcp_client.list_tools()
            assert any(t.name == "search" for t in tools)


async def test_mounted_lifespan_starts_service() -> None:
    """Lifespan properly starts/stops both FastMCP and WorkflowService.

    The combined lifespan triggers both the WorkflowService and FastMCP
    session manager. We verify this by calling an MCP tool without any
    manual service.start() call -- the lifespan handles it.
    """
    async with _mounted_server(
        workflows={"search": SearchWorkflow()},
        tools={"search": MCPToolConfig()},
    ) as (server, mcp):
        _, client = _make_asgi_mcp_client(server.app)
        async with client:
            result = await client.call_tool("search", {"query": "test"})
            assert result.content[0].text == "results for 'test' (limit=10)"  # type: ignore[union-attr]


async def test_mounted_mcp_no_trailing_slash() -> None:
    """MCP endpoint works at /mcp (no trailing slash) -- the path VS Code sends."""
    async with _mounted_server(
        workflows={"search": SearchWorkflow()},
        tools={"search": MCPToolConfig()},
    ) as (server, mcp):
        _, client = _make_asgi_mcp_client(server.app, mcp_path="/mcp")
        async with client:
            result = await client.call_tool("search", {"query": "test"})
            assert result.content[0].text == "results for 'test' (limit=10)"  # type: ignore[union-attr]
