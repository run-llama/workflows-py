# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import asyncio
import json
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from workflows import Context
from workflows.server import WorkflowServer
from workflows.workflow import Workflow


@pytest.fixture
def server() -> WorkflowServer:
    return WorkflowServer()


@pytest_asyncio.fixture
async def async_client(
    server: WorkflowServer,
    simple_test_workflow: Workflow,
    error_workflow: Workflow,
    streaming_workflow: Workflow,
) -> AsyncGenerator:
    server.add_workflow("test", simple_test_workflow)
    server.add_workflow("error", error_workflow)
    server.add_workflow("streaming", streaming_workflow)
    transport = ASGITransport(app=server.app)
    yield AsyncClient(transport=transport, base_url="http://test")

    # Clean up any running workflows to avoid event loop warnings
    for handler_id, handler in list(server._handlers.items()):
        if not handler.done():
            handler.cancel()
            try:
                await handler
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_list_workflows(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.get("/workflows")
        assert response.status_code == 200
        data = response.json()
        assert "workflows" in data
        assert set(data["workflows"]) == {"test", "error", "streaming"}


@pytest.mark.asyncio
async def test_run_workflow_success(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post(
            "/workflows/test/run", json={"kwargs": {"message": "hello"}}
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"] == "processed: hello"


@pytest.mark.asyncio
async def test_run_workflow_no_kwargs(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post("/workflows/test/run", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "processed: default"


@pytest.mark.asyncio
async def test_run_workflow_with_context(
    async_client: AsyncClient, server: WorkflowServer
) -> None:
    async with async_client as client:
        ctx: Context = Context(server._workflows["test"])
        await ctx.store.set("test_param", "message from context")
        ctx_dict = ctx.to_dict()
        response = await client.post("/workflows/test/run", json={"context": ctx_dict})
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "processed: message from context"


@pytest.mark.asyncio
async def test_run_workflow_with_start_event(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Test with simple start event containing message
        start_event_json = '{"__is_pydantic": true, "value": {"_data": {"message": "start event message"}}, "qualified_name": "workflows.events.StartEvent"}'
        response = await client.post(
            "/workflows/test/run", json={"start_event": start_event_json}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "processed: start event message"


@pytest.mark.asyncio
async def test_run_workflow_not_found(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post("/workflows/nonexistent/run", json={})
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_run_workflow_error(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post("/workflows/error/run", json={})
        assert response.status_code == 500
        assert "Test error" in response.text


@pytest.mark.asyncio
async def test_run_workflow_invalid_json(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post(
            "/workflows/test/run",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_run_workflow_invalid_start_event(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Test with invalid JSON for start_event
        response = await client.post(
            "/workflows/test/run", json={"start_event": "invalid json"}
        )
        assert response.status_code == 400
        assert "Validation error for 'start_event'" in response.text


@pytest.mark.asyncio
async def test_run_workflow_nowait_invalid_start_event(
    async_client: AsyncClient,
) -> None:
    async with async_client as client:
        # Test with invalid JSON for start_event in nowait endpoint
        response = await client.post(
            "/workflows/test/run-nowait", json={"start_event": "invalid json"}
        )
        assert response.status_code == 400
        assert "Validation error for 'start_event'" in response.text


@pytest.mark.asyncio
async def test_run_workflow_with_start_event_and_kwargs(
    async_client: AsyncClient,
) -> None:
    async with async_client as client:
        # Test that start_event takes precedence over kwargs
        start_event_json = '{"__is_pydantic": true, "value": {"_data": {"message": "start event priority"}}, "qualified_name": "workflows.events.StartEvent"}'
        response = await client.post(
            "/workflows/test/run",
            json={
                "start_event": start_event_json,
                "kwargs": {"message": "kwargs message"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        # start_event should take precedence
        assert data["result"] == "processed: start event priority"


@pytest.mark.asyncio
async def test_run_workflow_nowait_success(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post(
            "/workflows/test/run-nowait", json={"kwargs": {"message": "async"}}
        )
        assert response.status_code == 200
        data = response.json()
        assert "handler_id" in data
        assert "status" in data
        assert data["status"] == "started"
        assert len(data["handler_id"]) == 10  # Default nanoid length


@pytest.mark.asyncio
async def test_run_workflow_nowait_with_start_event(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Test with start event containing message
        start_event_json = '{"__is_pydantic": true, "value": {"_data": {"message": "async start event"}}, "qualified_name": "workflows.events.StartEvent"}'
        response = await client.post(
            "/workflows/test/run-nowait", json={"start_event": start_event_json}
        )
        assert response.status_code == 200
        data = response.json()
        assert "handler_id" in data
        assert "status" in data
        assert data["status"] == "started"
        assert len(data["handler_id"]) == 10  # Default nanoid length


@pytest.mark.asyncio
async def test_run_workflow_nowait_not_found(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post("/workflows/nonexistent/run-nowait", json={})
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_workflow_result(
    async_client: AsyncClient, server: WorkflowServer
) -> None:
    # Setup a context to test all the code paths
    ctx: Context = Context(server._workflows["test"])
    await ctx.store.set("test_param", "message from context")
    ctx_dict = ctx.to_dict()
    async with async_client as client:
        # run no-wait
        response = await client.post(
            "/workflows/test/run-nowait", json={"context": ctx_dict}
        )
        assert response.status_code == 200
        data = response.json()
        assert "handler_id" in data
        handler_id = data["handler_id"]

        await asyncio.sleep(0.1)

        # get result
        response = await client.get(f"/results/{handler_id}")
        assert response.status_code == 200

        # Verify the result content
        result_data = response.json()
        assert "result" in result_data
        assert result_data["result"] == "processed: message from context"


@pytest.mark.asyncio
async def test_get_workflow_result_error(
    async_client: AsyncClient, server: WorkflowServer
) -> None:
    async with async_client as client:
        # run no-wait
        response = await client.post("/workflows/error/run-nowait", json={})
        assert response.status_code == 200
        data = response.json()
        assert "handler_id" in data
        handler_id = data["handler_id"]

        await asyncio.sleep(0.1)

        # get result
        response = await client.get(f"/results/{handler_id}")
        assert response.status_code == 500

        # Verify the result content
        result_data = response.json()
        assert "error" in result_data
        assert result_data["error"] == "Test error"


@pytest.mark.asyncio
async def test_get_workflow_result_not_found(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.get("/results/nonexistent")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_stream_events_success(async_client: AsyncClient) -> None:
    """Test streaming events from a workflow."""
    async with async_client as client:
        # Start streaming workflow
        response = await client.post(
            "/workflows/streaming/run-nowait", json={"kwargs": {"count": 3}}
        )
        assert response.status_code == 200
        data = response.json()
        handler_id = data["handler_id"]

        # Stream events
        response = await client.get(f"/events/{handler_id}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

        # Collect streamed events
        events = []
        async for line in response.aiter_lines():
            if line.strip():
                event_data = json.loads(line)
                # Filter out empty events
                if event_data:
                    events.append(event_data)

        # Verify we got the expected events - NDJSON format returns full event objects
        stream_events = [
            e for e in events if "value" in e and "message" in e.get("value", {})
        ]
        assert len(stream_events) == 3
        for i, event in enumerate(stream_events):
            assert "qualified_name" in event
            assert event["qualified_name"] == "tests.server.conftest.StreamEvent"
            assert event["value"]["message"] == f"event_{i}"
            assert event["value"]["sequence"] == i


@pytest.mark.asyncio
async def test_stream_events_sse(async_client: AsyncClient) -> None:
    """Test streaming events using Server-Sent Events format."""
    async with async_client as client:
        # Start streaming workflow
        response = await client.post(
            "/workflows/streaming/run-nowait", json={"kwargs": {"count": 2}}
        )
        assert response.status_code == 200
        data = response.json()
        handler_id = data["handler_id"]

        # Stream events in SSE format
        response = await client.get(f"/events/{handler_id}?sse=true")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Collect streamed events
        events = []
        current_event = {}
        async for line in response.aiter_lines():
            line = line.strip()
            if line.startswith("event: "):
                # Extract event type
                current_event["event_type"] = line.removeprefix(
                    "event: "
                )  # Remove "event: " prefix
            elif line.startswith("data: "):
                # Extract JSON from SSE data line
                event_json = line.removeprefix("data: ")  # Remove "data: " prefix
                event_data = json.loads(event_json)
                # Filter out empty events
                if event_data:
                    current_event["data"] = event_data
                    events.append(current_event.copy())
                    current_event = {}

        # Verify we got event values (not full event objects)
        # SSE format returns event data with event_type field
        stream_events = [
            e
            for e in events
            if "data" in e
            and "message" in e.get("data", {})
            and "sequence" in e.get("data", {})
        ]
        assert len(stream_events) == 2

        for i, event in enumerate(stream_events):
            # SSE format returns event type and data separately
            assert event["event_type"] == "tests.server.conftest.StreamEvent"
            assert event["data"]["message"] == f"event_{i}"
            assert event["data"]["sequence"] == i


@pytest.mark.asyncio
async def test_stream_events_not_found(async_client: AsyncClient) -> None:
    """Test streaming events from non-existent handler."""
    async with async_client as client:
        response = await client.get("/events/nonexistent")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_stream_events_no_events(async_client: AsyncClient) -> None:
    """Test streaming from workflow that emits no events."""
    async with async_client as client:
        # Start simple workflow that doesn't emit events
        response = await client.post(
            "/workflows/test/run-nowait", json={"kwargs": {"message": "test"}}
        )
        assert response.status_code == 200
        data = response.json()
        handler_id = data["handler_id"]

        # Try to stream events
        response = await client.get(f"/events/{handler_id}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

        # Should get no stream events (may get StopEvent)
        events = []
        async for line in response.aiter_lines():
            if line.strip():
                event_data = json.loads(line)
                # Filter out empty events
                if event_data:
                    events.append(event_data)

        # Filter for actual stream events (not workflow control events like StopEvent)
        # Look for events with message and sequence in the value field (StreamEvent pattern)
        stream_events = [
            e
            for e in events
            if "value" in e
            and "message" in e.get("value", {})
            and "sequence" in e.get("value", {})
        ]
        assert len(stream_events) == 0
