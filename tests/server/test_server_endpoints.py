# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import asyncio
import json
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from tests.server.util import wait_for_passing
from workflows import Context
from workflows.server import WorkflowServer
from workflows.workflow import Workflow

# Prepare the event to send
from workflows.context.serializers import JsonSerializer
from tests.server.conftest import ExternalEvent


@pytest.fixture
def server() -> WorkflowServer:
    return WorkflowServer()


@pytest_asyncio.fixture
async def async_client(
    server: WorkflowServer,
    simple_test_workflow: Workflow,
    error_workflow: Workflow,
    streaming_workflow: Workflow,
    interactive_workflow: Workflow,
) -> AsyncGenerator:
    server.add_workflow("test", simple_test_workflow)
    server.add_workflow("error", error_workflow)
    server.add_workflow("streaming", streaming_workflow)
    server.add_workflow("interactive", interactive_workflow)
    transport = ASGITransport(app=server.app)
    yield AsyncClient(transport=transport, base_url="http://test")
    await server._close()


@pytest.mark.asyncio
async def test_run_workflow_with_start_event_str_plain(
    async_client: AsyncClient,
) -> None:
    async with async_client as client:
        # Provide start_event as a plain JSON string (no discriminators)
        start_event_json = json.dumps({"message": "plain string start"})
        response = await client.post(
            "/workflows/test/run", json={"start_event": start_event_json}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "processed: plain string start"


@pytest.mark.asyncio
async def test_run_workflow_with_start_event_dict_with_discriminators(
    async_client: AsyncClient,
) -> None:
    async with async_client as client:
        # Provide start_event as a dict with pydantic discriminators
        start_event_dict = {
            "__is_pydantic": True,
            "value": {"_data": {"message": "dict with discriminators"}},
            "qualified_name": "workflows.events.StartEvent",
        }
        response = await client.post(
            "/workflows/test/run", json={"start_event": start_event_dict}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "processed: dict with discriminators"


@pytest.mark.asyncio
async def test_run_workflow_with_start_event_dict_plain(
    async_client: AsyncClient,
) -> None:
    async with async_client as client:
        # Provide start_event as a plain dict (no discriminators)
        start_event_dict = {"message": "plain dict start"}
        response = await client.post(
            "/workflows/test/run", json={"start_event": start_event_dict}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "processed: plain dict start"


@pytest.mark.asyncio
async def test_run_workflow_with_nonconforming_start_event_type(
    async_client: AsyncClient,
) -> None:
    async with async_client as client:
        # Provide start_event of a different event type than the workflow's StartEvent
        wrong_event_dict = {
            "__is_pydantic": True,
            "value": {"_data": {"message": "should fail"}},
            "qualified_name": "workflows.events.StopEvent",
        }
        response = await client.post(
            "/workflows/test/run", json={"start_event": wrong_event_dict}
        )
        assert response.status_code == 400
        assert "Start event must be an instance of" in response.text


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
        assert set(data["workflows"]) == {"test", "error", "streaming", "interactive"}


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


@pytest.mark.asyncio
async def test_get_handlers_empty(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.get("/handlers")
        assert response.status_code == 200
        assert response.json() == {"handlers": []}


@pytest.mark.asyncio
async def test_get_handlers_with_running_workflows(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Start multiple workflows
        response1 = await client.post("/workflows/test/run-nowait", json={})
        handler_id1 = response1.json()["handler_id"]

        response2 = await client.post("/workflows/test/run-nowait", json={})
        handler_id2 = response2.json()["handler_id"]

        # Get handlers
        response = await client.get("/handlers")
        assert response.status_code == 200
        handlers = response.json()["handlers"]

        # Should have 2 handlers
        assert len(handlers) == 2
        handler_ids = {handler["handler_id"] for handler in handlers}
        assert handler_id1 in handler_ids
        assert handler_id2 in handler_ids

        # Check all required fields are present
        for handler in handlers:
            assert "handler_id" in handler
            assert "status" in handler
            assert "result" in handler
            assert "error" in handler
            assert handler["status"] == "running"
            assert handler["result"] is None  # Running workflows don't have results yet
            assert handler["error"] is None  # Running workflows don't have errors

        # Wait for workflows to complete to avoid warnings
        for handler_id in [handler_id1, handler_id2]:
            response = await client.get(f"/results/{handler_id}")
            while response.status_code == 202:
                await asyncio.sleep(0.01)
                response = await client.get(f"/results/{handler_id}")


async def validate_result_response(
    handler_id: str, client: AsyncClient, expected_status: int = 200
) -> Any:
    response = await client.get(f"/results/{handler_id}")
    assert response.status_code == expected_status
    return response.json()


@pytest.mark.asyncio
async def test_get_handlers_with_completed_workflow(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Start a workflow and wait for it to complete
        response = await client.post("/workflows/test/run-nowait", json={})
        handler_id = response.json()["handler_id"]

        await wait_for_passing(lambda: validate_result_response(handler_id, client))
        # Get handlers
        response = await client.get("/handlers")
        assert response.status_code == 200
        handlers = response.json()["handlers"]

        # Find our handler
        handler = next(h for h in handlers if h["handler_id"] == handler_id)
        assert handler["status"] == "completed"
        assert handler["result"] == "processed: default"
        assert handler["error"] is None


@pytest.mark.asyncio
async def test_get_handlers_with_failed_workflow(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Start an error workflow
        response = await client.post("/workflows/error/run-nowait", json={})
        handler_id = response.json()["handler_id"]

        # Wait a bit for workflow to fail
        await asyncio.sleep(0.1)

        result = await wait_for_passing(
            lambda: validate_result_response(handler_id, client, 500)
        )
        assert result["error"] == "Test error"

        # Get handlers
        response = await client.get("/handlers")
        assert response.status_code == 200
        handlers = response.json()["handlers"]

        # Find our handler
        handler = next(h for h in handlers if h["handler_id"] == handler_id)
        assert handler["status"] == "failed"
        assert handler["error"] is not None  # Should have an error
        assert "Test error" in handler["error"]  # Check error message
        assert handler["result"] is None  # Failed workflows don't have results


@pytest.mark.asyncio
async def test_post_event_to_running_workflow(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Start an interactive workflow
        response = await client.post("/workflows/interactive/run-nowait", json={})
        assert response.status_code == 200
        handler_id = response.json()["handler_id"]

        # Wait a bit for workflow to start
        await asyncio.sleep(0.1)

        serializer = JsonSerializer()
        event = ExternalEvent(response="Hello from test")
        event_str = serializer.serialize(event)

        # Send the event
        response = await client.post(f"/events/{handler_id}", json={"event": event_str})
        assert response.status_code == 200
        assert response.json() == {"status": "sent"}

        result = await wait_for_passing(
            lambda: validate_result_response(handler_id, client)
        )

        assert result["result"] == "received: Hello from test"


@pytest.mark.asyncio
async def test_get_workflow_result_returns_202_when_pending(
    async_client: AsyncClient,
) -> None:
    async with async_client as client:
        # Start workflow that waits for an external event and thus remains pending
        response = await client.post("/workflows/interactive/run-nowait", json={})
        assert response.status_code == 200
        handler_id = response.json()["handler_id"]

        response = await client.get(f"/results/{handler_id}")
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_get_workflow_result_multiple_times(
    async_client: AsyncClient,
) -> None:
    async with async_client as client:
        # Start and wait for completion
        response = await client.post(
            "/workflows/test/run-nowait", json={"kwargs": {"message": "cache-me"}}
        )
        assert response.status_code == 200
        handler_id = response.json()["handler_id"]

        # First fetch populates cache
        first = await wait_for_passing(
            lambda: validate_result_response(handler_id, client)
        )
        assert first["result"] == "processed: cache-me"

        second = await validate_result_response(handler_id, client)
        assert second == first


@pytest.mark.asyncio
async def test_post_event_handler_not_found(async_client: AsyncClient) -> None:
    async with async_client as client:
        response = await client.post(
            "/events/nonexistent_handler", json={"event": "{}"}
        )
        assert response.status_code == 404
        assert "Handler not found" in response.text


@pytest.mark.asyncio
async def test_post_event_to_completed_workflow(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Start and wait for a simple workflow to complete
        response = await client.post("/workflows/test/run-nowait", json={})
        handler_id = response.json()["handler_id"]

        # Wait for workflow to complete
        response = await client.get(f"/results/{handler_id}")
        while response.status_code == 202:
            await asyncio.sleep(0.01)
            response = await client.get(f"/results/{handler_id}")

        # Try to send event to completed workflow
        response = await client.post(f"/events/{handler_id}", json={"event": "{}"})
        assert response.status_code == 409
        assert "Workflow already completed" in response.text


@pytest.mark.asyncio
async def test_post_event_invalid_event_data(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Start an interactive workflow
        response = await client.post("/workflows/interactive/run-nowait", json={})
        handler_id = response.json()["handler_id"]

        # Send invalid event data
        response = await client.post(
            f"/events/{handler_id}", json={"event": "invalid json"}
        )
        assert response.status_code == 400
        assert "Failed to deserialize event" in response.text


@pytest.mark.asyncio
async def test_post_event_context_not_available(
    async_client: AsyncClient, server: WorkflowServer
) -> None:
    async with async_client as client:
        # Dumb test for code coverage. Inject a dummy handler with no context to trigger 500 path
        wrapper = SimpleNamespace(
            run_handler=SimpleNamespace(done=lambda: False, ctx=None)
        )

        handler_id = "noctx-1"
        server._handlers[handler_id] = wrapper  # type: ignore[assignment]

        try:
            response = await client.post(f"/events/{handler_id}", json={"event": "{}"})
            assert response.status_code == 500
            assert "Context not available" in response.text
        finally:
            server._handlers.pop(handler_id, None)


@pytest.mark.asyncio
async def test_post_event_body_parsing_error(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Start interactive workflow which waits for an event (keeps running)
        response = await client.post("/workflows/interactive/run-nowait", json={})
        assert response.status_code == 200
        handler_id = response.json()["handler_id"]

        # Send invalid JSON body (not JSON), triggers 500 from body parsing
        response = await client.post(
            f"/events/{handler_id}",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 500
        assert "Error processing request" in response.text


@pytest.mark.asyncio
async def test_post_event_missing_event_data(async_client: AsyncClient) -> None:
    async with async_client as client:
        # Start an interactive workflow
        response = await client.post("/workflows/interactive/run-nowait", json={})
        handler_id = response.json()["handler_id"]

        # Send request without event data
        response = await client.post(f"/events/{handler_id}", json={})
        assert response.status_code == 400
        assert "Event data is required" in response.text
