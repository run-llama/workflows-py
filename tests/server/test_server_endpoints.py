# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import asyncio
from collections import Counter
import json
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient, Response

from tests.server.util import wait_for_passing
from workflows import Context
from workflows.server import WorkflowServer
from workflows.workflow import Workflow
from datetime import datetime

# Prepare the event to send
from workflows.context.serializers import JsonSerializer
from tests.server.conftest import ExternalEvent


@pytest.fixture
def server() -> WorkflowServer:
    return WorkflowServer()


@pytest_asyncio.fixture
async def client(
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
    async with server.contextmanager():
        transport = ASGITransport(app=server.app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.mark.asyncio
async def test_run_workflow_with_start_event_str_plain(
    client: AsyncClient,
) -> None:
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
    client: AsyncClient,
) -> None:
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
    client: AsyncClient,
) -> None:
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
    client: AsyncClient,
) -> None:
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
async def test_health_check(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_list_workflows(client: AsyncClient) -> None:
    response = await client.get("/workflows")
    assert response.status_code == 200
    data = response.json()
    assert "workflows" in data
    assert set(data["workflows"]) == {"test", "error", "streaming", "interactive"}


@pytest.mark.asyncio
async def test_run_workflow_success(client: AsyncClient) -> None:
    response = await client.post(
        "/workflows/test/run", json={"kwargs": {"message": "hello"}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == "processed: hello"


@pytest.mark.asyncio
async def test_run_workflow_no_kwargs(client: AsyncClient) -> None:
    response = await client.post("/workflows/test/run", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "processed: default"


@pytest.mark.asyncio
async def test_run_workflow_with_context(
    client: AsyncClient, server: WorkflowServer
) -> None:
    ctx: Context = Context(server._workflows["test"])
    await ctx.store.set("test_param", "message from context")
    ctx_dict = ctx.to_dict()
    response = await client.post("/workflows/test/run", json={"context": ctx_dict})
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "processed: message from context"


@pytest.mark.asyncio
async def test_run_workflow_with_start_event(client: AsyncClient) -> None:
    # Test with simple start event containing message
    start_event_json = '{"__is_pydantic": true, "value": {"_data": {"message": "start event message"}}, "qualified_name": "workflows.events.StartEvent"}'
    response = await client.post(
        "/workflows/test/run", json={"start_event": start_event_json}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "processed: start event message"


@pytest.mark.asyncio
async def test_run_workflow_not_found(client: AsyncClient) -> None:
    response = await client.post("/workflows/nonexistent/run", json={})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_run_workflow_error(client: AsyncClient) -> None:
    response = await client.post("/workflows/error/run", json={})
    assert response.status_code == 500
    assert "Test error" in response.text


@pytest.mark.asyncio
async def test_run_workflow_invalid_json(client: AsyncClient) -> None:
    response = await client.post(
        "/workflows/test/run",
        content="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_run_workflow_invalid_start_event(client: AsyncClient) -> None:
    # Test with invalid JSON for start_event
    response = await client.post(
        "/workflows/test/run", json={"start_event": "invalid json"}
    )
    assert response.status_code == 400
    assert "Validation error for 'start_event'" in response.text


@pytest.mark.asyncio
async def test_run_workflow_nowait_invalid_start_event(
    client: AsyncClient,
) -> None:
    # Test with invalid JSON for start_event in nowait endpoint
    response = await client.post(
        "/workflows/test/run-nowait", json={"start_event": "invalid json"}
    )
    assert response.status_code == 400
    assert "Validation error for 'start_event'" in response.text


@pytest.mark.asyncio
async def test_run_workflow_with_start_event_and_kwargs(
    client: AsyncClient,
) -> None:
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
async def test_run_workflow_nowait_success(client: AsyncClient) -> None:
    response = await client.post(
        "/workflows/test/run-nowait", json={"kwargs": {"message": "async"}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "handler_id" in data
    assert "status" in data
    assert data["status"] == "running"
    assert len(data["handler_id"]) == 10  # Default nanoid length


@pytest.mark.asyncio
async def test_run_workflow_nowait_with_start_event(client: AsyncClient) -> None:
    # Test with start event containing message
    start_event_json = '{"__is_pydantic": true, "value": {"_data": {"message": "async start event"}}, "qualified_name": "workflows.events.StartEvent"}'
    response = await client.post(
        "/workflows/test/run-nowait", json={"start_event": start_event_json}
    )
    assert response.status_code == 200
    data = response.json()
    assert "handler_id" in data
    assert "status" in data
    assert data["status"] == "running"
    assert len(data["handler_id"]) == 10  # Default nanoid length


@pytest.mark.asyncio
async def test_run_workflow_nowait_not_found(client: AsyncClient) -> None:
    response = await client.post("/workflows/nonexistent/run-nowait", json={})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_workflow_result(client: AsyncClient, server: WorkflowServer) -> None:
    # Setup a context to test all the code paths
    ctx: Context = Context(server._workflows["test"])
    await ctx.store.set("test_param", "message from context")
    ctx_dict = ctx.to_dict()
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
    client: AsyncClient, server: WorkflowServer
) -> None:
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
    # Now returns detailed error string body
    assert "Test error" in response.text


@pytest.mark.asyncio
async def test_get_workflow_result_not_found(client: AsyncClient) -> None:
    response = await client.get("/results/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_stream_events_success(client: AsyncClient) -> None:
    """Test streaming events from a workflow."""
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
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Collect streamed events
    events: list[dict[str, Any]] = []
    async for line in response.aiter_lines():
        if line.strip():
            event_data = json.loads(line.removeprefix("data: "))
            assert isinstance(event_data, dict)
            # Filter out empty events
            if event_data:
                events.append(event_data)

    stream_events = [
        e for e in events if e["qualified_name"] == "tests.server.conftest.StreamEvent"
    ]
    assert len(stream_events) == 3
    for i, event in enumerate(stream_events):
        assert "qualified_name" in event
        assert event["value"]["message"] == f"event_{i}"
        assert event["value"]["sequence"] == i


@pytest.mark.asyncio
async def test_stream_events_sse(client: AsyncClient) -> None:
    """Test streaming events using Server-Sent Events format."""
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
            current_event["event_type"] = line.removeprefix("event: ")
        elif line.startswith("data: "):
            # Extract JSON from SSE data line
            event_json = line.removeprefix("data: ")
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
        if e["data"]["qualified_name"] == "tests.server.conftest.StreamEvent"
    ]
    assert len(stream_events) == 2

    for i, event in enumerate(stream_events):
        # SSE format returns event type and data separately
        assert "event_type" not in event
        assert event["data"]["value"]["message"] == f"event_{i}"
        assert event["data"]["value"]["sequence"] == i

    # stream completed
    response = await client.get(f"/events/{handler_id}?sse=true")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_stream_events_not_found(client: AsyncClient) -> None:
    """Test streaming events from non-existent handler."""
    response = await client.get("/events/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_stream_events_no_events(client: AsyncClient) -> None:
    """Test streaming from workflow that emits no events."""
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
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Should get no stream events (may get StopEvent)
    events = []
    async for line in response.aiter_lines():
        if line.strip():
            event_data = json.loads(line.removeprefix("data: "))
            # Filter out empty events
            if event_data:
                events.append(event_data)

    # Only control events and a final StopEvent
    event_types = [events["qualified_name"] for events in events]
    seen_types = set(event_types)
    assert seen_types == {
        "workflows.events.StopEvent",
        "workflows.events.EventsQueueChanged",
        "workflows.events.StepStateChanged",
    }
    assert event_types[-1] == "workflows.events.StopEvent"
    assert Counter(event_types)["workflows.events.StopEvent"] == 1


@pytest.mark.asyncio
async def test_get_handlers_empty(client: AsyncClient) -> None:
    response = await client.get("/handlers")
    assert response.status_code == 200
    assert response.json() == {"handlers": []}


@pytest.mark.asyncio
async def test_get_handlers_with_running_workflows(client: AsyncClient) -> None:
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
    return response.json() if expected_status == 200 else response.text


@pytest.mark.asyncio
async def test_get_handlers_with_completed_workflow(client: AsyncClient) -> None:
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
async def test_get_handlers_with_failed_workflow(client: AsyncClient) -> None:
    # Start an error workflow
    response = await client.post("/workflows/error/run-nowait", json={})
    handler_id = response.json()["handler_id"]

    # Wait a bit for workflow to fail
    await asyncio.sleep(0.1)

    result = await wait_for_passing(
        lambda: validate_result_response(handler_id, client, 500)
    )
    assert "Test error" in result

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
async def test_post_event_to_running_workflow(client: AsyncClient) -> None:
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
    client: AsyncClient,
) -> None:
    # Start workflow that waits for an external event and thus remains pending
    response = await client.post("/workflows/interactive/run-nowait", json={})
    assert response.status_code == 200
    handler_id = response.json()["handler_id"]

    response = await client.get(f"/results/{handler_id}")
    assert response.status_code == 202


@pytest.mark.asyncio
async def test_get_workflow_result_multiple_times(
    client: AsyncClient,
) -> None:
    # Start and wait for completion
    response = await client.post(
        "/workflows/test/run-nowait", json={"kwargs": {"message": "cache-me"}}
    )
    assert response.status_code == 200
    handler_id = response.json()["handler_id"]

    # First fetch populates cache
    first = await wait_for_passing(lambda: validate_result_response(handler_id, client))
    assert first["result"] == "processed: cache-me"

    second = await validate_result_response(handler_id, client)
    assert second == first


@pytest.mark.asyncio
async def test_post_event_handler_not_found(client: AsyncClient) -> None:
    response = await client.post("/events/nonexistent_handler", json={"event": "{}"})
    assert response.status_code == 404
    assert "Handler not found" in response.text


@pytest.mark.asyncio
async def test_post_event_to_completed_workflow(client: AsyncClient) -> None:
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
async def test_post_event_invalid_event_data(client: AsyncClient) -> None:
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
    client: AsyncClient, server: WorkflowServer
) -> None:
    # Dumb test for code coverage. Inject a dummy handler with no context to trigger 500 path
    wrapper = SimpleNamespace(run_handler=SimpleNamespace(done=lambda: False, ctx=None))

    handler_id = "noctx-1"
    server._handlers[handler_id] = wrapper  # type: ignore[assignment]

    try:
        response = await client.post(f"/events/{handler_id}", json={"event": "{}"})
        assert response.status_code == 500
        assert "Context not available" in response.text
    finally:
        server._handlers.pop(handler_id, None)


@pytest.mark.asyncio
async def test_post_event_body_parsing_error(client: AsyncClient) -> None:
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
async def test_post_event_missing_event_data(client: AsyncClient) -> None:
    # Start an interactive workflow
    response = await client.post("/workflows/interactive/run-nowait", json={})
    handler_id = response.json()["handler_id"]

    # Send request without event data
    response = await client.post(f"/events/{handler_id}", json={})
    assert response.status_code == 400
    assert "Event data is required" in response.text


@pytest.mark.asyncio
async def test_handler_datetime_fields_progress(client: AsyncClient) -> None:
    # Start interactive workflow which waits for an external event
    response = await client.post("/workflows/interactive/run-nowait", json={})
    assert response.status_code == 200
    handler_id = response.json()["handler_id"]

    # Snapshot initial times
    resp = await client.get("/handlers")
    assert resp.status_code == 200
    handlers = resp.json()["handlers"]
    item = next(h for h in handlers if h["handler_id"] == handler_id)
    started_at_1 = datetime.fromisoformat(item["started_at"])  # ISO 8601
    updated_at_1 = datetime.fromisoformat(item["updated_at"])  # ISO 8601
    assert started_at_1 <= updated_at_1
    assert item["completed_at"] is None

    # Send an external event to progress the workflow and update timestamps
    await asyncio.sleep(0.01)
    serializer = JsonSerializer()
    event = ExternalEvent(response="ts-check")
    event_str = serializer.serialize(event)
    send = await client.post(f"/events/{handler_id}", json={"event": event_str})
    assert send.status_code == 200

    # Check updated_at increased
    resp2 = await client.get("/handlers")
    assert resp2.status_code == 200
    item2 = next(h for h in resp2.json()["handlers"] if h["handler_id"] == handler_id)
    updated_at_2 = datetime.fromisoformat(item2["updated_at"])  # ISO 8601
    assert updated_at_2 >= updated_at_1

    # Wait for completion and check completed_at
    async def _wait_done() -> Response:
        r = await client.get(f"/results/{handler_id}")
        if r.status_code == 200:
            return r
        raise AssertionError("not done")

    await wait_for_passing(_wait_done)

    resp3 = await client.get("/handlers")
    item3 = next(h for h in resp3.json()["handlers"] if h["handler_id"] == handler_id)
    assert item3["status"] in {"completed", "failed"}
    if item3["status"] == "completed":
        assert item3["completed_at"] is not None
        completed_at = datetime.fromisoformat(item3["completed_at"])  # ISO 8601
        assert completed_at >= updated_at_2
