import pytest
import httpx

from httpx import ASGITransport, AsyncClient
from workflows.protocol.serializable_events import EventEnvelopeWithMetadata
from workflows.server.server import WorkflowServer
from workflows.client import WorkflowClient
from .client_workflows import (
    greeting_wf,
    crashing_wf,
    InputEvent,
    OutputEvent,
    GreetEvent,
)
from workflows.server.memory_workflow_store import MemoryWorkflowStore


@pytest.fixture()
def server() -> WorkflowServer:
    # Use MemoryWorkflowStore so get_handlers() can retrieve from persistence
    ws = WorkflowServer(workflow_store=MemoryWorkflowStore())
    ws.add_workflow(name="greeting", workflow=greeting_wf)
    ws.add_workflow(name="crashing", workflow=crashing_wf)
    return ws


@pytest.fixture()
def client(server: WorkflowServer) -> WorkflowClient:
    transport = ASGITransport(server.app)
    httpx_client = AsyncClient(transport=transport, base_url="http://test")
    return WorkflowClient(httpx_client=httpx_client)


@pytest.mark.asyncio
async def test_is_healthy(client: WorkflowClient) -> None:
    is_healthy = await client.is_healthy()
    assert is_healthy.status == "healthy"


@pytest.mark.asyncio
async def test_list_workflows(client: WorkflowClient) -> None:
    wfs = await client.list_workflows()
    assert len(wfs.workflows) == 2
    assert "greeting" in wfs.workflows
    assert "crashing" in wfs.workflows


@pytest.mark.asyncio
async def test_run_nowait_and_stream_events(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    assert handler.handler_id
    handler_id = handler.handler_id

    events = []
    async for event in client.get_workflow_events(handler_id=handler_id):
        assert isinstance(event, EventEnvelopeWithMetadata)
        events.append(event.load_event())
    assert len(events) == 3
    assert events[0] == InputEvent(greeting="hello", name="John")


@pytest.mark.asyncio
async def test_get_result_for_handler(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler_id = handler.handler_id
    # wait for completion
    async for event in client.get_workflow_events(handler_id=handler_id):
        pass

    result = await client.get_result(handler_id)
    assert result.result is not None
    res = OutputEvent.model_validate(result.result.value)
    assert "John" in res.greeting and "!" in res.greeting and "hello" in res.greeting

    # Result should be retrievable again and reference the same handler
    result_again = await client.get_result(handler_id)
    assert result_again == result


@pytest.mark.asyncio
async def test_get_handler(client: WorkflowClient) -> None:
    handler = await client.run_workflow(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    assert handler.status == "completed"
    handler_id = handler.handler_id

    handler_data = await client.get_handler(handler_id)
    assert handler_data.handler_id == handler_id
    assert handler_data.workflow_name == "greeting"
    assert handler_data.run_id == handler.run_id
    assert handler_data.status == "completed"
    assert handler_data.started_at is not None
    assert handler_data.updated_at is not None
    assert handler_data.completed_at is not None


@pytest.mark.asyncio
async def test_get_handlers(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler_id = handler.handler_id

    handlers = await client.get_handlers()
    assert len(handlers.handlers) == 1
    assert handlers.handlers[0].handler_id == handler_id


@pytest.mark.asyncio
async def test_get_handlers_filter_by_workflow_name(client: WorkflowClient) -> None:
    await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    await client.run_workflow_nowait("crashing", start_event={})

    handlers = await client.get_handlers(workflow_name=["greeting"])
    assert len(handlers.handlers) >= 1
    assert all(h.workflow_name == "greeting" for h in handlers.handlers)


@pytest.mark.asyncio
async def test_get_handlers_filter_by_status(client: WorkflowClient) -> None:
    await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    completed_handler = await client.run_workflow(
        "greeting", start_event=InputEvent(greeting="hi", name="Jane")
    )
    # Wait for the crashing workflow to fail
    try:
        await client.run_workflow(
            "crashing", start_event=InputEvent(greeting="test", name="test")
        )
    except Exception:
        pass

    handlers = await client.get_handlers(status=["completed"])
    handler_ids = {h.handler_id for h in handlers.handlers}
    assert completed_handler.handler_id in handler_ids

    failed_handlers = await client.get_handlers(status=["failed"])
    failed_ids = {h.handler_id for h in failed_handlers.handlers}
    assert len(failed_ids) == 1


@pytest.mark.asyncio
async def test_get_handlers_for_complex_workflows(
    client: WorkflowClient, server: WorkflowServer
) -> None:
    handler1 = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler1_id = handler1.handler_id

    handlers = await client.get_handlers()
    assert len(handlers.handlers) == 1
    assert handlers.handlers[0].handler_id == handler1_id

    handler2 = await client.run_workflow(
        "greeting", start_event=InputEvent(greeting="hello", name="Jane")
    )
    handler2_id = handler2.handler_id

    # Restart the server
    await server.stop()
    await server.start()

    handlers = await client.get_handlers()
    assert len(handlers.handlers) == 2
    assert handlers.handlers[0].handler_id == handler1_id
    assert handlers.handlers[1].handler_id == handler2_id


@pytest.mark.asyncio
async def test_run_workflow_sync_result(client: WorkflowClient) -> None:
    result = await client.run_workflow(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    assert result.result is not None
    res = OutputEvent.model_validate(result.result.value)
    assert "John" in res.greeting and "!" in res.greeting and "hello" in res.greeting


@pytest.mark.asyncio
async def test_stream_events_including_internal(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler_id = handler.handler_id

    events = []
    async for event in client.get_workflow_events(
        handler_id=handler_id, include_internal_events=True
    ):
        assert isinstance(event, EventEnvelopeWithMetadata)
        events.append(event.load_event())
    assert len(events) > 3


@pytest.mark.asyncio
async def test_cancel_handler(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler_id = handler.handler_id

    cancel_resp = await client.cancel_handler(handler_id=handler_id)
    assert cancel_resp.status == "cancelled"
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler_id = handler.handler_id

    cancel_resp = await client.cancel_handler(handler_id=handler_id, purge=True)
    assert cancel_resp.status == "deleted"


@pytest.mark.asyncio
async def test_send_event(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler_id = handler.handler_id

    # Send an event to the running workflow
    response = await client.send_event(
        handler_id=handler_id,
        event=GreetEvent(greeting="Bonjour John", exclamation_marks=5),
    )
    assert response.status == "sent"

    # Wait for completion
    async for event in client.get_workflow_events(handler_id=handler_id):
        pass

    # Verify workflow completed successfully
    result = await client.get_result(handler_id)
    assert result.result is not None


@pytest.mark.asyncio
async def test_error_message_format(client: WorkflowClient) -> None:
    """Test that error messages include method, URL, status code, and response body preview."""
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.run_workflow(
            "nonexistent_workflow",
            start_event=InputEvent(greeting="hello", name="John"),
        )

    error_message = str(exc_info.value)

    # Verify error message contains the expected components
    assert (
        "404 Not Found for POST http://test/workflows/nonexistent_workflow/run. Response: Workflow not found"
        == error_message
    )  # Status code
