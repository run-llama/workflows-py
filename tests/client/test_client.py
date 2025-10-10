import pytest

from httpx import ASGITransport
from workflows.server.server import WorkflowServer
from workflows.client import WorkflowClient
from .greeting_workflow import greeting_wf, InputEvent, OutputEvent


@pytest.fixture()
def server() -> WorkflowServer:
    ws = WorkflowServer()
    ws.add_workflow(name="greeting", workflow=greeting_wf)
    return ws


@pytest.fixture()
def client(server: WorkflowServer) -> WorkflowClient:
    transport = ASGITransport(server.app)
    return WorkflowClient(httpx_kwargs={"transport": transport})


@pytest.mark.asyncio
async def test_is_healthy(client: WorkflowClient) -> None:
    is_healthy = await client.is_healthy()
    assert is_healthy == {"status": "healthy"}


@pytest.mark.asyncio
async def test_list_workflows(client: WorkflowClient) -> None:
    wfs = await client.list_workflows()
    assert len(wfs["workflows"]) == 1
    assert wfs["workflows"][0] == "greeting"


@pytest.mark.asyncio
async def test_run_nowait_and_stream_events(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    assert isinstance(handler, dict)
    handler_id = handler["handler_id"]

    events = []
    async for event in client.get_workflow_events(handler_id=handler_id):
        assert isinstance(event, dict)
        events.append(event)
    assert len(events) == 3


@pytest.mark.asyncio
async def test_get_result_for_handler(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler_id = handler["handler_id"]
    # wait for completion
    async for event in client.get_workflow_events(handler_id=handler_id):
        pass

    result = await client.get_result(handler_id)
    assert result["result"] is not None
    res = OutputEvent.model_validate(result["result"])
    assert "John" in res.greeting and "!" in res.greeting and "hello" in res.greeting

    # Result should be retrievable again and reference the same handler
    result_again = await client.get_result(handler_id)
    assert result_again["handler_id"] == handler_id


@pytest.mark.asyncio
async def test_get_handlers(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler_id = handler["handler_id"]

    handlers = await client.get_handlers()
    assert len(handlers["handlers"]) == 1
    assert handlers["handlers"][0]["handler_id"] == handler_id


@pytest.mark.asyncio
async def test_run_workflow_sync_result(client: WorkflowClient) -> None:
    result = await client.run_workflow(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    assert result is not None
    res = OutputEvent.model_validate(result["result"])
    assert "John" in res.greeting and "!" in res.greeting and "hello" in res.greeting


@pytest.mark.asyncio
async def test_stream_events_including_internal(client: WorkflowClient) -> None:
    handler = await client.run_workflow_nowait(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    handler_id = handler["handler_id"]

    events = []
    async for event in client.get_workflow_events(
        handler_id=handler_id, include_internal_events=True
    ):
        assert isinstance(event, dict)
        events.append(event)
    assert len(events) > 3
