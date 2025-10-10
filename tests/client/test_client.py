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


@pytest.mark.asyncio
async def test_client(server: WorkflowServer) -> None:
    transport = ASGITransport(server.app)
    client = WorkflowClient(httpx_kwargs={"transport": transport})
    is_healthy = await client.is_healthy()
    assert is_healthy
    wfs = await client.list_workflows()
    assert len(wfs["workflows"]) == 1
    assert wfs["workflows"][0] == "greeting"
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
    result = await client.get_result(handler_id)
    assert result is not None
    res = OutputEvent.model_validate(result)
    assert "John" in res.greeting and "!" in res.greeting and "hello" in res.greeting
    result = await client.get_result(handler_id)
    assert result["handler_id"] == handler_id
    handlers = await client.get_handlers()
    assert len(handlers["handlers"]) == 1
    assert handlers["handlers"][0]["handler_id"] == handler_id
    result = await client.run_workflow(
        "greeting", start_event=InputEvent(greeting="hello", name="John")
    )
    assert result is not None
    res = OutputEvent.model_validate(result)
    assert "John" in res.greeting and "!" in res.greeting and "hello" in res.greeting
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
