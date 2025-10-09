import pytest

from httpx import ASGITransport
from workflows.server.server import WorkflowServer
from workflows.client import WorkflowClient
from .greeting_workflow import greeting_wf, InputEvent, OutputEvent
from .hitl_workflow import hitl_wf


@pytest.fixture()
def server() -> WorkflowServer:
    ws = WorkflowServer()
    ws.add_workflow(name="greeting", workflow=greeting_wf)
    ws.add_workflow(name="human", workflow=hitl_wf)
    return ws


@pytest.mark.asyncio
async def test_client(server: WorkflowServer) -> None:
    transport = ASGITransport(server.app)
    client = WorkflowClient(httpx_kwargs={"transport": transport})
    is_healthy = await client.is_healthy()
    assert is_healthy
    wfs = await client.list_workflows()
    assert isinstance(wfs, list)
    assert len(wfs) == 2
    assert wfs[0] == "greeting"
    assert wfs[1] == "human"
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
    result = await client.get_result(handler_id, as_handler=True)
    assert isinstance(result, dict)
    assert result["handler_id"] == handler_id
    handlers = await client.get_handlers()
    assert isinstance(handlers, list)
    assert len(handlers) == 1
    assert handlers[0] == result
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
    # handler = await client.run_workflow_nowait("human")
    # handler_id = handler["handler_id"]
    # async for event in client.get_workflow_events(handler_id=handler_id):
    #     if event.get("qualified_name", "") == "tests.client.hitl_workflow.RequestEvent":
    #         sent_event = await client.send_event(handler_id=handler_id, event=ResponseEvent(response="John"))
    #         assert sent_event
    #     if event.get("qualified_name", "") == "tests.client.hitl_workflow.HiddenEvent":
    #         sent_event = await client.send_event(handler_id=handler_id, event=ResponseEvent(response="Yes"))
    #         assert sent_event
    # result = await client.get_result(handler_id)
    # assert result is not None
    # res = OutEvent.model_validate(result)
    # assert res.output == "Got it!: Yes"
