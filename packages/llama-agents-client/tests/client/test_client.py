# ty: ignore[invalid-argument-type, not-iterable]
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from client_test_workflows import (
    CrashingWorkflow,
    GreetEvent,
    GreetingWorkflow,
    InputEvent,
    OutputEvent,
)
from httpx import ASGITransport, AsyncClient
from llama_agents.client import WorkflowClient
from llama_agents.client.protocol.serializable_events import (
    EventEnvelopeWithMetadata,
)
from llama_agents.server import MemoryWorkflowStore
from llama_agents.server.server import WorkflowServer
from workflows import Context


@pytest.fixture()
def server() -> WorkflowServer:
    # Use MemoryWorkflowStore so get_handlers() can retrieve from persistence
    ws = WorkflowServer(workflow_store=MemoryWorkflowStore())
    ws.add_workflow(name="greeting", workflow=GreetingWorkflow())
    ws.add_workflow(name="crashing", workflow=CrashingWorkflow())
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
        '404 Not Found for POST http://test/workflows/nonexistent_workflow/run. Response: {"detail":"Workflow not found"}'
        == error_message
    )


def _envelope(msg: str) -> EventEnvelopeWithMetadata:
    return EventEnvelopeWithMetadata(
        value={"msg": msg}, qualified_name=None, type="TestEvent", types=None
    )


# Each "connection" in a script is a list of SSE events to yield, optionally
# ending with an exception to simulate a disconnect. A bare exception means
# the connection fails before yielding any data.
ConnectionScript = list[tuple[int, EventEnvelopeWithMetadata] | Exception] | Exception


class FakeStreamClient:
    """Mock httpx client that replays a scripted sequence of SSE connections."""

    def __init__(self, script: list[ConnectionScript]) -> None:
        self._script = list(script)
        self.captured_params: list[dict[str, str]] = []
        self._call = 0

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[AsyncMock]:
        self.captured_params.append(params or {})
        assert self._call < len(self._script), "More connections than scripted"
        entry = self._script[self._call]
        self._call += 1

        if isinstance(entry, Exception):
            raise entry

        events = entry
        tail_error: Exception | None = None
        # If the last element is an exception, pop it as a mid-stream error
        if events and isinstance(events[-1], Exception):
            tail_error = events[-1]  # type: ignore[assignment]
            events = events[:-1]  # type: ignore[assignment]

        resp = AsyncMock()
        resp.status_code = 200

        async def aiter_lines() -> AsyncIterator[str]:
            for seq, env in events:  # type: ignore[union-attr]
                yield f"id: {seq}"
                yield f"data: {env.model_dump_json()}"
                yield ""
            if tail_error is not None:
                raise tail_error

        resp.aiter_lines = aiter_lines
        yield resp


async def _collect(
    script: list[ConnectionScript], **kwargs: object
) -> list[EventEnvelopeWithMetadata]:
    fake = FakeStreamClient(script)
    wf_client = WorkflowClient(httpx_client=fake)  # type: ignore[arg-type]
    events = [
        e
        async for e in wf_client.get_workflow_events(handler_id="h", **kwargs)  # type: ignore[arg-type]
    ]
    return events


@pytest.mark.asyncio
async def test_reconnect_resumes_from_last_sequence() -> None:
    e1, e2, e3 = _envelope("first"), _envelope("second"), _envelope("third")
    fake = FakeStreamClient(
        [
            [(0, e1), httpx.RemoteProtocolError("reset")],
            [(1, e2), (2, e3)],
        ]
    )
    wf_client = WorkflowClient(httpx_client=fake)  # type: ignore[arg-type]
    events = [e async for e in wf_client.get_workflow_events(handler_id="h")]

    assert [e.value["msg"] for e in events] == ["first", "second", "third"]
    assert fake.captured_params[0]["after_sequence"] == "-1"
    assert fake.captured_params[1]["after_sequence"] == "0"


@pytest.mark.asyncio
async def test_reconnect_exceeds_max_attempts_raises() -> None:
    with pytest.raises(ConnectionError, match="after 2 attempts"):
        await _collect(
            [httpx.ConnectError("refused")] * 3,
            max_reconnect_attempts=2,
        )


@pytest.mark.asyncio
async def test_reconnect_resets_attempts_on_success() -> None:
    e1, e2 = _envelope("a"), _envelope("b")
    events = await _collect(
        [
            [(0, e1), httpx.ReadError("broken")],
            httpx.ReadError("broken again"),
            [(1, e2)],
        ],
        max_reconnect_attempts=2,
    )
    assert [e.value["msg"] for e in events] == ["a", "b"]


@pytest.mark.asyncio
async def test_timeout_exception_not_retried() -> None:
    with pytest.raises(TimeoutError, match="Timeout"):
        await _collect([httpx.ReadTimeout("timed out")])


@pytest.mark.asyncio
async def test_get_workflow_events_tracks_last_sequence() -> None:
    e1, e2, e3 = _envelope("a"), _envelope("b"), _envelope("c")
    fake = FakeStreamClient([[(0, e1), (1, e2), (2, e3)]])
    wf_client = WorkflowClient(httpx_client=fake)  # type: ignore[arg-type]

    stream = wf_client.get_workflow_events(handler_id="h")
    assert stream.last_sequence == -1

    sequences: list[int | str] = []
    async for event in stream:
        sequences.append(stream.last_sequence)

    assert sequences == [0, 1, 2]
    assert stream.last_sequence == 2


@pytest.mark.asyncio
async def test_get_workflow_events_with_now() -> None:
    e1 = _envelope("a")
    fake = FakeStreamClient([[(5, e1)]])
    wf_client = WorkflowClient(httpx_client=fake)  # type: ignore[arg-type]

    stream = wf_client.get_workflow_events(handler_id="h", after_sequence="now")
    assert stream.last_sequence == "now"

    async for _ in stream:
        pass

    assert stream.last_sequence == 5
    assert fake.captured_params[0]["after_sequence"] == "now"


def test_client_constructor_validation() -> None:
    # Test neither httpx_client nor base_url provided
    with pytest.raises(
        ValueError, match="Either httpx_client or base_url must be provided"
    ):
        WorkflowClient()  # type: ignore[call-overload]
    # Test both provided
    with pytest.raises(
        ValueError, match="Only one of httpx_client or base_url must be provided"
    ):
        WorkflowClient(httpx_client=httpx.AsyncClient(), base_url="http://test")  # type: ignore[call-overload]


@pytest.mark.asyncio
async def test_client_5xx_error_handling() -> None:
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content="Internal Server Error")

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        wf_client = WorkflowClient(httpx_client=client)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await wf_client.is_healthy()
        assert "500 Internal Server Error" in str(exc_info.value)
        assert "Response: Internal Server Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_dict_event_serialization(client: WorkflowClient) -> None:
    res = await client.run_workflow(
        "greeting",
        start_event={"greeting": "hello", "name": "John"},
    )
    assert res.status == "completed"

    res_nowait = await client.run_workflow_nowait(
        "greeting",
        start_event={
            "type": "InputEvent",
            "value": {"greeting": "hello", "name": "John"},
        },
    )
    assert res_nowait.handler_id is not None

    handler = await client.run_workflow_nowait(
        "greeting",
        start_event={
            "type": "InputEvent",
            "value": {"greeting": "hello", "name": "John"},
        },
    )
    send_res = await client.send_event(
        handler.handler_id,
        event={
            "type": "GreetEvent",
            "value": {"greeting": "Bonjour John", "exclamation_marks": 5},
        },
    )
    assert send_res.status == "sent"

    with pytest.raises(ValueError, match="Impossible to serialize the start event"):
        await client.run_workflow("greeting", start_event=object())  # type: ignore[argument-type]

    with pytest.raises(ValueError, match="Impossible to serialize the start event"):
        await client.run_workflow_nowait("greeting", start_event=object())  # type: ignore[argument-type]

    with pytest.raises(ValueError, match="Error while serializing the provided event"):
        await client.send_event(handler.handler_id, event=object())  # type: ignore[argument-type]

    bad_ctx = MagicMock(spec=Context)
    bad_ctx.to_dict.side_effect = RuntimeError("Serialize failed")

    with pytest.raises(ValueError, match="Impossible to serialize the context"):
        await client.run_workflow("greeting", context=bad_ctx)

    with pytest.raises(ValueError, match="Impossible to serialize the context"):
        await client.run_workflow_nowait("greeting", context=bad_ctx)


@pytest.mark.asyncio
async def test_event_stream_lifecycle_errors() -> None:
    import asyncio

    from llama_agents.client.client import (
        EventStream,
        _QueuedDone,
        _QueuedError,
        _QueuedEvent,
    )

    queue: asyncio.Queue = asyncio.Queue()
    stream = EventStream(queue, None, -1)

    e1 = _envelope("ok")
    await queue.put(_QueuedEvent(sequence=1, event=e1))
    await queue.put(_QueuedDone())

    events = []
    async for event in stream:
        events.append(event)
    assert len(events) == 1
    assert events[0] == e1

    with pytest.raises(RuntimeError, match="EventStream can only be iterated once"):
        async for _ in stream:
            pass

    stream2 = EventStream(queue, None, -1)
    await queue.put(_QueuedError(ValueError("Stream error")))
    with pytest.raises(ValueError, match="Stream error"):
        async for _ in stream2:
            pass

    # Coverage for self._task is None in aclose
    await stream2.aclose()


@pytest.mark.asyncio
async def test_get_workflow_events_404_response() -> None:
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        wf_client = WorkflowClient(httpx_client=client)
        stream = wf_client.get_workflow_events(handler_id="h")
        with pytest.raises(ValueError, match="Handler not found"):
            async for _ in stream:
                pass


@pytest.mark.asyncio
async def test_get_workflow_events_204_response() -> None:
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        wf_client = WorkflowClient(httpx_client=client)
        stream = wf_client.get_workflow_events(handler_id="h")
        events = []
        async for event in stream:
            events.append(event)
        assert len(events) == 0
