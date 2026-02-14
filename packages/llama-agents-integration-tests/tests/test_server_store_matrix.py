# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""End-to-end HTTP tests matrix-tested across workflow stores (memory, sqlite)."""

from __future__ import annotations

import asyncio
import socket
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Awaitable, Callable, TypeVar

import httpx
import pytest
import uvicorn
from llama_agents.client.client import WorkflowClient
from llama_agents.server import (
    MemoryWorkflowStore,
    SqliteWorkflowStore,
    WorkflowServer,
)
from llama_agents.server._store.postgres_workflow_store import PostgresWorkflowStore
from llama_agents_integration_tests.fake_agent_data import (
    FakeAgentDataBackend,
    create_agent_data_store,
)
from llama_agents_integration_tests.postgres import get_asyncpg_dsn
from workflows import Context, Workflow, step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)

T = TypeVar("T")


# -- Utilities --


async def _get_handler_raw(base_url: str, handler_id: str) -> dict[str, Any]:
    """Get handler data without raising on non-200 status (server returns 500 for failed/cancelled)."""
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        resp = await client.get(f"/handlers/{handler_id}")
        return resp.json()  # type: ignore[no-any-return]


async def wait_for_passing(
    func: Callable[[], Awaitable[T]],
    max_duration: float = 5.0,
    interval: float = 0.05,
) -> T:
    start_time = time.monotonic()
    last_exception: Exception | None = None
    while time.monotonic() - start_time < max_duration:
        remaining_duration = max_duration - (time.monotonic() - start_time)
        try:
            return await asyncio.wait_for(func(), timeout=remaining_duration)
        except Exception as e:
            last_exception = e
            await asyncio.sleep(interval)
    if last_exception:
        raise last_exception
    raise TimeoutError(f"Timed out after {max_duration}s")


@asynccontextmanager
async def live_server(
    server_factory: Callable[[], WorkflowServer],
) -> AsyncGenerator[tuple[str, WorkflowServer], None]:
    """Start a live HTTP server for testing with atomic port acquisition."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        port = sock.getsockname()[1]

        server = server_factory()

        config = uvicorn.Config(
            server.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            loop="asyncio",
        )
        uv_server = uvicorn.Server(config)

        task = asyncio.create_task(uv_server.serve(sockets=[sock]))

        base_url = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(base_url=base_url, timeout=1.0) as client:
            for _ in range(50):
                try:
                    resp = await client.get("/health")
                    if resp.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.01)
            else:
                uv_server.should_exit = True
                await task
                raise RuntimeError("Live server did not start in time")

        try:
            yield base_url, server
        finally:
            uv_server.should_exit = True
            try:
                await task
            finally:
                await server.stop()
    finally:
        try:
            sock.close()
        except Exception:
            pass


# -- Workflow definitions --


class SimpleTestWorkflow(Workflow):
    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        message = await ctx.store.get("test_param", None)
        if message is None:
            message = getattr(ev, "message", "default")
        return StopEvent(result=f"processed: {message}")


class ErrorWorkflow(Workflow):
    @step
    async def error_step(self, ev: StartEvent) -> StopEvent:
        raise ValueError("Test error")


class StreamEvent(Event):
    message: str
    sequence: int


class StreamingWorkflow(Workflow):
    @step
    async def stream_data(self, ctx: Context, ev: StartEvent) -> StopEvent:
        count = getattr(ev, "count", 3)
        for i in range(count):
            ctx.write_event_to_stream(StreamEvent(message=f"event_{i}", sequence=i))
            await asyncio.sleep(0.01)
        return StopEvent(result=f"completed_{count}_events")


class RequestedExternalEvent(InputRequiredEvent):
    message: str


class ExternalEvent(HumanResponseEvent):
    response: str


class InteractiveWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> RequestedExternalEvent:
        return RequestedExternalEvent(message="ping")

    @step
    async def end(self, ctx: Context, ev: ExternalEvent) -> StopEvent:
        if ev.response == "error":
            raise RuntimeError("Error response received")
        return StopEvent(result=f"received: {ev.response}")


# -- Fixtures --


@pytest.fixture(
    params=[
        "memory",
        "sqlite",
        "agent_data",
        pytest.param("postgres", marks=pytest.mark.docker),
    ]
)
async def server_with_store(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[tuple[str, WorkflowServer, str], None]:
    store_type: str = request.param

    pg_store: PostgresWorkflowStore | None = None

    if store_type == "postgres":
        postgres_container = request.getfixturevalue("postgres_container")
        dsn = get_asyncpg_dsn(postgres_container)
        pg_store = PostgresWorkflowStore(dsn=dsn)
        await pg_store.start()
        await pg_store.run_migrations()

    def make_server() -> WorkflowServer:
        if store_type == "memory":
            store = MemoryWorkflowStore()
        elif store_type == "sqlite":
            db_path = tmp_path_factory.mktemp("sqlite") / "test.db"
            store = SqliteWorkflowStore(str(db_path))
        elif store_type == "postgres":
            assert pg_store is not None
            store = pg_store
        else:
            store = create_agent_data_store(FakeAgentDataBackend(), monkeypatch)
        server = WorkflowServer(workflow_store=store)
        server.add_workflow("test", SimpleTestWorkflow())
        server.add_workflow("streaming", StreamingWorkflow())
        server.add_workflow(
            "interactive", InteractiveWorkflow(), additional_events=[ExternalEvent]
        )
        server.add_workflow("error", ErrorWorkflow())
        return server

    async with live_server(make_server) as (base_url, server):
        try:
            yield base_url, server, store_type
        finally:
            if pg_store is not None:
                await pg_store.close()


# -- Tests --


@pytest.mark.asyncio
async def test_sync_run(server_with_store: tuple[str, WorkflowServer, str]) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)
    result = await client.run_workflow(
        "test", start_event={"__is_pydantic_event": True}
    )
    assert result.status == "completed"
    assert result.result is not None
    assert result.result.value.get("result") == "processed: default"


@pytest.mark.asyncio
async def test_async_run_and_poll(
    server_with_store: tuple[str, WorkflowServer, str],
) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)
    started = await client.run_workflow_nowait("test")
    handler_id = started.handler_id

    async def check_completed() -> None:
        data = await client.get_handler(handler_id)
        assert data.status == "completed"
        assert data.result is not None
        assert data.result.value.get("result") == "processed: default"

    await wait_for_passing(check_completed)


@pytest.mark.asyncio
async def test_sse_event_streaming(
    server_with_store: tuple[str, WorkflowServer, str],
) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)
    started = await client.run_workflow_nowait("streaming")
    handler_id = started.handler_id

    events_seen: list[StreamEvent] = []
    async for ev in client.get_workflow_events(handler_id):
        event = ev.load_event([StreamEvent])
        if isinstance(event, StreamEvent):
            events_seen.append(event)

    assert len(events_seen) == 3
    assert [e.sequence for e in events_seen] == [0, 1, 2]


@pytest.mark.asyncio
async def test_send_external_event(
    server_with_store: tuple[str, WorkflowServer, str],
) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)
    started = await client.run_workflow_nowait("interactive")
    handler_id = started.handler_id

    async for ev in client.get_workflow_events(handler_id):
        event = ev.load_event([RequestedExternalEvent])
        if isinstance(event, RequestedExternalEvent):
            sent = await client.send_event(handler_id, ExternalEvent(response="pong"))
            assert sent.status == "sent"
            break

    async def check_completed() -> None:
        data = await client.get_handler(handler_id)
        assert data.status == "completed"
        assert data.result is not None
        assert data.result.value.get("result") == "received: pong"

    await wait_for_passing(check_completed)


@pytest.mark.asyncio
async def test_cancel_handler(
    server_with_store: tuple[str, WorkflowServer, str],
) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)
    started = await client.run_workflow_nowait("interactive")
    handler_id = started.handler_id

    # Wait for the workflow to reach the waiting state by polling
    async def is_running() -> None:
        data = await client.get_handler(handler_id)
        assert data.status == "running"

    await wait_for_passing(is_running)

    result = await client.cancel_handler(handler_id)
    assert result.status == "cancelled"

    async def check_cancelled() -> None:
        data = await _get_handler_raw(base_url, handler_id)
        assert data["status"] == "cancelled"

    await wait_for_passing(check_cancelled)


@pytest.mark.asyncio
async def test_list_handlers(
    server_with_store: tuple[str, WorkflowServer, str],
) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)

    # Run two workflows
    await client.run_workflow("test", start_event={"__is_pydantic_event": True})
    await client.run_workflow("test", start_event={"__is_pydantic_event": True})

    handlers = await client.get_handlers()
    assert len(handlers.handlers) >= 2


@pytest.mark.asyncio
async def test_error_workflow_status(
    server_with_store: tuple[str, WorkflowServer, str],
) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)
    started = await client.run_workflow_nowait("error")
    handler_id = started.handler_id

    async def check_failed() -> None:
        data = await _get_handler_raw(base_url, handler_id)
        assert data["status"] == "failed"
        assert data["error"] is not None
        assert "Test error" in data["error"]

    await wait_for_passing(check_failed)


@pytest.mark.asyncio
async def test_streaming_workflow_events(
    server_with_store: tuple[str, WorkflowServer, str],
) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)
    started = await client.run_workflow_nowait("streaming")
    handler_id = started.handler_id

    stream_events: list[StreamEvent] = []
    saw_stop = False
    async for ev in client.get_workflow_events(handler_id):
        event = ev.load_event([StreamEvent])
        if isinstance(event, StreamEvent):
            stream_events.append(event)
        elif isinstance(event, StopEvent):
            saw_stop = True

    assert len(stream_events) == 3
    assert saw_stop


@pytest.mark.asyncio
async def test_cursor_resume(
    server_with_store: tuple[str, WorkflowServer, str],
) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)

    # Run a streaming workflow synchronously first so all events are stored
    result = await client.run_workflow(
        "streaming", start_event={"__is_pydantic_event": True}
    )
    assert result.status == "completed"
    handler_id = result.handler_id

    # Now fetch events with after_sequence to get only later events
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as http_client:
        async with http_client.stream(
            "GET",
            f"/events/{handler_id}",
            params={"sse": "false", "after_sequence": "1"},
            headers={"Connection": "keep-alive"},
        ) as response:
            lines = []
            async for line in response.aiter_lines():
                if line.strip():
                    lines.append(line)

    # Should have events after sequence 1 (i.e. sequence 2+ and StopEvent)
    assert len(lines) >= 1


@pytest.mark.asyncio
async def test_concurrent_workflows(
    server_with_store: tuple[str, WorkflowServer, str],
) -> None:
    base_url, _server, _store_type = server_with_store
    client = WorkflowClient(base_url=base_url)

    # Start multiple workflows concurrently
    tasks = [client.run_workflow_nowait("test") for _ in range(5)]
    started = await asyncio.gather(*tasks)
    handler_ids = [s.handler_id for s in started]

    async def all_completed() -> None:
        for hid in handler_ids:
            data = await client.get_handler(hid)
            assert data.status == "completed"

    await wait_for_passing(all_completed)


# -- Durability tests (sqlite only) --


@pytest.mark.asyncio
async def test_server_restart_resumes_workflow(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    db_path = str(tmp_path_factory.mktemp("sqlite_restart") / "test.db")

    def make_server() -> WorkflowServer:
        store = SqliteWorkflowStore(db_path)
        server = WorkflowServer(workflow_store=store)
        server.add_workflow(
            "interactive", InteractiveWorkflow(), additional_events=[ExternalEvent]
        )
        return server

    # Start workflow, then stop server
    async with live_server(make_server) as (base_url, _server):
        client = WorkflowClient(base_url=base_url)
        started = await client.run_workflow_nowait("interactive")
        handler_id = started.handler_id

        # Wait for the workflow to reach the waiting state
        async for ev in client.get_workflow_events(handler_id):
            event = ev.load_event([RequestedExternalEvent])
            if isinstance(event, RequestedExternalEvent):
                break

    # Restart with same store - workflow should resume
    async with live_server(make_server) as (base_url2, _server2):
        client2 = WorkflowClient(base_url=base_url2)

        # Send the event to complete the workflow
        sent = await client2.send_event(
            handler_id, ExternalEvent(response="after_restart")
        )
        assert sent.status == "sent"

        async def check_completed() -> None:
            data = await client2.get_handler(handler_id)
            assert data.status == "completed"
            assert data.result is not None
            assert data.result.value.get("result") == "received: after_restart"

        await wait_for_passing(check_completed)


@pytest.mark.asyncio
async def test_idle_release_and_reload(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    db_path = str(tmp_path_factory.mktemp("sqlite_idle") / "test.db")

    def make_server() -> WorkflowServer:
        store = SqliteWorkflowStore(db_path)
        server = WorkflowServer(workflow_store=store)
        server.add_workflow(
            "interactive", InteractiveWorkflow(), additional_events=[ExternalEvent]
        )
        return server

    async with live_server(make_server) as (base_url, _server):
        client = WorkflowClient(base_url=base_url)
        started = await client.run_workflow_nowait("interactive")
        handler_id = started.handler_id

        # Wait for handler to be running (idle internally)
        async def is_running() -> None:
            data = await client.get_handler(handler_id)
            assert data.status == "running"

        await wait_for_passing(is_running, max_duration=3.0)

        # Give time for idle release to happen
        await asyncio.sleep(0.5)

        # Send event via HTTP to trigger reload from idle state
        sent = await client.send_event(handler_id, ExternalEvent(response="after_idle"))
        assert sent.status == "sent"

        async def check_completed() -> None:
            data = await client.get_handler(handler_id)
            assert data.status == "completed"
            assert data.result is not None
            assert data.result.value.get("result") == "received: after_idle"

        await wait_for_passing(check_completed)
