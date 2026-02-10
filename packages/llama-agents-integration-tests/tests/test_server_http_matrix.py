# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Parameterized live HTTP server integration tests across storage backends."""

from __future__ import annotations

import asyncio
import socket
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
import pytest
import uvicorn
from dbos import DBOS, DBOSConfig
from llama_agents.client.client import WorkflowClient
from llama_agents.dbos import DBOSRuntime
from llama_agents.server import MemoryWorkflowStore, SqliteWorkflowStore, WorkflowServer
from testcontainers.postgres import PostgresContainer
from workflows import Context, Workflow, step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)


# Workflow definitions
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
        return StopEvent(result=f"received: {ev.response}")


class CumulativeWorkflow(Workflow):
    @step
    async def accumulate(self, ctx: Context, ev: StartEvent) -> StopEvent:
        current_count = await ctx.store.get("count", 0)
        increment = getattr(ev, "increment", 1)
        new_count = current_count + increment
        await ctx.store.set("count", new_count)
        run_history = await ctx.store.get("run_history", [])
        run_history.append(f"run_{len(run_history) + 1}")
        await ctx.store.set("run_history", run_history)
        return StopEvent(result=f"count: {new_count}, runs: {len(run_history)}")


class SimpleTestWorkflow(Workflow):
    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        message = await ctx.store.get("test_param", None)
        if message is None:
            message = getattr(ev, "message", "default")
        return StopEvent(result=f"processed: {message}")


class WaitableExternalEvent(Event):
    response: str


class WaitingWorkflow(Workflow):
    @step
    async def start_and_wait(self, ctx: Context, ev: StartEvent) -> StopEvent:
        external = await ctx.wait_for_event(WaitableExternalEvent)
        return StopEvent(result=f"received: {external.response}")


# Helper functions
async def wait_for_passing(
    func: Any, max_duration: float = 5.0, interval: float = 0.05
) -> Any:
    start_time = time.monotonic()
    last_exception = None
    while time.monotonic() - start_time < max_duration:
        remaining = max_duration - (time.monotonic() - start_time)
        try:
            return await asyncio.wait_for(func(), timeout=remaining)
        except Exception as e:
            last_exception = e
            await asyncio.sleep(interval)
    raise last_exception or TimeoutError(
        f"wait_for_passing timed out after {max_duration}s"
    )


class live_server:
    def __init__(self, server_factory: Any) -> None:
        self.server_factory = server_factory
        self.sock: socket.socket | None = None
        self.task: Any = None
        self.uv_server: Any = None

    async def __aenter__(self) -> tuple[str, WorkflowServer]:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(128)
        port = self.sock.getsockname()[1]
        self.server = self.server_factory()
        await self.server.start()
        config = uvicorn.Config(
            self.server.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            loop="asyncio",
        )
        self.uv_server = uvicorn.Server(config)
        self.task = asyncio.create_task(self.uv_server.serve(sockets=[self.sock]))
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
                self.uv_server.should_exit = True
                await self.task
                raise RuntimeError("Live server did not start in time")
        return base_url, self.server

    async def __aexit__(self, *args: Any) -> None:
        if self.uv_server:
            self.uv_server.should_exit = True
        if self.task:
            try:
                await self.task
            except Exception:
                pass
        if hasattr(self.server, "stop"):
            await self.server.stop()
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


def _get_backend_params() -> list[Any]:
    return [
        pytest.param("memory", id="memory"),
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", marks=pytest.mark.docker, id="postgres"),
    ]


@pytest.fixture(scope="module")
def postgres_container() -> Any:
    with PostgresContainer("postgres:17", driver=None) as container:
        yield container


@pytest.fixture
async def backend_server(
    request: Any, tmp_path: Path, postgres_container: Any
) -> AsyncGenerator[tuple[str, WorkflowServer], None]:
    backend = request.param

    if backend == "memory":

        def factory() -> WorkflowServer:
            store = MemoryWorkflowStore()
            server = WorkflowServer(workflow_store=store)
            server.add_workflow("SimpleTestWorkflow", SimpleTestWorkflow())
            server.add_workflow("StreamingWorkflow", StreamingWorkflow())
            server.add_workflow("InteractiveWorkflow", InteractiveWorkflow())
            server.add_workflow("CumulativeWorkflow", CumulativeWorkflow())
            server.add_workflow("WaitingWorkflow", WaitingWorkflow())
            return server

        async with live_server(factory) as (base_url, server):
            yield base_url, server

    elif backend == "sqlite":
        db_path = tmp_path / "test.db"

        def factory() -> WorkflowServer:
            store = SqliteWorkflowStore(db_path=str(db_path))
            server = WorkflowServer(workflow_store=store)
            server.add_workflow("SimpleTestWorkflow", SimpleTestWorkflow())
            server.add_workflow("StreamingWorkflow", StreamingWorkflow())
            server.add_workflow("InteractiveWorkflow", InteractiveWorkflow())
            server.add_workflow("CumulativeWorkflow", CumulativeWorkflow())
            server.add_workflow("WaitingWorkflow", WaitingWorkflow())
            return server

        async with live_server(factory) as (base_url, server):
            yield base_url, server

    elif backend == "postgres":
        connection_url = postgres_container.get_connection_url()
        dbos_config: DBOSConfig = {
            "name": "wf-server-http-pg",
            "system_database_url": connection_url,
            "run_admin_server": False,
            "notification_listener_polling_interval_sec": 0.01,
        }
        DBOS(config=dbos_config)
        dbos_runtime = DBOSRuntime(polling_interval_sec=0.01)
        dbos_runtime.launch()
        store = dbos_runtime.create_workflow_store()

        def factory() -> WorkflowServer:
            server = WorkflowServer(
                workflow_store=store,
                runtime=dbos_runtime.build_server_runtime(),
            )
            server.add_workflow("SimpleTestWorkflow", SimpleTestWorkflow())
            server.add_workflow("StreamingWorkflow", StreamingWorkflow())
            server.add_workflow("InteractiveWorkflow", InteractiveWorkflow())
            server.add_workflow("CumulativeWorkflow", CumulativeWorkflow())
            server.add_workflow("WaitingWorkflow", WaitingWorkflow())
            return server

        async with live_server(factory) as (base_url, server):
            try:
                yield base_url, server
            finally:
                dbos_runtime.destroy()

    else:
        raise ValueError(f"Unknown backend: {backend}")


# Tests
@pytest.mark.asyncio
@pytest.mark.parametrize("backend_server", _get_backend_params(), indirect=True)
async def test_basic_run_and_result(
    backend_server: tuple[str, WorkflowServer],
) -> None:
    base_url, server = backend_server
    client = WorkflowClient(base_url=base_url)

    start_event = StartEvent(message="test_message")  # type: ignore[call-arg]
    result = await client.run_workflow("SimpleTestWorkflow", start_event=start_event)
    assert result.result is not None
    assert result.result.value["result"] == "processed: test_message"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_server", _get_backend_params(), indirect=True)
async def test_streaming_and_interactive(
    backend_server: tuple[str, WorkflowServer],
) -> None:
    base_url, server = backend_server
    client = WorkflowClient(base_url=base_url)

    started = await client.run_workflow_nowait("InteractiveWorkflow")
    handler_id = started.handler_id

    saw_prompt = False
    async for ev in client.get_workflow_events(handler_id):
        event = ev.load_event()
        if isinstance(event, RequestedExternalEvent):
            saw_prompt = True
            sent = await client.send_event(handler_id, ExternalEvent(response="pong"))
            assert sent.status == "sent"
            break
    assert saw_prompt

    async def check_completed() -> str:
        handler = await client.get_handler(handler_id)
        assert handler.status == "completed"
        assert handler.result is not None
        return handler.result.value["result"]

    result = await wait_for_passing(check_completed, max_duration=5.0)
    assert result == "received: pong"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_server", _get_backend_params(), indirect=True)
async def test_reconnect_stream(
    backend_server: tuple[str, WorkflowServer],
) -> None:
    base_url, server = backend_server
    client = WorkflowClient(base_url=base_url)

    started = await client.run_workflow_nowait("InteractiveWorkflow")
    handler_id = started.handler_id

    saw_prompt = False
    events_before = 0
    async for ev in client.get_workflow_events(handler_id):
        events_before += 1
        event = ev.load_event()
        if isinstance(event, RequestedExternalEvent):
            saw_prompt = True
            break
    assert saw_prompt

    stop_seen = asyncio.Event()

    async def consume_again() -> None:
        async for ev in client.get_workflow_events(handler_id):
            event = ev.load_event()
            if isinstance(event, StopEvent):
                stop_seen.set()
                break

    consume_task = asyncio.create_task(consume_again())

    await asyncio.sleep(0.05)

    sent = await client.send_event(handler_id, ExternalEvent(response="reconnect_test"))
    assert sent.status == "sent"

    await asyncio.wait_for(stop_seen.wait(), timeout=5.0)
    consume_task.cancel()

    async def check_completed() -> str:
        handler = await client.get_handler(handler_id)
        assert handler.status == "completed"
        assert handler.result is not None
        return handler.result.value["result"]

    result = await wait_for_passing(check_completed, max_duration=5.0)
    assert result == "received: reconnect_test"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_server", _get_backend_params(), indirect=True)
async def test_cumulative_rerun(
    backend_server: tuple[str, WorkflowServer],
) -> None:
    base_url, server = backend_server
    client = WorkflowClient(base_url=base_url)

    start_event1 = StartEvent(increment=5)  # type: ignore[call-arg]
    result1 = await client.run_workflow("CumulativeWorkflow", start_event=start_event1)
    assert result1.result is not None
    result_str1 = result1.result.value["result"]
    assert "count: 5" in result_str1
    assert "runs: 1" in result_str1

    handler_id = result1.handler_id

    start_event2 = StartEvent(increment=3)  # type: ignore[call-arg]
    result2 = await client.run_workflow(
        "CumulativeWorkflow", handler_id=handler_id, start_event=start_event2
    )
    assert result2.result is not None
    result_str2 = result2.result.value["result"]
    assert "count: 8" in result_str2
    assert "runs: 2" in result_str2
