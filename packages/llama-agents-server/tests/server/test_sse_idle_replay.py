# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Regression tests for the SSE cursor race on idle/HITL workflows.

An implicit "now" cursor on a handler that has already emitted a HITL event
and gone idle must rewind to the pause segment, otherwise the fresh subscriber
misses the prompt and hangs.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from llama_agents.client.client import WorkflowClient
from llama_agents.server import MemoryWorkflowStore, WorkflowServer
from llama_agents.server._store.abstract_workflow_store import HandlerQuery
from server_test_fixtures import (  # type: ignore[import]
    live_server,
    wait_for_passing,
)
from workflows import Context, Workflow, step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
    WorkflowIdleEvent,
)


class PromptEvent(InputRequiredEvent):
    prompt: str


class AnswerEvent(HumanResponseEvent):
    answer: str


class PausingWorkflow(Workflow):
    """Emits a HITL event, then waits for the human response."""

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> PromptEvent:
        return PromptEvent(prompt="what is your name?")

    @step
    async def finish(self, ctx: Context, ev: AnswerEvent) -> StopEvent:
        return StopEvent(result=f"hello, {ev.answer}")


def _make_server() -> WorkflowServer:
    server = WorkflowServer(workflow_store=MemoryWorkflowStore())
    server.add_workflow(
        "pausing",
        PausingWorkflow(),
        additional_events=[PromptEvent, AnswerEvent],
    )
    return server


async def _drive_until_idle(client: WorkflowClient, handler_id: str) -> None:
    """Drain events until WorkflowIdleEvent is persisted."""

    async def consume() -> None:
        async for env in client.get_workflow_events(
            handler_id,
            include_internal_events=True,
            after_sequence=-1,
        ):
            if env.type == WorkflowIdleEvent.__name__:
                return
        raise AssertionError("stream ended before WorkflowIdleEvent")

    await asyncio.wait_for(consume(), timeout=5.0)


@pytest.mark.asyncio
async def test_late_sse_client_receives_hitl_event_and_workflow_completes() -> None:
    """End-to-end invariant: late subscriber receives the prompt, answers, run completes.

    With the default runtime both idle signals (``idle_since`` and the
    ``WorkflowIdleEvent`` tail) are true after idling; this test pins the
    user-visible happy path. Branch-isolation for the DBOS-deferred
    ``idle_since=None`` case is covered by the next test.
    """
    async with live_server(_make_server) as (base_url, _server):
        client = WorkflowClient(base_url=base_url)
        started = await client.run_workflow_nowait("pausing")
        handler_id = started.handler_id

        await _drive_until_idle(client, handler_id)

        received: list[str] = []

        async def read_until_prompt() -> None:
            stream = client.get_workflow_events(
                handler_id,
                include_internal_events=False,
                after_sequence="now",
            )
            async for env in stream:
                received.append(env.type)
                if env.type == PromptEvent.__name__:
                    return
            raise AssertionError(
                f"stream ended without yielding PromptEvent (saw: {received})"
            )

        await asyncio.wait_for(read_until_prompt(), timeout=3.0)

        # Unblock the workflow and confirm it completes end-to-end.
        send = await client.send_event(handler_id, AnswerEvent(answer="world"))
        assert send.status == "sent"

        async def handler_completed() -> None:
            data = await client.get_handler(handler_id)
            assert data.status == "completed"
            assert data.result is not None
            assert data.result.value.get("result") == "hello, world"

        await wait_for_passing(handler_completed, max_duration=5.0, interval=0.05)


@pytest.mark.asyncio
async def test_fresh_sse_client_replays_when_idle_since_is_unset() -> None:
    """Rewind must fire on the event-log tail signal alone.

    Pins the DBOS scenario: ``DBOSIdleReleaseDecorator`` defers the handler's
    ``idle_since`` column until its idle_timeout expires, so the row looks
    "running" while the workflow is in fact blocked on a human response.
    """
    async with live_server(_make_server) as (base_url, server):
        client = WorkflowClient(base_url=base_url)
        started = await client.run_workflow_nowait("pausing")
        handler_id = started.handler_id

        await _drive_until_idle(client, handler_id)

        store = server._workflow_store
        found = await store.query(HandlerQuery(handler_id_in=[handler_id]))
        assert found, "handler must exist before clearing idle_since"
        persistent = found[0]
        assert persistent.run_id is not None
        await store.update_handler_status(persistent.run_id, idle_since=None)

        received: list[str] = []

        async def read_until_prompt() -> None:
            stream = client.get_workflow_events(
                handler_id,
                include_internal_events=False,
                after_sequence="now",
            )
            async for env in stream:
                received.append(env.type)
                if env.type == PromptEvent.__name__:
                    return
            raise AssertionError(
                f"stream ended without yielding PromptEvent (saw: {received})"
            )

        await asyncio.wait_for(read_until_prompt(), timeout=3.0)


@pytest.mark.asyncio
async def test_fresh_sse_client_not_rewound_when_not_idle() -> None:
    """Non-idle handlers must keep the unchanged "now = only new events" semantic."""

    class SlowStreamingEvent(Event):
        value: int

    class SlowStreamingWorkflow(Workflow):
        @step
        async def emit(self, ctx: Context, ev: StartEvent) -> StopEvent:
            for i in range(3):
                ctx.write_event_to_stream(SlowStreamingEvent(value=i))
                await asyncio.sleep(0.05)
            return StopEvent(result="done")

    def make_server() -> WorkflowServer:
        server = WorkflowServer(workflow_store=MemoryWorkflowStore())
        server.add_workflow(
            "slow",
            SlowStreamingWorkflow(),
            additional_events=[SlowStreamingEvent],
        )
        return server

    async with live_server(make_server) as (base_url, _server):
        client = WorkflowClient(base_url=base_url)
        started = await client.run_workflow_nowait("slow")
        handler_id = started.handler_id

        async def completed() -> None:
            data = await client.get_handler(handler_id)
            assert data.status == "completed"

        await wait_for_passing(completed, max_duration=3.0, interval=0.05)

        async with httpx.AsyncClient(base_url=base_url, timeout=3.0) as http:
            resp = await http.get(
                f"/events/{handler_id}",
                params={
                    "sse": "true",
                    "include_internal": "false",
                    "after_sequence": "now",
                },
            )
            assert resp.status_code == 204, (
                f"expected 204 (no new events) for completed non-idle run, "
                f"got {resp.status_code}: {resp.text[:200]}"
            )
