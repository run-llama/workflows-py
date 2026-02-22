# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Subprocess runner for DBOS idle release e2e test over live HTTP.

Starts a real HTTP server with uvicorn, exercises the full idle release cycle
via WorkflowClient (the same path a real user takes), and verifies both
event stream continuity and handler completion status.

Usage:
    python idle_release_http_runner.py \
        --db-url "sqlite+pysqlite:///path/to/db?check_same_thread=false" \
        --idle-timeout 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn  # noqa: E402
from llama_agents.client import WorkflowClient  # noqa: E402
from llama_agents.server import WorkflowServer  # noqa: E402
from pydantic import Field  # noqa: E402
from runner_common import setup_dbos  # noqa: E402  # ty: ignore[unresolved-import]
from workflows import Context, Workflow, step  # noqa: E402
from workflows.events import (  # noqa: E402
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
    WorkflowIdleEvent,
)


class AskEvent(InputRequiredEvent):
    prompt: str = Field(default="Give me data")


class DataEvent(HumanResponseEvent):
    response: str = Field(default="")


class IdleWorkflow(Workflow):
    @step
    async def handle(self, ctx: Context, ev: StartEvent) -> StopEvent:
        data = await ctx.wait_for_event(
            DataEvent,
            waiter_event=AskEvent(prompt="Give me data"),
            waiter_id="data_req",
        )
        return StopEvent(result=f"got:{data.response}")


async def run(db_url: str, idle_timeout: float) -> None:
    dbos_runtime = setup_dbos(db_url, app_name="test-idle-http")

    wf = IdleWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    store = dbos_runtime.create_workflow_store()
    server_runtime = dbos_runtime.build_server_runtime(idle_timeout=idle_timeout)

    server = WorkflowServer(
        runtime=server_runtime,
        workflow_store=store,
        idle_timeout=60.0,
    )
    server.add_workflow("idle_wf", wf, additional_events=[DataEvent])

    # Bind to a random port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]

    config = uvicorn.Config(
        server.app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        loop="asyncio",
    )
    uv_server = uvicorn.Server(config)

    async def exercise() -> None:
        import httpx

        base_url = f"http://127.0.0.1:{port}"

        # Wait for server to be ready
        async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as hc:
            for _ in range(50):
                try:
                    resp = await hc.get("/health")
                    if resp.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.05)
            else:
                print("ERROR:server_not_ready", flush=True)
                return

        client = WorkflowClient(base_url=base_url)

        # 1. Start workflow
        handler = await client.run_workflow_nowait("idle_wf")
        print(f"HANDLER:{handler.handler_id}", flush=True)

        # 2. Watch event stream for WorkflowIdleEvent
        saw_idle = False
        stream_events: list[str] = []
        async for env in client.get_workflow_events(
            handler.handler_id, include_internal_events=True
        ):
            event = env.load_event([WorkflowIdleEvent, AskEvent])
            etype = type(event).__name__
            stream_events.append(etype)
            print(f"STREAM:{etype}", flush=True)
            if isinstance(event, WorkflowIdleEvent):
                saw_idle = True
                break

        if not saw_idle:
            print("ERROR:no_idle_event", flush=True)
            return
        print("IDLE_DETECTED", flush=True)

        # 3. Wait for idle timeout + buffer so the decorator releases
        await asyncio.sleep(idle_timeout + 1.0)
        print("TIMEOUT_ELAPSED", flush=True)

        # 4. Check handler status — should still be "running" (idle but not completed)
        pre_status = await client.get_handler(handler.handler_id)
        print(f"PRE_RESUME_STATUS:{pre_status.status}", flush=True)

        # 5. Send event to trigger resume
        send_result = await client.send_event(
            handler.handler_id, DataEvent(response="world")
        )
        print(f"SEND_STATUS:{send_result.status}", flush=True)

        # 6. Watch resumed event stream for StopEvent
        resumed_events: list[str] = []
        async for env in client.get_workflow_events(
            handler.handler_id, include_internal_events=True
        ):
            event = env.load_event([StopEvent])
            etype = type(event).__name__
            resumed_events.append(etype)
            print(f"RESUMED_STREAM:{etype}", flush=True)
            if isinstance(event, StopEvent):
                break

        # 7. Check handler status — should be "completed"
        post_status = await client.get_handler(handler.handler_id)
        for _ in range(20):
            if post_status.status == "completed":
                break
            await asyncio.sleep(0.1)
            post_status = await client.get_handler(handler.handler_id)
        print(f"FINAL_STATUS:{post_status.status}", flush=True)

        # 8. Check result
        if post_status.result is not None:
            result_val = post_status.result.value.get("result", "")
            print(f"RESULT:{result_val}", flush=True)
        else:
            print("ERROR:no_result", flush=True)

        print("SUCCESS", flush=True)

    try:
        await server.start()
        server_task = asyncio.create_task(uv_server.serve(sockets=[sock]))
        try:
            await asyncio.wait_for(exercise(), timeout=30.0)
        finally:
            uv_server.should_exit = True
            await server_task
            await server.stop()
    except Exception as e:
        import traceback

        print(f"ERROR:{type(e).__name__}:{e}", flush=True)
        traceback.print_exc()
    finally:
        sock.close()
        dbos_runtime.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--idle-timeout", type=float, default=0.5)
    args = parser.parse_args()
    asyncio.run(run(args.db_url, args.idle_timeout))


if __name__ == "__main__":
    main()
