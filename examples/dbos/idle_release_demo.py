"""
Idle release demo — confirms INFO logging when a workflow idles and resumes.

Starts a server with a short idle_timeout (5s). A background task starts
a workflow, waits for idle release, then sends an event to trigger resume.
Watch the server logs for:
  - INFO ... Released idle DBOS handler [run_id=...]
  - INFO ... Resumed DBOS workflow via continue-as-new [old_run_id=..., new_run_id=...]

Run:
    uv run examples/dbos/idle_release_demo.py
"""

from __future__ import annotations

import asyncio
import logging

from dbos import DBOS
from llama_agents.client import WorkflowClient
from llama_agents.dbos import DBOSRuntime
from llama_agents.server import WorkflowServer
from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import (
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)

logging.basicConfig(level=logging.INFO)

DBOS(config={"name": "idle-release-demo", "run_admin_server": False})

IDLE_TIMEOUT = 5.0


class AskName(InputRequiredEvent):
    prompt: str = Field(default="What is your name?")


class UserInput(HumanResponseEvent):
    response: str = Field(default="")


class GreeterWorkflow(Workflow):
    @step
    async def ask(self, ctx: Context, ev: StartEvent) -> AskName:
        return AskName()

    @step
    async def greet(self, ctx: Context, ev: UserInput) -> StopEvent:
        return StopEvent(result={"greeting": f"Hello, {ev.response}!"})


async def drive_workflow(port: int) -> None:
    """Background task that exercises the idle → release → resume cycle."""
    await asyncio.sleep(1.0)  # let server start

    client = WorkflowClient(base_url=f"http://localhost:{port}")
    print("\n--- Starting workflow (nowait) ---")
    handler = await client.run_workflow_nowait("greeter")
    print(f"--- handler_id={handler.handler_id} ---")

    print(f"--- Waiting {IDLE_TIMEOUT + 2}s for idle release ---")
    await asyncio.sleep(IDLE_TIMEOUT + 2)

    print("--- Sending UserInput to trigger resume ---")
    await client.send_event(
        handler.handler_id,
        UserInput(response="world"),
    )

    print("--- Streaming result ---")
    stream = client.get_workflow_events(handler.handler_id)
    async for event in stream:
        print(f"--- Received event: {event.type} ---")
        if event.type == "StopEvent":
            result = event.value.get("result")
            print(f"--- Result: {result} ---")
            break


async def main() -> None:
    port = 8000
    runtime = DBOSRuntime()

    server = WorkflowServer(
        workflow_store=runtime.create_workflow_store(),
        runtime=runtime.build_server_runtime(idle_timeout=IDLE_TIMEOUT),
    )
    server.add_workflow("greeter", GreeterWorkflow(runtime=runtime))

    print(f"Serving on http://localhost:{port} (idle_timeout={IDLE_TIMEOUT}s)")
    driver = asyncio.create_task(drive_workflow(port))
    await server.start()
    try:
        await server.serve(host="0.0.0.0", port=port)
    finally:
        driver.cancel()
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
