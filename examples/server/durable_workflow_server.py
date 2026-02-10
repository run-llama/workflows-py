# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Durable workflow server with SqliteWorkflowStore.

Run the server:
    python examples/server/durable_workflow_server.py

Run a client that resumes in-progress workflows or starts a new one:
    python examples/server/durable_workflow_server.py --client

Try stopping the server mid-run (Ctrl+C), restarting it, then running
the client again — it picks up where it left off.
"""

import asyncio
import sys

from llama_agents.server import SqliteWorkflowStore, WorkflowServer
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

# -- Events & state -----------------------------------------------------------


class ProgressEvent(Event):
    item: str
    index: int
    total: int


class ProcessInput(StartEvent):
    inputs: list[str] = Field(default_factory=lambda: ["alpha", "beta", "gamma"])


class ProcessOutput(StopEvent):
    processed: list[str]


class ProcessingState(BaseModel):
    processed: list[str] = Field(default_factory=list)


# -- Workflow -----------------------------------------------------------------


class DurableProcessingWorkflow(Workflow):
    """Processes inputs one at a time, checkpointing after each."""

    @step
    async def process_inputs(
        self,
        ev: ProcessInput,
        ctx: Context[ProcessingState],
    ) -> ProcessOutput:
        inputs = ev.inputs
        state = await ctx.store.get_state()
        processed = list(state.processed)

        for i, item in enumerate(inputs):
            if item in processed:
                continue

            await asyncio.sleep(1)  # simulate work
            result = f"processed:{item}"
            processed.append(result)

            async with ctx.store.edit_state() as s:
                s.processed = list(processed)

            ctx.write_event_to_stream(
                ProgressEvent(item=result, index=i + 1, total=len(inputs))
            )

        return ProcessOutput(processed=processed)


# -- Server -------------------------------------------------------------------


async def run_server() -> None:
    store = SqliteWorkflowStore(db_path="durable_handlers.db")
    server = WorkflowServer(workflow_store=store)
    server.add_workflow(
        "processing",
        DurableProcessingWorkflow(timeout=120),
    )
    print("Starting durable workflow server on http://localhost:8000")
    await server.serve(host="localhost", port=8000)


# -- Inline client ------------------------------------------------------------


async def run_client() -> None:
    from llama_agents.client import WorkflowClient

    client = WorkflowClient(base_url="http://localhost:8000")

    # Resume an existing run, or start a new one
    handlers = await client.get_handlers(
        workflow_name=["processing"], status=["running"]
    )
    if handlers.handlers:
        handler_id = handlers.handlers[0].handler_id
        print(f"Resuming {handler_id}")
    else:
        handler = await client.run_workflow_nowait(
            "processing",
            start_event=ProcessInput(inputs=["alpha", "beta", "gamma", "delta"]),
        )
        handler_id = handler.handler_id
        print(f"Started {handler_id}")

    async for event in client.get_workflow_events(handler_id):
        print(f"  [{event.type}] {event.value}")

    result = await client.get_handler(handler_id)
    print(f"Status: {result.status}")
    print(f"Result: {result.result}")


# -- Entrypoint --------------------------------------------------------------

if __name__ == "__main__":
    try:
        if "--client" in sys.argv:
            asyncio.run(run_client())
        else:
            asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
