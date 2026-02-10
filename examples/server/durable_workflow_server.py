# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Durable workflow server using SqliteWorkflowStore.

This example shows how to make a WorkflowServer persist workflow handler state
to a SQLite database. When the server restarts, previously started workflows
can be resumed from where they left off instead of being lost.

Run this server:
    python examples/server/durable_workflow_server.py

Then interact with it using the companion client:
    python examples/server/durable_workflow_client.py
"""

import asyncio

from llama_agents.server import SqliteWorkflowStore, WorkflowServer
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent


# -- Events ------------------------------------------------------------------


class ProgressEvent(Event):
    """Streamed to the client after each item is processed."""

    item: str
    index: int
    total: int


class ProcessInput(StartEvent):
    """Typed input for the processing workflow."""

    items: list[str] = Field(default_factory=lambda: ["alpha", "beta", "gamma"])


class ProcessOutput(StopEvent):
    """Typed output with the list of processed results."""

    processed: list[str]


# -- State model --------------------------------------------------------------


class ProcessingState(BaseModel):
    """Tracks which items have been processed so far.

    Stored inside the workflow Context and automatically serialized by the
    WorkflowServer when it persists handler state to SQLite.
    """

    processed: list[str] = Field(default_factory=list)


# -- Workflow -----------------------------------------------------------------


class DurableProcessingWorkflow(Workflow):
    """A multi-step processing workflow whose progress survives server restarts.

    Each item is processed one at a time with the current progress saved to the
    context state store. If the server restarts mid-run, the SqliteWorkflowStore
    allows the WorkflowServer to restore the handler and the context, so the
    workflow resumes from the last checkpoint rather than starting over.
    """

    @step
    async def process_items(
        self,
        ev: ProcessInput,
        ctx: Context[ProcessingState],
    ) -> ProcessOutput:
        items = ev.items

        # Retrieve already-processed items from a previous run (if any).
        state = await ctx.store.get_state()
        processed = list(state.processed)

        for i, item in enumerate(items):
            if item in processed:
                # Skip items completed before a restart.
                continue

            # Simulate work
            await asyncio.sleep(1)
            result = f"processed:{item}"
            processed.append(result)

            # Persist progress in the context state store so it is captured
            # the next time the server snapshots the handler to SQLite.
            async with ctx.store.edit_state() as s:
                s.processed = list(processed)

            ctx.write_event_to_stream(
                ProgressEvent(item=result, index=i + 1, total=len(items))
            )

        return ProcessOutput(processed=processed)


# -- Server setup -------------------------------------------------------------


async def main() -> None:
    # The SqliteWorkflowStore persists handler metadata (status, context
    # snapshot, result) to a local SQLite file. This means that if the server
    # process is killed and restarted, handlers that were running can be
    # recovered automatically.
    store = SqliteWorkflowStore(db_path="durable_handlers.db")

    server = WorkflowServer(workflow_store=store)
    server.add_workflow(
        "processing",
        DurableProcessingWorkflow(timeout=120),
    )

    print("Starting durable workflow server on http://localhost:8000")
    print("Handler state is persisted to durable_handlers.db")
    await server.serve(host="localhost", port=8000)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
