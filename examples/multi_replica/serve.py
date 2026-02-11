#!/usr/bin/env python3
"""
Multi-Replica Server

Single replica server that can be run standalone.

Usage:
    python examples/multi_replica/serve.py --port 8001
"""

from __future__ import annotations

import argparse
import asyncio

from dbos import DBOS
from llama_agents.dbos import DBOSRuntime
from llama_agents.server import WorkflowServer
from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

POSTGRES_DSN = "postgresql://workflows:workflows@localhost:5433/workflows"


class Tick(Event):
    count: int = Field(description="Current count")


class CounterResult(StopEvent):
    final_count: int = Field(description="Final counter value")


class CounterWorkflow(Workflow):
    """Counts to 20 with 1s delays, emitting Tick stream events."""

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> Tick:
        await ctx.store.set("count", 0)
        print("[Start] Initializing counter to 0")
        return Tick(count=0)

    @step
    async def increment(self, ctx: Context, ev: Tick) -> Tick | CounterResult:
        count = ev.count + 1
        await ctx.store.set("count", count)
        ctx.write_event_to_stream(Tick(count=count))
        print(f"[Tick {count:2d}] count = {count}")

        if count >= 20:
            return CounterResult(final_count=count)

        await asyncio.sleep(1.0)
        return Tick(count=count)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Replica Server")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    DBOS(
        config={
            "name": "multi-replica",
            "system_database_url": POSTGRES_DSN,
            "run_admin_server": False,
        }
    )

    runtime = DBOSRuntime()
    runtime.launch()
    store = runtime.create_workflow_store()
    server_runtime = runtime.build_server_runtime()

    server = WorkflowServer(workflow_store=store, runtime=server_runtime)
    server.add_workflow("counter", CounterWorkflow(), additional_events=[Tick])

    print(f"Serving on port {args.port}")
    await server.serve(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
