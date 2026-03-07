"""
Quick-start: durable workflow server with DBOS.

Run with:
    uv run examples/dbos/server_quickstart.py

By default this uses SQLite (zero setup). To use Postgres instead,
comment out the SQLite section and uncomment the Postgres section below.
"""

from __future__ import annotations

import asyncio

# ---------------------------------------------------------------------------
# SQLite (default — no external database needed)
# ---------------------------------------------------------------------------
from dbos import DBOS
from llama_agents.dbos import DBOSRuntime
from llama_agents.server import WorkflowServer
from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

# ---------------------------------------------------------------------------
# SQLite (default — no external database needed)
# ---------------------------------------------------------------------------
DBOS(config={"name": "quickstart", "run_admin_server": False})

# ---------------------------------------------------------------------------
# Postgres (uncomment this block and comment out the SQLite block above)
# ---------------------------------------------------------------------------
# from dbos import DBOS
#
# DBOS(
#     config={
#         "name": "quickstart",
#         "system_database_url": "postgresql://user:pass@localhost:5432/mydb",
#         "run_admin_server": False,
#         # Assign a unique executor_id per replica for multi-node setups:
#         # "executor_id": "replica-1",
#     }
# )
# ---------------------------------------------------------------------------


# -- Define a simple workflow ------------------------------------------------


class Tick(Event):
    count: int = Field(description="Current count")


class CounterResult(StopEvent):
    final_count: int = Field(description="Final counter value")


class CounterWorkflow(Workflow):
    """Counts to 20, emitting stream events along the way."""

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> Tick:
        return Tick(count=0)

    @step
    async def tick(self, ctx: Context, ev: Tick) -> Tick | CounterResult:
        count = ev.count + 1
        ctx.write_event_to_stream(Tick(count=count))
        print(f"  tick {count}")
        await asyncio.sleep(0.5)
        if count >= 20:
            return CounterResult(final_count=count)
        return Tick(count=count)


# -- Wire it up and serve ----------------------------------------------------


async def main() -> None:
    runtime = DBOSRuntime()

    server = WorkflowServer(
        workflow_store=runtime.create_workflow_store(),
        runtime=runtime.build_server_runtime(),
    )
    server.add_workflow("counter", CounterWorkflow(runtime=runtime, timeout=None))

    print("Serving on http://localhost:8000")
    print("Try: curl -X POST http://localhost:8000/workflows/counter/run")
    await server.start()
    await server.serve(
        host="0.0.0.0",
        port=8000,
        uvicorn_config={"timeout_graceful_shutdown": 0.1},
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
