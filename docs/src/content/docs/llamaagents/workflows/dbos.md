---
sidebar:
  order: 14
title: DBOS Durable Execution
---

The [durable workflows](/python/llamaagents/workflows/durable_workflows) page shows how to make workflows survive restarts and errors using manual context snapshots. The `llama-agents-dbos` package removes that manual work by plugging a [DBOS](https://docs.dbos.dev/)-backed runtime into your workflows. Every step transition is persisted automatically, so a crashed workflow resumes exactly where it left off — no snapshot code required.

## Installation

```bash
pip install llama-agents-dbos
```

:::note
`llama-agents-dbos` requires Python 3.10 or later.
:::

## Quick Start — Standalone Durable Workflow

The simplest way to use DBOS is with SQLite (zero external dependencies). Define a workflow as usual, pass a `DBOSRuntime`, and your state is persisted automatically.

```python
import asyncio

from dbos import DBOS
from llama_agents.dbos import DBOSRuntime
from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent


# 1. Configure DBOS — SQLite by default
DBOS(config={"name": "counter-example", "run_admin_server": False})


# 2. Define events and workflow (nothing DBOS-specific here)
class Tick(Event):
    count: int = Field(description="Current count")


class CounterResult(StopEvent):
    final_count: int = Field(description="Final counter value")


class CounterWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> Tick:
        await ctx.store.set("count", 0)
        print("[Start] Initializing counter to 0")
        return Tick(count=0)

    @step
    async def increment(self, ctx: Context, ev: Tick) -> Tick | CounterResult:
        count = ev.count + 1
        await ctx.store.set("count", count)
        print(f"[Tick {count:2d}] count = {count}")

        if count >= 20:
            return CounterResult(final_count=count)

        await asyncio.sleep(0.5)
        return Tick(count=count)


# 3. Create runtime, attach to workflow, and launch
runtime = DBOSRuntime()
workflow = CounterWorkflow(runtime=runtime)
runtime.launch()


async def main() -> None:
    result = await workflow.run(run_id="counter-run-1")
    print(f"Result: final_count = {result.final_count}")


asyncio.run(main())
```

If you kill the process mid-run (e.g. Ctrl+C at tick 8), calling `workflow.run(run_id="counter-run-1")` again will resume from tick 8 instead of restarting from zero.

| Persists over `run` calls | ✅ |
| --- | --- |
| Persists over process restarts | ✅ |
| Survives runtime errors | ✅ |

## Durable Workflow Server

`DBOSRuntime` integrates with `WorkflowServer` so every workflow you serve gets durable execution out of the box. The runtime provides both the persistence store and the server runtime:

```python
import asyncio

from dbos import DBOS
from llama_agents.dbos import DBOSRuntime
from llama_agents.server import WorkflowServer
from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

DBOS(config={"name": "quickstart", "run_admin_server": False})


class Tick(Event):
    count: int = Field(description="Current count")


class CounterResult(StopEvent):
    final_count: int = Field(description="Final counter value")


class CounterWorkflow(Workflow):
    """Counts to 5, emitting stream events along the way."""

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> Tick:
        return Tick(count=0)

    @step
    async def tick(self, ctx: Context, ev: Tick) -> Tick | CounterResult:
        count = ev.count + 1
        ctx.write_event_to_stream(Tick(count=count))
        print(f"  tick {count}")
        await asyncio.sleep(0.5)
        if count >= 5:
            return CounterResult(final_count=count)
        return Tick(count=count)


async def main() -> None:
    runtime = DBOSRuntime()

    server = WorkflowServer(
        workflow_store=runtime.create_workflow_store(),
        runtime=runtime.build_server_runtime(),
    )
    server.add_workflow("counter", CounterWorkflow(runtime=runtime))

    print("Serving on http://localhost:8000")
    print("Try: curl -X POST http://localhost:8000/workflows/counter/run")
    await server.start()
    try:
        await server.serve(host="0.0.0.0", port=8000)
    finally:
        await server.stop()


asyncio.run(main())
```

The workflow debugger UI at `http://localhost:8000/` works exactly the same as with the default runtime — DBOS is transparent to the server layer.

## Idle Release

Long-running workflows that wait for external input (human-in-the-loop, webhooks, etc.) can sit idle in memory for extended periods. The `idle_timeout` parameter tells the DBOS runtime to release idle workflows from memory and resume them automatically when new events arrive:

```python
import asyncio

from dbos import DBOS
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

DBOS(config={"name": "idle-release-demo", "run_admin_server": False})


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


async def main() -> None:
    runtime = DBOSRuntime()

    server = WorkflowServer(
        workflow_store=runtime.create_workflow_store(),
        # Release workflows after 30 seconds of inactivity
        runtime=runtime.build_server_runtime(idle_timeout=30.0),
    )
    server.add_workflow("greeter", GreeterWorkflow(runtime=runtime))

    await server.start()
    try:
        await server.serve(host="0.0.0.0", port=8000)
    finally:
        await server.stop()


asyncio.run(main())
```

When the greeter workflow emits `AskName` and no input arrives within 30 seconds, the runtime releases it from memory. Once a `UserInput` event is sent (via `POST /events/{handler_id}`), the runtime transparently restores the workflow from the database and delivers the event. The caller never knows the workflow was released.

## Using Postgres for Multi-Replica Deployments

SQLite works well for single-process setups. For production deployments that need multiple server replicas, switch to Postgres. Each replica must have a unique `executor_id`:

```python
from dbos import DBOS

DBOS(config={
    "name": "my-app",
    "system_database_url": "postgresql://user:pass@localhost:5432/mydb",
    "run_admin_server": False,
    "executor_id": "replica-1",  # unique per replica
})
```

With Postgres:
- Multiple replicas share the same database for coordination
- Each replica owns and recovers the workflows it started
- Events sent to any replica are delivered to the correct owner via `pg_notify`
- If a replica crashes, its workflows are automatically recovered on restart

See the `examples/dbos/server_replicas.py` example for a complete multi-replica demo.

## How It Works

Under the hood, `DBOSRuntime` wraps the standard workflow runtime with a decorator chain that intercepts every step transition:

1. **Tick persistence** — each step completion is recorded to the database
2. **Event interception** — events are routed through DBOS durable messaging
3. **Idle release** — optional decorator that releases inactive workflows from memory

On resume, the runtime replays the persisted ticks to rebuild the workflow's `Context` and `store`, then continues execution from the last recorded step. This replay is deterministic — the same sequence of events always produces the same state.

For a detailed look at the internal architecture (adapter boundaries, cross-process event delivery, crash recovery), see the [DBOS adapter architecture doc](https://github.com/run-llama/workflows-py/blob/dev/packages/llama-agents-dbos/ARCHITECTURE.md).
