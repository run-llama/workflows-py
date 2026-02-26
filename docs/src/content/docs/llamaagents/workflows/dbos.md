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

See the `examples/dbos/server_replicas.py` example for a complete multi-replica demo.

## Execution Model

Understanding the DBOS execution model helps you write workflows that behave correctly across restarts and replicas.

### Replica ownership

Each replica is identified by its `executor_id` and **owns** every workflow it starts. A workflow and all of its steps run in the same process — there is no distribution of individual steps across replicas. This means your steps can safely rely on local state like in-memory caches, local files, or process-level singletons. The trade-off is that a single workflow's workload cannot be spread across multiple replicas.

### Journaling and replay

Step completions and stream events are journaled to the database. When a workflow resumes after a crash or an idle release, the runtime replays the journal to rebuild the workflow's `Context` and `store`, then continues from the last recorded step.

Because recovery is replay-based, **steps may execute more than once** if they were interrupted before the journal entry was committed. Design steps to be idempotent where possible, or use the context store to track progress within a step (as shown in the [durable workflows](/python/llamaagents/workflows/durable_workflows) page).

### Scaling and draining

Replica IDs and replica counts must be stable. If you scale down and remove a replica, any workflows that replica owned will be abandoned until that `executor_id` comes back. Before removing a replica, drain it by letting its in-flight workflows complete and not routing new work to it.

### Code changes and versioning

Since resumption is based on journal replay, changing a workflow's code while historical runs are still in progress can cause non-determinism — for example, a step that now accepts a different set of events than when the run was originally started. To avoid this:
- **Drain in-flight workflows** before deploying code changes, or
- **Register the updated workflow under a new name** so that old runs continue against the original code and new runs use the updated version

A workflow's name defaults to its module-qualified class name (e.g. `my_app.CounterWorkflow`). You can set it explicitly with the `workflow_name` parameter:

```python
wf = CounterWorkflow(runtime=runtime, workflow_name="counter-v2")
```

When using a server, the name passed to `add_workflow` is the HTTP route name, independent of the workflow's internal name:

```python
server.add_workflow("counter-v2", CounterWorkflow(runtime=runtime, workflow_name="counter-v2"))
```

### Event streaming behavior

When using `handler.stream_events()` in-process (outside of a server), DBOS streams are replayed from the beginning on each call. This means you will receive all events the workflow has ever emitted, not just new ones.

The [workflow server](/python/llamaagents/workflows/deployment) uses a cursor-based approach instead — its `GET /events/{handler_id}` endpoint tracks position so each consumer only receives events once.

### Crash recovery

When a replica restarts, DBOS automatically detects and relaunches any incomplete workflows belonging to its `executor_id`. No manual intervention is required — the replica picks up where it left off by replaying its journal.
