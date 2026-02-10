---
sidebar:
  order: 12
title: Workflow Server
---

`WorkflowServer` exposes your workflows over HTTP and ships with a built-in debugger UI.

## Quick start

```python
# my_server.py
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from llama_agents.server import WorkflowServer


class GreetingWorkflow(Workflow):
    @step
    async def greet(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=f"Hello, {ev.get('name', 'World')}!")


server = WorkflowServer()
server.add_workflow("greet", GreetingWorkflow())
```

Run with the CLI:

```bash
python -m workflows.server my_server.py
```

Or programmatically:

```python
await server.serve(host="0.0.0.0", port=8080)
```

Defaults to `0.0.0.0:8080`. Configure with `WORKFLOWS_PY_SERVER_HOST` / `WORKFLOWS_PY_SERVER_PORT`.

## Debugger UI

A visual debugger is served at `/` — run, inspect, and send events to any registered workflow.

![Workflow Debugger UI](./assets/ui_sample.png)

Use `server.add_workflow(..., additional_events=[...])` to expose hidden events (e.g. from `ctx.wait_for_event()`) in the UI.

## API endpoints

| Method | Path                           | Description                        |
|--------|--------------------------------|------------------------------------|
| `GET`  | `/health`                      | Health check                       |
| `GET`  | `/workflows`                   | List workflows                     |
| `POST` | `/workflows/{name}/run`        | Run synchronously, return result   |
| `POST` | `/workflows/{name}/run-nowait` | Start async, return `handler_id`   |
| `GET`  | `/handlers/{handler_id}`       | Get handler status / result        |
| `GET`  | `/events/{handler_id}`         | Stream events (NDJSON / SSE)       |
| `POST` | `/events/{handler_id}`         | Send an event to a running workflow|
| `GET`  | `/handlers`                    | List all handlers                  |
| `POST` | `/handlers/{handler_id}/cancel`| Cancel a run                       |

The run endpoints accept a JSON body with `start_event`, `context`, and `handler_id` (to resume a previous run).

## Persistence

Handler state is in-memory by default. To survive restarts, pass a `SqliteWorkflowStore`:

```python
from llama_agents.server import SqliteWorkflowStore, WorkflowServer

store = SqliteWorkflowStore(db_path="handlers.db")
server = WorkflowServer(workflow_store=store)
```

That's it — status, context snapshots, and results are persisted to SQLite. Running workflows resume from their last checkpoint after a restart.

For multi-process deployments, implement `AbstractWorkflowStore` backed by a shared database.

### Example: resumable processing workflow

This workflow processes inputs one at a time, checkpointing progress after each. If the server restarts mid-run, it picks up where it left off.

```python
import asyncio
from llama_agents.server import SqliteWorkflowStore, WorkflowServer
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent


class ProgressEvent(Event):
    item: str
    index: int
    total: int


class ProcessInput(StartEvent):
    inputs: list[str] = Field(
        default_factory=lambda: ["alpha", "beta", "gamma"]
    )


class ProcessOutput(StopEvent):
    processed: list[str]


class ProcessingState(BaseModel):
    processed: list[str] = Field(default_factory=list)


class DurableProcessingWorkflow(Workflow):
    @step
    async def process_inputs(
        self, ev: ProcessInput, ctx: Context[ProcessingState]
    ) -> ProcessOutput:
        state = await ctx.store.get_state()
        processed = list(state.processed)

        for i, item in enumerate(ev.inputs):
            if item in processed:
                continue

            await asyncio.sleep(1)  # simulate work
            result = f"processed:{item}"
            processed.append(result)

            async with ctx.store.edit_state() as s:
                s.processed = list(processed)

            ctx.write_event_to_stream(
                ProgressEvent(item=result, index=i + 1, total=len(ev.inputs))
            )

        return ProcessOutput(processed=processed)


async def main() -> None:
    store = SqliteWorkflowStore(db_path="durable_handlers.db")
    server = WorkflowServer(workflow_store=store)
    server.add_workflow("processing", DurableProcessingWorkflow(timeout=120))
    await server.serve(host="localhost", port=8000)


if __name__ == "__main__":
    asyncio.run(main())
```

Start the server, then kick off a run (or resume one already in progress):

```python
import asyncio
from llama_agents.client import WorkflowClient


async def main() -> None:
    client = WorkflowClient(base_url="http://localhost:8000")

    # Resume an existing run, or start a new one
    handlers = await client.get_handlers(
        workflow_name=["processing"], status=["running"]
    )
    if handlers.handlers:
        handler_id = handlers.handlers[0].handler_id
        print(f"Resuming {handler_id}")
    else:
        h = await client.run_workflow_nowait("processing")
        handler_id = h.handler_id
        print(f"Started {handler_id}")

    async for event in client.get_workflow_events(handler_id):
        print(f"  [{event.type}] {event.value}")

    result = await client.get_handler(handler_id)
    print(f"Done: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
```

Stop the server mid-run with `Ctrl+C`, restart it, and run the client again — it picks up where it left off.
