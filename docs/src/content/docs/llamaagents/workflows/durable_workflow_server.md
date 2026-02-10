---
sidebar:
  order: 14
title: Durable Workflow Server
---

When you serve workflows over HTTP with `WorkflowServer`, handler state is kept in memory by default. If the server process restarts, all running and completed handlers are lost.

By passing a `SqliteWorkflowStore` to the `WorkflowServer`, handler metadata — including the serialized workflow context — is automatically persisted to a SQLite database. This means that:

- Completed workflow results survive server restarts.
- Running workflows can be resumed from their last checkpoint after a crash.
- You can query historical handler state for debugging or auditing.

## Quick start

Install the server extra:

```bash
pip install 'llama-index-workflows[server]'
```

Create a server with SQLite-backed persistence:

```python
import asyncio
from llama_agents.server import SqliteWorkflowStore, WorkflowServer
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


class GreetingWorkflow(Workflow):
    @step
    async def greet(self, ev: StartEvent) -> StopEvent:
        name = ev.get("name", "World")
        return StopEvent(result=f"Hello, {name}!")


async def main() -> None:
    store = SqliteWorkflowStore(db_path="handlers.db")
    server = WorkflowServer(workflow_store=store)
    server.add_workflow("greet", GreetingWorkflow())
    await server.serve(host="localhost", port=8000)


if __name__ == "__main__":
    asyncio.run(main())
```

That single change — `SqliteWorkflowStore(db_path="handlers.db")` — is all that is needed. The `WorkflowServer` will now persist every handler update (status changes, context snapshots, results) to the `handlers.db` file.

## How it works

`WorkflowServer` manages the lifecycle of workflow runs through *handlers*. Each handler tracks:

| Field | Description |
|---|---|
| `handler_id` | Unique identifier for the run |
| `workflow_name` | Which workflow was started |
| `status` | `running`, `completed`, `failed`, or `cancelled` |
| `ctx` | Serialized workflow context (state store + event queues) |
| `result` | The `StopEvent` produced on completion |
| `error` | Error message if the run failed |
| `started_at` / `updated_at` / `completed_at` | Timestamps |

When you supply a `SqliteWorkflowStore`, every state transition is written to a SQLite table. On server startup, handlers that were previously running can be recovered and resumed.

### Store interface

`SqliteWorkflowStore` implements the `AbstractWorkflowStore` interface:

```python
class AbstractWorkflowStore(ABC):
    async def query(self, query: HandlerQuery) -> list[PersistentHandler]: ...
    async def update(self, handler: PersistentHandler) -> None: ...
    async def delete(self, query: HandlerQuery) -> int: ...
```

You can use `HandlerQuery` to filter handlers by id, workflow name, or status:

```python
from llama_agents.server import HandlerQuery

# Find all completed handlers for a specific workflow
completed = await store.query(
    HandlerQuery(workflow_name_in=["greet"], status_in=["completed"])
)
for h in completed:
    print(h.handler_id, h.result)
```

## Full example: durable processing workflow

The following example defines a workflow that processes a list of items one at a time, saving progress to the context state store after each item. If the server restarts mid-run, the workflow picks up where it left off.

### Server

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
    items: list[str] = Field(
        default_factory=lambda: ["alpha", "beta", "gamma"]
    )


class ProcessOutput(StopEvent):
    processed: list[str]


class ProcessingState(BaseModel):
    processed: list[str] = Field(default_factory=list)


class DurableProcessingWorkflow(Workflow):
    @step
    async def process_items(
        self,
        ev: ProcessInput,
        ctx: Context[ProcessingState],
    ) -> ProcessOutput:
        items = ev.items
        state = await ctx.store.get_state()
        processed = list(state.processed)

        for i, item in enumerate(items):
            if item in processed:
                continue  # already done before restart

            await asyncio.sleep(1)  # simulate work
            result = f"processed:{item}"
            processed.append(result)

            # Save progress so the next snapshot captures it.
            async with ctx.store.edit_state() as s:
                s.processed = list(processed)

            ctx.write_event_to_stream(
                ProgressEvent(item=result, index=i + 1, total=len(items))
            )

        return ProcessOutput(processed=processed)


async def main() -> None:
    store = SqliteWorkflowStore(db_path="durable_handlers.db")
    server = WorkflowServer(workflow_store=store)
    server.add_workflow(
        "processing", DurableProcessingWorkflow(timeout=120)
    )
    await server.serve(host="localhost", port=8000)


if __name__ == "__main__":
    asyncio.run(main())
```

### Client

```python
import asyncio
from llama_agents.client import WorkflowClient
from workflows.events import StartEvent


async def main() -> None:
    client = WorkflowClient(base_url="http://localhost:8000")
    await client.is_healthy()

    handler = await client.run_workflow_nowait(
        "processing",
        start_event=StartEvent(
            items=["alpha", "beta", "gamma", "delta"]
        ),
    )
    print(f"handler_id: {handler.handler_id}")

    async for event in client.get_workflow_events(handler.handler_id):
        print(f"  [{event.type}] {event.value}")

    result = await client.get_handler(handler.handler_id)
    print(f"Status: {result.status}")
    print(f"Result: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Try it

1. Start the server:
   ```bash
   python examples/server/durable_workflow_server.py
   ```

2. In another terminal, run the client:
   ```bash
   python examples/server/durable_workflow_client.py
   ```

3. While the workflow is processing, stop the server with `Ctrl+C` and restart it. The handler state is preserved in `durable_handlers.db`.

## Comparison with in-memory storage

| | In-memory (default) | SqliteWorkflowStore |
|---|---|---|
| Handler state persists across restarts | No | Yes |
| Completed results queryable after restart | No | Yes |
| Additional dependencies | None | None (uses stdlib `sqlite3`) |
| Performance overhead | Lowest | Minimal (local file I/O) |

## When to use a workflow store

- **Development / single-process deployments**: `SqliteWorkflowStore` is a good fit — zero external dependencies, file-based persistence.
- **Multi-process / distributed deployments**: Implement `AbstractWorkflowStore` backed by a shared database (e.g. PostgreSQL) so all server instances share handler state.
- **Ephemeral / stateless tasks**: The default in-memory store is fine when you don't need to survive restarts.
