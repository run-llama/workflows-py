---
sidebar:
  order: 13
title: Workflow Client
---

`WorkflowClient` is a Python client for interacting with a running [`WorkflowServer`](/python/llamaagents/workflows/server/).

## Basic usage

```python
import asyncio
from llama_agents.client import WorkflowClient
from workflows.events import StartEvent


async def main() -> None:
    client = WorkflowClient(base_url="http://localhost:8080")

    handler = await client.run_workflow_nowait(
        "greet", start_event=StartEvent(name="John")
    )

    async for event in client.get_workflow_events(handler.handler_id):
        print(event.type, event.value)

    result = await client.get_handler(handler.handler_id)
    print(result.result)


asyncio.run(main())
```

## Methods

| Method | Description |
|--------|-------------|
| `list_workflows()` | List registered workflows |
| `is_healthy()` | Check server health |
| `run_workflow(name, ...)` | Run and wait for result |
| `run_workflow_nowait(name, ...)` | Start async, return handler |
| `get_handler(handler_id)` | Get handler status / result |
| `get_handlers(status, workflow_name)` | List handlers with optional filters |
| `get_workflow_events(handler_id)` | Stream events (async generator) |
| `send_event(handler_id, event)` | Send an event to a running workflow |
| `cancel_handler(handler_id)` | Cancel a running workflow |

## Human-in-the-loop

For workflows that require external input, stream events until you see an input request, then send a response:

```python
from llama_agents.client import WorkflowClient
from workflows.events import InputRequiredEvent, HumanResponseEvent


class RequestEvent(InputRequiredEvent):
    prompt: str


class ResponseEvent(HumanResponseEvent):
    response: str


async def main() -> None:
    client = WorkflowClient(base_url="http://localhost:8080")
    handler = await client.run_workflow_nowait("human")

    async for event in client.get_workflow_events(handler.handler_id):
        if event.type == "RequestEvent":
            name = input(event.value.get("prompt", ""))
            await client.send_event(
                handler.handler_id,
                event=ResponseEvent(response=name),
            )

    result = await client.get_handler(handler.handler_id)
    print(result.result)
```
