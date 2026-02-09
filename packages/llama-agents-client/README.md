# LlamaAgents Client

Async HTTP client for interacting with deployed [`llama-agents-server`](https://pypi.org/project/llama-agents-server/) instances.

## Installation

```bash
pip install llama-agents-client
```

## Quick Start

```python
import asyncio
from llama_agents.client import WorkflowClient

async def main():
    client = WorkflowClient(base_url="http://localhost:8080")

    # Run a workflow asynchronously
    handler = await client.run_workflow_nowait("my_workflow")

    # Stream events as they are produced
    async for event in client.get_workflow_events(handler.handler_id):
        print(f"Event: {event.type} -> {event.value}")

    # Get the final result
    result = await client.get_handler(handler.handler_id)
    print(f"Result: {result.result} (status: {result.status})")

asyncio.run(main())
```

## Features

- Run workflows synchronously or asynchronously
- Stream events in real-time as a workflow executes
- Human-in-the-loop support via `send_event` for injecting events into running workflows
- Bring your own `httpx.AsyncClient` for custom auth, headers, or transport

## Documentation

See the full [deployment guide](https://developers.llamaindex.ai/python/llamaagents/workflows/deployment/) for detailed usage and API reference.
