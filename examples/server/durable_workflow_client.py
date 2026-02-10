# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Client for the durable workflow server example.

Starts a processing workflow, streams progress events, and fetches the final
result. You can stop and restart the server mid-run to see durability in action.

Usage:
    # 1. Start the server in one terminal:
    python examples/server/durable_workflow_server.py

    # 2. Run this client in another terminal:
    python examples/server/durable_workflow_client.py
"""

import asyncio

from llama_agents.client import WorkflowClient
from workflows.events import StartEvent


async def main() -> None:
    client = WorkflowClient(base_url="http://localhost:8000")

    # Verify the server is up.
    await client.is_healthy()
    workflows = await client.list_workflows()
    print(f"Available workflows: {workflows}")

    # Start the processing workflow asynchronously.
    handler = await client.run_workflow_nowait(
        "processing",
        start_event=StartEvent(items=["alpha", "beta", "gamma", "delta"]),
    )
    handler_id = handler.handler_id
    print(f"Workflow started — handler_id: {handler_id}")

    # Stream progress events as they arrive.
    print("Streaming events:")
    async for event in client.get_workflow_events(handler_id):
        print(f"  [{event.type}] {event.value}")

    # Fetch the final result.
    result = await client.get_handler(handler_id)
    print(f"Status : {result.status}")
    print(f"Result : {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
