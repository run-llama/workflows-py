import asyncio

from workflows.client.client import WorkflowClient

from workflows.events import StartEvent
from pydantic import Field

from typing import Literal


class InputNumbers(StartEvent):
    a: int
    b: int
    operation: Literal["sum", "subtraction"] = Field(default="sum")


async def main() -> None:
    client = WorkflowClient(protocol="http", host="localhost", port=8000)
    workflows = await client.list_workflows()
    print("===== AVAILABLE WORKFLOWS ====")
    print(workflows)
    is_healthy = await client.is_healthy()
    print("==== HEALTH CHECK ====")
    print("Healthy" if is_healthy else "Not Healty :(")
    ping_time = await client.ping()
    print("==== PING TIME ====")
    print(ping_time, "ms")
    handler = await client.run_workflow_nowait(
        "add_or_subtract",
        start_event=InputNumbers(a=1, b=3, operation="sum"),
        context=None,
    )
    print("==== STARTING THE WORKFLOW ===")
    print(f"Workflow running with handler: {handler}")
    print("=== STREAMING EVENTS ===")
    async for event in client.get_workflow_events(handler):
        print("Received data:", event)
    # Poll for result
    result = handler.status.value
    while result == "running":
        try:
            result = await client.get_workflow_result(handler)
            if result != "running":
                break
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)

    print(f"Final result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
