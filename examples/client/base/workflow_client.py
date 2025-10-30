import asyncio

from workflows.client import WorkflowClient

from workflows.events import StartEvent
from pydantic import Field

from typing import Literal


class InputNumbers(StartEvent):
    a: int
    b: int
    operation: Literal["addition", "subtraction"] = Field(default="addition")


async def main() -> None:
    client = WorkflowClient(base_url="http://localhost:8000")
    workflows = await client.list_workflows()
    print("===== AVAILABLE WORKFLOWS ====")
    print(workflows)
    await client.is_healthy()  # will raise an exception if the server is not healthy
    handler = await client.run_workflow_nowait(
        "add_or_subtract",
        start_event=InputNumbers(a=1, b=3, operation="addition"),
        context=None,
    )
    handler_id = handler.handler_id
    print("==== STARTING THE WORKFLOW ===")
    print(f"Workflow running with handler ID: {handler_id}")
    print("=== STREAMING EVENTS ===")

    async for event in client.get_workflow_events(handler_id=handler_id):
        print(f"Received event type={event.type} data={event.value}")
    result = await client.get_handler(handler_id)

    print(f"Final result: {result.result} (status: {result.status})")


if __name__ == "__main__":
    asyncio.run(main())
