import asyncio

from workflows.client import WorkflowClient

from workflows.events import StartEvent
from pydantic import PrivateAttr, model_validator, Field

from typing import Literal, Callable, Self


class InputNumbers(StartEvent):
    a: int
    b: int
    operation: Literal["sum", "subtraction"] = Field(default="sum")
    _function: Callable[[int, int], int] = PrivateAttr(default=lambda a, b: a + b)

    @model_validator(mode="after")
    def assign_function(self) -> Self:
        if self.operation == "subtraction":
            self._function = lambda a, b: a - b
        return self


async def main() -> None:
    client = WorkflowClient(protocol="http", host="localhost", port=8000)
    workflows = await client.list_workflows()
    print("===== AVAILABLE WORKFLOWS ====")
    print(workflows)
    is_healthy = await client.is_healthy()
    print("==== HEALTH CHECK ====")
    print("Healthy" if is_healthy else "Not Healty :(")
    handler = await client.run_workflow_nowait(
        "add_or_subtract",
        start_event=InputNumbers(a=1, b=3, operation="sum"),
        context=None,
    )
    handler_id = handler["handler_id"]
    print("==== STARTING THE WORKFLOW ===")
    print(f"Workflow running with handler ID: {handler_id}")
    print("=== STREAMING EVENTS ===")

    async for event in client.get_workflow_events(handler_id=handler_id):
        print("Received data:", event)

    # Poll for result
    result = None
    while result is None:
        try:
            result = await client.get_result(handler_id)
            if result is not None:
                break
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)

    print(f"Final result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
