import asyncio

from workflows.client.client import WorkflowClient

from workflows.events import StartEvent, HumanResponseEvent
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


class HumanApprovedResult(HumanResponseEvent):
    approved: bool


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
    handler_id = await client.run_workflow_nowait(
        "add_or_subtract",
        start_event=InputNumbers(a=1, b=3, operation="sum"),
        context=None,
    )
    print("==== STARTING THE WORKFLOW ===")
    print(f"Workflow running with handler ID: {handler_id}")
    print("=== STREAMING EVENTS ===")

    def handle_event(event_data: dict) -> None:
        print(f"Received event: {event_data}")

    # Stream events in background
    stream_task = asyncio.create_task(
        client.stream_events(
            handler_id=handler_id,
            event_callback=handle_event,
            sse=True,  # Use Server-Sent Events
        )
    )

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

    # Cancel streaming task
    stream_task.cancel()
    try:
        await stream_task
    except asyncio.CancelledError:
        pass

    print(f"Final result: {result}")
    return result


if __name__ == "__main__":
    asyncio.run(main())
