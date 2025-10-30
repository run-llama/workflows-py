import asyncio

from workflows.client import WorkflowClient
from workflows.events import (
    StopEvent,
    HumanResponseEvent,
)


class ResponseEvent(HumanResponseEvent):
    response: str


class OutEvent(StopEvent):
    output: str


async def main() -> None:
    client = WorkflowClient(base_url="http://localhost:8000")
    handler = await client.run_workflow_nowait("human")
    handler_id = handler.handler_id
    print(handler_id)
    async for event in client.get_workflow_events(handler_id=handler_id):
        if "RequestEvent" == (event.type):
            print(
                "Workflow is requiring human input:",
                (event.value or {}).get("prompt", ""),
            )
            name = input("Reply here: ")
            sent_event = await client.send_event(
                handler_id=handler_id,
                event=ResponseEvent(response=name.capitalize().strip()),
            )
            msg = "Event has been sent" if sent_event else "Event failed to send"
            print(msg)
    result = await client.get_handler(handler_id)
    print(f"Workflow complete with status: {result.status})")
    res = OutEvent.model_validate(result.result)
    print("Received final message:", res.output)


if __name__ == "__main__":
    asyncio.run(main())
