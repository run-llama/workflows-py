from workflows import Workflow, step
from workflows.context import Context
from workflows.events import (
    StartEvent,
    StopEvent,
    InputRequiredEvent,
    HumanResponseEvent,
)
from workflows.server import WorkflowServer


class RequestEvent(InputRequiredEvent):
    prompt: str


class ResponseEvent(HumanResponseEvent):
    response: str


class OutEvent(StopEvent):
    output: str


class HumanInTheLoopWorkflow(Workflow):
    @step
    async def prompt_human(self, ev: StartEvent, ctx: Context) -> RequestEvent:
        return RequestEvent(prompt="What is your name?")

    @step
    async def greet_human(self, ev: ResponseEvent) -> OutEvent:
        return OutEvent(output=f"Hello, {ev.response}")


async def main() -> None:
    server = WorkflowServer()
    server.add_workflow("human", HumanInTheLoopWorkflow(timeout=1000))
    try:
        await server.serve("localhost", 8000)
    except KeyboardInterrupt:
        return
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
