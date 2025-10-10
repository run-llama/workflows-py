from workflows import Workflow, step, Context
from workflows.events import StartEvent, StopEvent, Event
from pydantic import Field
from workflows.server import WorkflowServer

from typing import Literal


class InputNumbers(StartEvent):
    a: int
    b: int
    operation: Literal["addition", "subtraction"] = Field(default="addition")


class CalculationEvent(Event):
    result: int


class OutputEvent(StopEvent):
    message: str


class AddOrSubtractWorkflow(Workflow):
    @step
    async def first_step(self, ev: InputNumbers, ctx: Context) -> OutputEvent | None:
        ctx.write_event_to_stream(ev)
        result = ev.a + ev.b if ev.operation == "addition" else ev.a - ev.b
        ctx.write_event_to_stream(CalculationEvent(result=result))
        return OutputEvent(
            message=f"You {ev.operation} operation ({ev.operation}) between {ev.a} and {ev.b}: {result}"
        )


async def main() -> None:
    server = WorkflowServer()
    server.add_workflow("add_or_subtract", AddOrSubtractWorkflow(timeout=1000))
    try:
        await server.serve("localhost", 8000)
    except KeyboardInterrupt:
        return
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
