from workflows import Workflow, step, Context
from workflows.events import StartEvent, StopEvent, Event
from pydantic import Field
from workflows.server import WorkflowServer

from typing import Literal


class InputNumbers(StartEvent):
    a: int
    b: int
    operation: Literal["sum", "subtraction"] = Field(default="sum")


class CalculationEvent(Event):
    result: int


class OutputEvent(StopEvent):
    message: str


class AddOrSubtractWorkflow(Workflow):
    @step
    async def first_step(
        self, ev: InputNumbers, ctx: Context
    ) -> CalculationEvent | None:
        ctx.write_event_to_stream(ev)
        result = ev.a + ev.b if ev.operation == "sum" else ev.a - ev.b
        async with ctx.store.edit_state() as state:
            state.operation = ev.operation
            state.a = ev.a
            state.b = ev.b
            state.result = result
        ctx.write_event_to_stream(CalculationEvent(result=result))
        return CalculationEvent(result=result)

    @step
    async def second_step(self, ev: CalculationEvent, ctx: Context) -> OutputEvent:
        state = await ctx.store.get_state()
        return OutputEvent(
            message=f"You approved the result from your operation ({state.operation}) between {state.a} and {state.b}: {ev.result}"
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
