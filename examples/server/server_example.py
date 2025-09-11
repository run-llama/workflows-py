import asyncio

from workflows import Workflow, step
from workflows.context import Context
from workflows.events import Event, StartEvent, StopEvent
from workflows.server import WorkflowServer


class StreamEvent(Event):
    sequence: int


class GreetingInput(StartEvent):
    greeting: str
    name: str
    exclamation_marks: int
    formal: bool


class GreetingOutput(StopEvent):
    full_greeting: str
    is_formal: bool
    length: int


# Define a simple workflow
class GreetingWorkflow(Workflow):
    @step
    async def greet(self, ctx: Context, ev: GreetingInput) -> GreetingOutput:
        for i in range(3):
            ctx.write_event_to_stream(StreamEvent(sequence=i))
            await asyncio.sleep(0.3)

        name = getattr(ev, "name", "World")
        greeting = getattr(ev, "greeting", "Hello")
        excl_marks = getattr(ev, "exclamation_marks", 1)
        formal = getattr(ev, "formal", False)
        if formal:
            greeting = "Good Morning Your Honor"
        name = name + "!" * excl_marks
        return GreetingOutput(
            full_greeting=f"{greeting}, {name}",
            length=len(f"{greeting}, {name}"),
            is_formal=formal,
        )


class ProgressEvent(Event):
    step: str
    progress: int
    message: str


class MathWorkflow(Workflow):
    @step
    async def calculate(self, ev: StartEvent) -> StopEvent:
        a = getattr(ev, "a", 0)
        b = getattr(ev, "b", 0)
        operation = getattr(ev, "operation", "add")

        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        elif operation == "subtract":
            result = a - b
        elif operation == "divide":
            result = a / b if b != 0 else None
        else:
            result = None

        return StopEvent(
            result={"a": a, "b": b, "operation": operation, "result": result}
        )


class ProcessingWorkflow(Workflow):
    """Example workflow that demonstrates event streaming with progress updates."""

    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        items = getattr(ev, "items", ["item1", "item2", "item3", "item4", "item5"])

        ctx.write_event_to_stream(
            ProgressEvent(
                step="start",
                progress=0,
                message=f"Starting processing of {len(items)} items",
            )
        )

        results = []
        for i, item in enumerate(items):
            # Simulate processing time
            await asyncio.sleep(0.5)

            # Emit progress event
            progress = int((i + 1) / len(items) * 100)
            ctx.write_event_to_stream(
                ProgressEvent(
                    step="processing",
                    progress=progress,
                    message=f"Processed {item} ({i + 1}/{len(items)})",
                )
            )

            results.append(f"processed_{item}")

        ctx.write_event_to_stream(
            ProgressEvent(
                step="complete",
                progress=100,
                message="Processing completed successfully",
            )
        )

        return StopEvent(result={"processed_items": results, "total": len(results)})


async def main() -> None:
    server = WorkflowServer()

    # Register workflows
    server.add_workflow("greeting", GreetingWorkflow())
    server.add_workflow("math", MathWorkflow())
    server.add_workflow("processing", ProcessingWorkflow())

    await server.serve(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
