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
        items = getattr(ev, "things", ["item1", "item2", "item3", "item4", "item5"])

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


class ComplicatedInput(StartEvent):
    age: int
    name: str
    terrestrian: bool
    language: str


class ChildTerrestrialEvent(Event):
    greeting: str


class AdultTerrestrialEvent(Event):
    greeting: str


class ExtraTerrestrialEvent(Event):
    language: str
    greeting: str


class ComplicatedWorkflow(Workflow):
    @step
    async def first_step(
        self,
        ev: ComplicatedInput,
        ctx: Context,
    ) -> ChildTerrestrialEvent | AdultTerrestrialEvent | ExtraTerrestrialEvent:
        ctx.write_event_to_stream(ev)
        await asyncio.sleep(1)
        if ev.age < 18 and ev.terrestrian:
            ctx.write_event_to_stream(
                ChildTerrestrialEvent(
                    greeting=f"Hello, terrestrial child named {ev.name}"
                )
            )
            return ChildTerrestrialEvent(
                greeting=f"Hello, terrestrial child named {ev.name}"
            )
        elif ev.age >= 18 and ev.terrestrian:
            ctx.write_event_to_stream(
                AdultTerrestrialEvent(
                    greeting=f"My regards, terrestrial adult named {ev.name}"
                )
            )
            return AdultTerrestrialEvent(
                greeting=f"My regards, terrestrial adult named {ev.name}"
            )
        else:
            if ev.language.lower() == "martian":
                ctx.write_event_to_stream(
                    ExtraTerrestrialEvent(greeting="Ifmmp uifsf!", language="martian")
                )
                return ExtraTerrestrialEvent(
                    greeting="Ifmmp uifsf!", language="martian"
                )
            elif ev.language.lower() == "venusian":
                ctx.write_event_to_stream(
                    ExtraTerrestrialEvent(greeting="!ereht olleH", language="venusian")
                )
                return ExtraTerrestrialEvent(
                    greeting="!ereht olleH", language="venusian"
                )
            else:
                ctx.write_event_to_stream(
                    ExtraTerrestrialEvent(
                        greeting="Sorry, I do not speak your language",
                        language=ev.language,
                    )
                )
                return ExtraTerrestrialEvent(
                    greeting="Sorry, I do not speak your language", language=ev.language
                )

    @step
    async def terrestrial_child_step(self, ev: ChildTerrestrialEvent) -> StopEvent:
        await asyncio.sleep(1)
        return StopEvent(result="Hello back, old person")

    @step
    async def terrestrial_adult_step(self, ev: AdultTerrestrialEvent) -> StopEvent:
        await asyncio.sleep(1)
        return StopEvent(result="Hello back, young fella")

    @step
    async def extraterrestrial_step(self, ev: ExtraTerrestrialEvent) -> StopEvent:
        await asyncio.sleep(1)
        if ev.language == "martian":
            return StopEvent(result="Ifmmp cbdl!")
        elif ev.language == "venusian":
            return StopEvent(result="kcab olleH")
        else:
            return StopEvent(result="Xyubpifhabpmfsh!")


async def main() -> None:
    server = WorkflowServer()

    # Register workflows
    server.add_workflow("universe_communication", ComplicatedWorkflow())
    server.add_workflow("greeting", GreetingWorkflow())
    server.add_workflow("math", MathWorkflow())
    server.add_workflow("processing", ProcessingWorkflow())

    await server.serve(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
