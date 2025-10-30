import random

from workflows import Workflow, step, Context
from workflows.events import StartEvent, Event, StopEvent


class InputEvent(StartEvent):
    greeting: str
    name: str


class GreetEvent(Event):
    greeting: str
    exclamation_marks: int


class OutputEvent(StopEvent):
    greeting: str


class GreetingWorkflow(Workflow):
    @step
    async def first_step(self, ev: InputEvent, ctx: Context) -> GreetEvent:
        ctx.write_event_to_stream(ev)
        return GreetEvent(
            greeting=f"{ev.greeting} {ev.name}", exclamation_marks=random.randint(1, 10)
        )

    @step
    async def second_step(self, ev: GreetEvent, ctx: Context) -> OutputEvent:
        ctx.write_event_to_stream(ev)
        return OutputEvent(greeting=f"{ev.greeting}{'!' * ev.exclamation_marks}")


greeting_wf = GreetingWorkflow()


class CrashingWorkflow(Workflow):
    @step
    async def crashing_step(self, ev: StartEvent) -> StopEvent:
        raise ValueError("Workflow crashed intentionally")


crashing_wf = CrashingWorkflow()
