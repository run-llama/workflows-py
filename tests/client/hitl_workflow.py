import asyncio

from workflows import Workflow, step
from workflows.context import Context
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
    InputRequiredEvent,
    HumanResponseEvent,
)


class RequestEvent(InputRequiredEvent):
    prompt: str


class ResponseEvent(HumanResponseEvent):
    response: str


class HiddenEvent(Event):
    prompt: str


class OutEvent(StopEvent):
    output: str


class HumanInTheLoopWorkflow(Workflow):
    @step
    async def prompt_human(self, ev: StartEvent) -> RequestEvent:
        await asyncio.sleep(1)
        return RequestEvent(prompt="What is your name?")

    @step
    async def greet_human(self, ctx: Context, ev: ResponseEvent) -> OutEvent:
        follow_up = f"Nice to meet you, {ev.response}! Did you get my hidden message?"
        resp = await ctx.wait_for_event(
            ResponseEvent, waiter_event=HiddenEvent(prompt=follow_up)
        )
        return OutEvent(output=f"Got it!: {resp.response}")


hitl_wf = HumanInTheLoopWorkflow()
