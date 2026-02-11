# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Basic HITL workflow fixture with AskInputEvent and UserInput events."""

from __future__ import annotations

from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, InputRequiredEvent, StartEvent, StopEvent
from workflows.workflow import Workflow


class AskInputEvent(InputRequiredEvent):
    prefix: str = Field(default="Enter: ")


class UserInput(Event):
    response: str = Field(default="")


class TestWorkflow(Workflow):
    @step
    async def ask(self, ctx: Context, ev: StartEvent) -> AskInputEvent:
        await ctx.store.set("asked", True)
        print("STEP:ask:complete", flush=True)
        return AskInputEvent()

    @step
    async def process(self, ctx: Context, ev: UserInput) -> StopEvent:
        await ctx.store.set("processed", ev.response)
        print("STEP:process:complete", flush=True)
        return StopEvent(result={"response": ev.response})
