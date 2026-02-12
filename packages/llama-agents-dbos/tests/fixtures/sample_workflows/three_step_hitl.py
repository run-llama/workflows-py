# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Three-step HITL workflow fixture with name and quest input events."""

from __future__ import annotations

from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import (
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.workflow import Workflow


class NameInputEvent(InputRequiredEvent):
    prefix: str = Field(default="Name: ")


class NameResponseEvent(HumanResponseEvent):
    response: str = Field(default="")


class QuestInputEvent(InputRequiredEvent):
    prefix: str = Field(default="Quest: ")


class QuestResponseEvent(HumanResponseEvent):
    response: str = Field(default="")


class HITLWorkflow(Workflow):
    @step
    async def ask_name(self, ctx: Context, ev: StartEvent) -> NameInputEvent:
        await ctx.store.set("asked_name", True)
        print("STEP:ask_name:complete", flush=True)
        return NameInputEvent()

    @step
    async def ask_quest(self, ctx: Context, ev: NameResponseEvent) -> QuestInputEvent:
        await ctx.store.set("name", ev.response)
        print(f"STEP:ask_quest:got_name={ev.response}", flush=True)
        print("STEP:ask_quest:complete", flush=True)
        return QuestInputEvent()

    @step
    async def complete(self, ctx: Context, ev: QuestResponseEvent) -> StopEvent:
        name = await ctx.store.get("name", default="unknown")
        print(f"STEP:complete:got_quest={ev.response}", flush=True)
        return StopEvent(result={"name": name, "quest": ev.response})
