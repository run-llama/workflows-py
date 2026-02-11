# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Chained workflow fixture with StepOneEvent, StepTwoEvent, and ChainedWorkflow."""

from __future__ import annotations

from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.workflow import Workflow


class StepOneEvent(Event):
    value: str = Field(default="one")


class StepTwoEvent(Event):
    value: str = Field(default="two")


class ChainedWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> StepOneEvent:
        await ctx.store.set("step_one", True)
        print("STEP:one:complete", flush=True)
        return StepOneEvent()

    @step
    async def step_two(self, ctx: Context, ev: StepOneEvent) -> StepTwoEvent:
        await ctx.store.set("step_two", True)
        print("STEP:two:complete", flush=True)
        return StepTwoEvent()

    @step
    async def step_three(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
        await ctx.store.set("step_three", True)
        print("STEP:three:complete", flush=True)
        return StopEvent(result="done")
