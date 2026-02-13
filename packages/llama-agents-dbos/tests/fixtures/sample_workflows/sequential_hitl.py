# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Sequential HITL workflow fixture with ProcessedEvent and WaitForInputEvent."""

from __future__ import annotations

import asyncio
import random

from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.workflow import Workflow


class ProcessedEvent(Event):
    value: str = Field(default="")


class WaitForInputEvent(InputRequiredEvent):
    prompt: str = Field(default="")


class UserContinueEvent(HumanResponseEvent):
    continue_value: str = Field(default="")


class SequentialHITLWorkflow(Workflow):
    @step
    async def process(self, ctx: Context, ev: StartEvent) -> ProcessedEvent:
        await asyncio.sleep(random.uniform(0.01, 0.05))
        print("STEP:process:complete", flush=True)
        return ProcessedEvent(value="processed")

    @step
    async def ask_user(self, ctx: Context, ev: ProcessedEvent) -> WaitForInputEvent:
        print("STEP:ask_user:triggering_wait", flush=True)
        return WaitForInputEvent(prompt=f"Got {ev.value}")

    @step
    async def finalize(self, ctx: Context, ev: UserContinueEvent) -> StopEvent:
        print(f"STEP:finalize:complete:{ev.continue_value}", flush=True)
        return StopEvent(result={"continue": ev.continue_value})
