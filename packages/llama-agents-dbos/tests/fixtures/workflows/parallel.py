# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Parallel workflow fixture with ResultAEvent, ResultBEvent, and ParallelWorkflow."""

from __future__ import annotations

import asyncio
import random

from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.workflow import Workflow


class ResultAEvent(Event):
    value: str = Field(default="")


class ResultBEvent(Event):
    value: str = Field(default="")


class ParallelWorkflow(Workflow):
    @step
    async def branch_a(self, ctx: Context, ev: StartEvent) -> ResultAEvent:
        # Variable processing time - may complete before or after branch_b
        await asyncio.sleep(random.uniform(0.01, 0.05))
        print("STEP:branch_a:complete", flush=True)
        return ResultAEvent(value="a_result")

    @step
    async def branch_b(self, ctx: Context, ev: StartEvent) -> ResultBEvent:
        # Variable processing time - may complete before or after branch_a
        await asyncio.sleep(random.uniform(0.01, 0.05))
        print("STEP:branch_b:complete", flush=True)
        return ResultBEvent(value="b_result")

    @step
    async def finish_a(self, ctx: Context, ev: ResultAEvent) -> StopEvent:
        print("STEP:finish_a:complete", flush=True)
        return StopEvent(result={"winner": "a", "value": ev.value})

    @step
    async def finish_b(self, ctx: Context, ev: ResultBEvent) -> StopEvent:
        print("STEP:finish_b:complete", flush=True)
        return StopEvent(result={"winner": "b", "value": ev.value})
