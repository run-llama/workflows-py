# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""HITL counter workflow for double-restart testing."""

from __future__ import annotations

import asyncio
from typing import Any

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


class CounterTickEvent(InputRequiredEvent):
    count: int = Field(default=0)


class CounterContinueEvent(HumanResponseEvent):
    pass


class CounterWorkflow(Workflow):
    def __init__(self, target: int = 40, **kwargs: Any) -> None:
        super().__init__(timeout=None, **kwargs)
        self._target = target

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> CounterTickEvent:
        await ctx.store.set("count", 0)
        print("STEP:start:complete", flush=True)
        return CounterTickEvent(count=0)

    @step
    async def on_tick(
        self, ctx: Context, ev: CounterContinueEvent
    ) -> CounterTickEvent | StopEvent:
        count = await ctx.store.get("count", default=0)
        count += 1
        await ctx.store.set("count", count)
        print(f"STEP:increment:{count}", flush=True)

        if count >= self._target:
            return StopEvent(result={"count": count})

        await asyncio.sleep(0.2)
        return CounterTickEvent(count=count)
