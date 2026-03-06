# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Non-HITL counter workflow matching the docs quickstart exactly.

Uses plain Tick events (not InputRequired), no DBOS.send/recv in the
workflow itself. This is the exact pattern from the DBOS quickstart docs
and durable_workflow.py example.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent


class Tick(Event):
    count: int = Field(description="Current count")


class CounterResult(StopEvent):
    final_count: int = Field(description="Final counter value")


class SimpleCounterWorkflow(Workflow):
    """Looping counter — increments until reaching target."""

    def __init__(self, target: int = 20, **kwargs: Any) -> None:
        super().__init__(timeout=None, **kwargs)
        self._target = target

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> Tick:
        await ctx.store.set("count", 0)
        print("STEP:start:complete", flush=True)
        return Tick(count=0)

    @step
    async def increment(self, ctx: Context, ev: Tick) -> Tick | CounterResult:
        count = ev.count + 1
        await ctx.store.set("count", count)
        print(f"STEP:increment:{count}", flush=True)

        if count >= self._target:
            return CounterResult(final_count=count)

        await asyncio.sleep(0.5)
        return Tick(count=count)
