# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Concurrent workers workflow fixture with num_workers=2."""

from __future__ import annotations

import asyncio
import random

from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.workflow import Workflow


class WorkItem(Event):
    item_id: int = Field(default=0)


class WorkDone(Event):
    item_id: int = Field(default=0)


class ConcurrentWorkersWorkflow(Workflow):
    @step
    async def dispatch(self, ctx: Context, ev: StartEvent) -> WorkItem:
        # Dispatch work items that will be processed by concurrent workers
        ctx.send_event(WorkItem(item_id=1))
        ctx.send_event(WorkItem(item_id=2))
        print("STEP:dispatch:complete", flush=True)
        return WorkItem(item_id=0)

    @step(num_workers=2)
    async def worker(self, ctx: Context, ev: WorkItem) -> WorkDone:
        # Variable processing time for each item
        await asyncio.sleep(random.uniform(0.01, 0.05))
        print(f"STEP:worker:{ev.item_id}:complete", flush=True)
        return WorkDone(item_id=ev.item_id)

    @step
    async def finish(self, ctx: Context, ev: WorkDone) -> StopEvent:
        # First WorkDone to arrive ends the workflow
        print(f"STEP:finish:{ev.item_id}:complete", flush=True)
        return StopEvent(result={"first_done": ev.item_id})
