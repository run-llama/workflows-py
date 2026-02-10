# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Streaming stress workflow fixture with many concurrent stream writes."""

from __future__ import annotations

import asyncio

from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.workflow import Workflow


class ProgressEvent(Event):
    progress: int = Field(default=0)


class WorkItem(Event):
    item_id: int = Field(default=0)


class WorkDone(Event):
    item_id: int = Field(default=0)
    total_processed: int = Field(default=0)


class FanOutComplete(Event):
    pass


class StreamingStressWorkflow(Workflow):
    @step
    async def fan_out(self, ctx: Context, ev: StartEvent) -> FanOutComplete:
        # Fire many stream writes and internal events concurrently
        # This creates many background tasks that call DBOS operations
        for i in range(15):
            ctx.write_event_to_stream(ProgressEvent(progress=i))
            ctx.send_event(WorkItem(item_id=i))
        print("STEP:fan_out:dispatched_15_items", flush=True)
        # Write completion signal to stream for interrupt tests
        ctx.write_event_to_stream(ProgressEvent(progress=999))
        return FanOutComplete()

    @step(num_workers=4)
    async def process_work(self, ctx: Context, ev: WorkItem) -> WorkDone:
        # Each worker also writes to stream, creating more concurrent DBOS ops
        await asyncio.sleep(0.01)  # Small delay to increase interleaving
        ctx.write_event_to_stream(ProgressEvent(progress=100 + ev.item_id))
        print(f"STEP:process_work:{ev.item_id}:complete", flush=True)
        return WorkDone(item_id=ev.item_id)

    @step
    async def after_fanout(self, ctx: Context, ev: FanOutComplete) -> None:
        # Consume FanOutComplete, don't trigger anything
        print("STEP:after_fanout:complete", flush=True)
        return None

    @step
    async def collect(self, ctx: Context, ev: WorkDone) -> StopEvent | None:
        results = ctx.collect_events(ev, [WorkDone] * 15)
        if results is None:
            return None
        print("STEP:collect:all_done", flush=True)
        return StopEvent(result={"collected": len(results)})
