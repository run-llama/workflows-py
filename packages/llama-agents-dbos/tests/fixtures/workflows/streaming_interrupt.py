# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Streaming workflow for interrupt/resume testing.

Like streaming_stress but collects all WorkDone events before completing,
ensuring stream events (including the interrupt signal) are visible before
the workflow can finish.
"""

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


class FanOutComplete(Event):
    pass


class StreamingInterruptWorkflow(Workflow):
    @step
    async def fan_out(self, ctx: Context, ev: StartEvent) -> FanOutComplete:
        for i in range(15):
            ctx.write_event_to_stream(ProgressEvent(progress=i))
            ctx.send_event(WorkItem(item_id=i))
        print("STEP:fan_out:dispatched_15_items", flush=True)
        ctx.write_event_to_stream(ProgressEvent(progress=999))
        return FanOutComplete()

    @step(num_workers=4)
    async def process_work(self, ctx: Context, ev: WorkItem) -> WorkDone:
        await asyncio.sleep(0.01)
        ctx.write_event_to_stream(ProgressEvent(progress=100 + ev.item_id))
        print(f"STEP:process_work:{ev.item_id}:complete", flush=True)
        return WorkDone(item_id=ev.item_id)

    @step
    async def after_fanout(self, ctx: Context, ev: FanOutComplete) -> None:
        print("STEP:after_fanout:complete", flush=True)
        return None

    @step
    async def collect(self, ctx: Context, ev: WorkDone) -> StopEvent | None:
        # Wait for all workers to finish, preventing early completion
        # that could race with stream event consumption
        results = ctx.collect_events(ev, [WorkDone] * 15)
        if results is None:
            return None
        print("STEP:collect:all_done", flush=True)
        return StopEvent(result={"collected": len(results)})
