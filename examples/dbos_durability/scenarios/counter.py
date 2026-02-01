"""
Counter Persistence Scenario

Tests workflow completion caching.

NOTE: Step-level replay is NOT currently implemented. Individual step functions
are not wrapped as DBOS operations, so they re-execute on resume of interrupted
workflows. However, COMPLETED workflows return cached results instantly.

Workflow design:
- step1: counter = 1, print, sleep 2s
- step2: counter += 1 (now 2), print, sleep 2s
- step3: counter += 1 (now 3), print, sleep 5s (prompt ctrl+c)
- step4: counter += 1 (now 4), print final

Important note: ctx.store uses InMemoryStateStore which is NOT persisted.

What user observes:
- First run: Steps 1-4 all print with incrementing counter, final = 4
- Resume of COMPLETED: No steps execute, cached result returned instantly
- Resume of INTERRUPTED: All steps re-execute (step-level replay not implemented)
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.plugins.dbos import DBOSRuntime

from ..shared import register_scenario


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class Count1Event(Event):
    count: int = Field(description="Counter after step 1")


class Count2Event(Event):
    count: int = Field(description="Counter after step 2")


class Count3Event(Event):
    count: int = Field(description="Counter after step 3")


class CounterResult(StopEvent):
    final_count: int = Field(description="Final counter value")


class CounterWorkflow(Workflow):
    """Counter workflow demonstrating step return value replay."""

    @step
    async def step1(self, ev: StartEvent) -> Count1Event:
        count = 1
        ts = get_timestamp()
        print(f"[{ts}] Step 1 EXECUTING - Counter: {count}")
        await asyncio.sleep(2)
        return Count1Event(count=count)

    @step
    async def step2(self, ev: Count1Event) -> Count2Event:
        count = ev.count + 1
        ts = get_timestamp()
        print(f"[{ts}] Step 2 EXECUTING - Counter: {count}")
        await asyncio.sleep(2)
        return Count2Event(count=count)

    @step
    async def step3(self, ev: Count2Event) -> Count3Event:
        count = ev.count + 1
        ts = get_timestamp()
        print(f"[{ts}] Step 3 EXECUTING - Counter: {count}")
        print("    >>> Press Ctrl+C NOW to test durability! <<<")
        await asyncio.sleep(5)
        return Count3Event(count=count)

    @step
    async def step4(self, ev: Count3Event) -> CounterResult:
        count = ev.count + 1
        ts = get_timestamp()
        print(f"[{ts}] Step 4 EXECUTING - Final Counter: {count}")
        return CounterResult(final_count=count)


@register_scenario("counter")
async def run_counter(run_id: str, resume: bool) -> None:
    """Run the counter persistence scenario."""
    print("\n" + "=" * 60)
    print("COUNTER PERSISTENCE SCENARIO")
    print("=" * 60)
    print("\nThis scenario demonstrates step return value replay.")
    print("Counter increments are preserved via DBOS step recording.\n")

    if resume:
        print("RESUMING - Completed steps will NOT print 'EXECUTING'.")
        print("The counter will continue from where it left off.\n")
    else:
        print("STARTING NEW - All 4 steps will execute.\n")

    # Create runtime and workflow
    runtime = DBOSRuntime(polling_interval_sec=0.1)
    workflow = CounterWorkflow(runtime=runtime)
    runtime.launch()

    try:
        # Run workflow with explicit run_id for DBOS tracking
        ctx = Context(workflow)
        handler = ctx._workflow_run(workflow, StartEvent(), run_id=run_id)
        result = await handler

        print("\n" + "-" * 60)
        print("RESULT:")
        if isinstance(result, CounterResult):
            print(f"  Final counter value: {result.final_count}")
            if result.final_count == 4:
                print("  SUCCESS: Counter reached expected value of 4!")
            else:
                print(f"  WARNING: Expected 4, got {result.final_count}")
        else:
            print(f"  {result}")
        print("-" * 60)

    finally:
        runtime.destroy()
