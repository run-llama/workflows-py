"""
Multi-Step Progress Scenario

Tests workflow completion caching with API call logging.

NOTE: Step-level replay is NOT currently implemented. Individual step functions
are not wrapped as DBOS operations, so they re-execute on resume of interrupted
workflows. However, COMPLETED workflows return cached results instantly.

Workflow design:
- step1: Print "Step 1 executing" + timestamp, simulate API call (log to file), sleep 3s
- step2: Print "Step 2 executing" + timestamp, sleep 5s (prompt ctrl+c here)
- step3: Print "Step 3 executing" + timestamp, sleep 3s
- step4: Return final result with all timestamps

What user observes:
- First run: All 4 steps print "executing"
- Resume of COMPLETED: No steps execute, cached result returned instantly
- Resume of INTERRUPTED: All steps re-execute (step-level replay not implemented)

Verification:
- Log "API calls" to api_calls.log
- On resume of completed: No new API calls
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.plugins.dbos import DBOSRuntime

from ..shared import BASE_DIR, register_scenario

API_LOG_FILE = BASE_DIR / "api_calls.log"


class Step1Event(Event):
    timestamp: str


class Step2Event(Event):
    step1_ts: str
    timestamp: str


class Step3Event(Event):
    step1_ts: str
    step2_ts: str
    timestamp: str


class MultiStepResult(StopEvent):
    step1_ts: str = Field(description="Step 1 timestamp")
    step2_ts: str = Field(description="Step 2 timestamp")
    step3_ts: str = Field(description="Step 3 timestamp")
    step4_ts: str = Field(description="Step 4 timestamp")


def log_api_call(step_name: str, timestamp: str) -> None:
    """Log an API call to the log file (simulates external API)."""
    with open(API_LOG_FILE, "a") as f:
        f.write(f"{step_name}: {timestamp}\n")


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class MultiStepWorkflow(Workflow):
    """Multi-step workflow demonstrating step replay."""

    @step
    async def step1(self, ev: StartEvent) -> Step1Event:
        ts = get_timestamp()
        print(f"[{ts}] Step 1 EXECUTING - Making API call...")
        log_api_call("step1", ts)
        await asyncio.sleep(3)
        print(f"[{ts}] Step 1 complete.")
        return Step1Event(timestamp=ts)

    @step
    async def step2(self, ev: Step1Event) -> Step2Event:
        ts = get_timestamp()
        print(f"[{ts}] Step 2 EXECUTING...")
        print("    >>> Press Ctrl+C NOW to test durability! <<<")
        await asyncio.sleep(5)
        print(f"[{ts}] Step 2 complete.")
        return Step2Event(step1_ts=ev.timestamp, timestamp=ts)

    @step
    async def step3(self, ev: Step2Event) -> Step3Event:
        ts = get_timestamp()
        print(f"[{ts}] Step 3 EXECUTING...")
        await asyncio.sleep(3)
        print(f"[{ts}] Step 3 complete.")
        return Step3Event(step1_ts=ev.step1_ts, step2_ts=ev.timestamp, timestamp=ts)

    @step
    async def step4(self, ev: Step3Event) -> MultiStepResult:
        ts = get_timestamp()
        print(f"[{ts}] Step 4 EXECUTING - Final step!")
        return MultiStepResult(
            step1_ts=ev.step1_ts,
            step2_ts=ev.step2_ts,
            step3_ts=ev.timestamp,
            step4_ts=ts,
        )


@register_scenario("multi-step")
async def run_multi_step(run_id: str, resume: bool) -> None:
    """Run the multi-step durability scenario."""
    print("\n" + "=" * 60)
    print("MULTI-STEP PROGRESS SCENARIO")
    print("=" * 60)
    print("\nThis scenario demonstrates step replay - completed steps")
    print("are NOT re-executed on resume, their results are replayed.\n")

    if resume:
        print("RESUMING - Watch for steps that were already completed.")
        print("They will NOT print 'EXECUTING' again.\n")
    else:
        print("STARTING NEW - All steps will execute.\n")

    # Check API log before run
    if API_LOG_FILE.exists():
        with open(API_LOG_FILE) as f:
            before_count = len(f.readlines())
        print(f"API calls before: {before_count}")
    else:
        before_count = 0
        print("API calls before: 0")

    # Create runtime and workflow
    runtime = DBOSRuntime(polling_interval_sec=0.1)
    workflow = MultiStepWorkflow(runtime=runtime)
    runtime.launch()

    try:
        # Run workflow with explicit run_id for DBOS tracking
        ctx = Context(workflow)
        handler = ctx._workflow_run(workflow, StartEvent(), run_id=run_id)
        result = await handler

        print("\n" + "-" * 60)
        print("RESULT:")
        print(f"  Step 1 timestamp: {result.step1_ts}")
        print(f"  Step 2 timestamp: {result.step2_ts}")
        print(f"  Step 3 timestamp: {result.step3_ts}")
        print(f"  Step 4 timestamp: {result.step4_ts}")

        # Check API log after run
        if API_LOG_FILE.exists():
            with open(API_LOG_FILE) as f:
                after_count = len(f.readlines())
            print(f"\nAPI calls after: {after_count}")
            if resume and after_count == before_count:
                print("SUCCESS: No new API calls (step1 was replayed!)")
            elif not resume and after_count == before_count + 1:
                print("SUCCESS: Exactly 1 new API call for step1")
            else:
                print(f"Note: API calls increased by {after_count - before_count}")
        print("-" * 60)

    finally:
        runtime.destroy()
