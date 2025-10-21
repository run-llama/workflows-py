"""
Human-in-the-Loop Recovery Scenario

Tests event delivery durability.

NOTE: Step-level replay is NOT currently implemented. Individual step functions
are not wrapped as DBOS operations. However, DBOS.recv and DBOS.send ARE durable,
meaning event delivery is recorded. COMPLETED workflows return cached results instantly.

Workflow design:
- step1: Emit InputRequiredEvent("What is your name?")
- step2: Receive HumanResponseEvent, emit InputRequiredEvent("What is your quest?")
- step3: Receive HumanResponseEvent, return greeting with both answers

What user observes:
- First run: Both questions asked, greeting returned
- Resume of COMPLETED: Result returned instantly, no questions asked
- Resume of INTERRUPTED: Steps re-execute (step-level replay not implemented)
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime

from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import (
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.plugins.dbos import DBOSRuntime

from ..shared import register_scenario


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class NameInputEvent(InputRequiredEvent):
    prefix: str = Field(default="What is your name? ")


class NameResponseEvent(HumanResponseEvent):
    """Response to the name prompt."""

    response: str = Field(description="User's name")
    pass


class QuestInputEvent(InputRequiredEvent):
    prefix: str = Field(default="What is your quest? ")


class QuestResponseEvent(HumanResponseEvent):
    """Response to the quest prompt."""

    response: str = Field(description="User's quest")
    pass


class HITLResult(StopEvent):
    name: str = Field(description="User's name")
    quest: str = Field(description="User's quest")


class HITLWorkflow(Workflow):
    """Human-in-the-loop workflow demonstrating event durability."""

    @step
    async def ask_name(self, ctx: Context, ev: StartEvent) -> NameInputEvent:
        ts = get_timestamp()
        print(f"[{ts}] Step 1 EXECUTING - Asking for name...")
        return NameInputEvent()

    @step
    async def ask_quest(self, ctx: Context, ev: NameResponseEvent) -> QuestInputEvent:
        ts = get_timestamp()
        # Store the name for later
        await ctx.store.set("name", ev.response)
        print(f"[{ts}] Step 2 EXECUTING - Got name: {ev.response}")
        print(f"[{ts}] Step 2 - Asking for quest...")
        print("    >>> Press Ctrl+C NOW to test durability! <<<")
        return QuestInputEvent()

    @step
    async def complete(self, ctx: Context, ev: QuestResponseEvent) -> HITLResult:
        ts = get_timestamp()
        name = await ctx.store.get("name", default="Unknown")
        quest = ev.response
        print(f"[{ts}] Step 3 EXECUTING - Got quest: {quest}")
        return HITLResult(name=name, quest=quest)


async def read_input(prompt: str) -> str:
    """Read input from stdin asynchronously."""
    print(prompt, end="", flush=True)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sys.stdin.readline)


@register_scenario("hitl")
async def run_hitl(run_id: str, resume: bool) -> None:
    """Run the human-in-the-loop durability scenario."""
    print("\n" + "=" * 60)
    print("HUMAN-IN-THE-LOOP RECOVERY SCENARIO")
    print("=" * 60)
    print("\nThis scenario demonstrates that human input survives restart.")
    print("Your responses are recorded durably.\n")

    if resume:
        print("RESUMING - Your previous answers should be preserved.")
        print("Watch for which prompts appear (completed ones won't re-ask).\n")
    else:
        print("STARTING NEW - You'll be asked two questions.\n")

    # Create runtime and workflow
    runtime = DBOSRuntime(polling_interval_sec=0.1)
    workflow = HITLWorkflow(runtime=runtime)
    runtime.launch()

    try:
        # Run workflow with explicit run_id for DBOS tracking
        ctx = Context(workflow)
        handler = ctx._workflow_run(workflow, StartEvent(), run_id=run_id)

        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                # Read input from user
                response = await read_input(event.prefix)
                response = response.strip()

                # Send the appropriate response type based on which input was requested
                if handler.ctx:
                    if isinstance(event, NameInputEvent):
                        handler.ctx.send_event(NameResponseEvent(response=response))
                    elif isinstance(event, QuestInputEvent):
                        handler.ctx.send_event(QuestResponseEvent(response=response))

        result = await handler

        print("\n" + "-" * 60)
        print("RESULT:")
        if isinstance(result, HITLResult):
            print(f"  Name: {result.name}")
            print(f"  Quest: {result.quest}")
            print(
                f"  Greeting: Hello {result.name}! Good luck on your quest to {result.quest}!"
            )
        else:
            print(f"  {result}")
        print("-" * 60)

    finally:
        runtime.destroy()
