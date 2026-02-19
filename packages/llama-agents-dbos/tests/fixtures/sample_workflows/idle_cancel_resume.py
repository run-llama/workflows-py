# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Workflow that goes idle waiting for external input, for cancel/resume testing.

The workflow:
1. Receives StartEvent, stores a prefix, emits InputRequiredEvent
2. Goes idle waiting for ExternalDataEvent via wait_for_event
3. When resumed and event arrives, combines prefix + data and returns StopEvent
"""

from __future__ import annotations

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


class AskForDataEvent(InputRequiredEvent):
    prompt: str = Field(default="Provide data")


class ExternalDataEvent(HumanResponseEvent):
    response: str = Field(default="")


class IdleCancelResumeWorkflow(Workflow):
    @step
    async def request_data(self, ctx: Context, ev: StartEvent) -> StopEvent:
        print("STEP:request_data:waiting", flush=True)
        result = await ctx.wait_for_event(
            ExternalDataEvent,
            waiter_event=AskForDataEvent(prompt="Provide data"),
            waiter_id="data_request",
        )
        print(f"STEP:request_data:received:{result.response}", flush=True)
        return StopEvent(result=f"got:{result.response}")
