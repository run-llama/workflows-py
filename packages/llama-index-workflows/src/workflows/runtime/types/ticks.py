# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
Ticks (events) that drive the control loop.

The control loop waits for ticks to arrive, then processes them through a reducer
to produce updated state and commands. Ticks represent all the different kinds of
events that can occur during workflow execution:
  - New events added to the workflow
  - Step function execution completing
  - Timeout occurring
  - User cancellation
  - External event publishing requests
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, TypeAdapter
from workflows.runtime.types.results import StepFunctionResult
from workflows.runtime.types.serialization_helpers import SerializableEvent


class TickStepResult(BaseModel):
    """When processed, executes a step function and publishes the result"""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    type: Literal["step_result"] = "step_result"
    step_name: str
    worker_id: int
    event: SerializableEvent
    result: list[Annotated[StepFunctionResult, Discriminator("type")]]


class TickAddEvent(BaseModel):
    """When sent, adds an event to the workflow's event queue"""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    type: Literal["add_event"] = "add_event"
    event: SerializableEvent
    step_name: str | None = None
    attempts: int | None = None
    first_attempt_at: float | None = None


class TickCancelRun(BaseModel):
    """When processed, cancels the workflow run"""

    model_config = ConfigDict(frozen=True)
    type: Literal["cancel_run"] = "cancel_run"


class TickPublishEvent(BaseModel):
    """When sent, publishes an event to workflow consumers, e.g. a UI or a callback"""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    type: Literal["publish_event"] = "publish_event"
    event: SerializableEvent


class TickTimeout(BaseModel):
    """When processed, times the workflow out, cancelling it"""

    model_config = ConfigDict(frozen=True)
    type: Literal["timeout"] = "timeout"
    timeout: float


class TickIdleCheck(BaseModel):
    """Scheduled after state appears idle, to re-check after async events drain.

    Appended to tick_buffer when the reducer sees quiescent state. Processed
    on the next loop iteration after asyncio.sleep(0), giving in-flight
    ctx.send_event() calls a chance to deliver via the pull task.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["idle_check"] = "idle_check"


WorkflowTick = Annotated[
    Union[
        TickStepResult,
        TickAddEvent,
        TickCancelRun,
        TickPublishEvent,
        TickTimeout,
        TickIdleCheck,
    ],
    Discriminator("type"),
]

WorkflowTickAdapter: TypeAdapter[WorkflowTick] = TypeAdapter(WorkflowTick)
