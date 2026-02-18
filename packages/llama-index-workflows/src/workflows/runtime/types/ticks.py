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

from dataclasses import dataclass
from typing import Generic, Union

from workflows.decorators import R
from workflows.events import Event
from workflows.runtime.types.results import StepFunctionResult


@dataclass(frozen=True)
class TickStepResult(Generic[R]):
    """When processed, executes a step function and publishes the result"""

    step_name: str
    worker_id: int
    event: Event
    result: list[StepFunctionResult[R]]


@dataclass(frozen=True)
class TickAddEvent:
    """When sent, adds an event to the workflow's event queue"""

    event: Event
    step_name: str | None = None
    attempts: int | None = None
    first_attempt_at: float | None = None


@dataclass(frozen=True)
class TickCancelRun:
    """When processed, cancels the workflow run"""

    pass


@dataclass(frozen=True)
class TickPublishEvent:
    """When sent, publishes an event to workflow consumers, e.g. a UI or a callback"""

    event: Event


@dataclass(frozen=True)
class TickTimeout:
    """When processed, times the workflow out, cancelling it"""

    timeout: float


WorkflowTick = Union[
    TickStepResult[R], TickAddEvent, TickCancelRun, TickPublishEvent, TickTimeout
]
