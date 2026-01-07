# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""
Commands returned by the control loop's tick reducer.

The control loop follows a reducer pattern:
  1. Wait for a tick (event, step result, timeout, etc.)
  2. Reduce the tick with current state -> (new_state, commands)
  3. Execute commands (which may spawn async tasks or queue new ticks)
  4. Repeat

Commands represent imperative actions to take after processing a tick,
such as starting workers, queuing events, or completing the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from workflows.events import Event, StopEvent


@dataclass(frozen=True)
class CommandRunWorker:
    step_name: str
    event: Event
    id: int


@dataclass(frozen=True)
class CommandQueueEvent:
    event: Event
    step_name: str | None = None
    delay: float | None = None
    attempts: int | None = None
    first_attempt_at: float | None = None


@dataclass(frozen=True)
class CommandHalt:
    exception: Exception


@dataclass(frozen=True)
class CommandCompleteRun:
    result: StopEvent


@dataclass(frozen=True)
class CommandFailWorkflow:
    step_name: str
    exception: Exception


@dataclass(frozen=True)
class CommandPublishEvent:
    event: Event


WorkflowCommand = Union[
    CommandRunWorker,
    CommandQueueEvent,
    CommandHalt,
    CommandCompleteRun,
    CommandFailWorkflow,
    CommandPublishEvent,
]


def indicates_exit(command: WorkflowCommand) -> bool:
    return (
        isinstance(command, CommandCompleteRun)
        or isinstance(command, CommandFailWorkflow)
        or isinstance(command, CommandHalt)
    )
