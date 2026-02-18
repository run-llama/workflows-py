# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import dataclasses
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    Union,
)

from workflows.decorators import R
from workflows.events import Event

if TYPE_CHECKING:
    pass


EventType = TypeVar("EventType", bound=Event)

#################################################################
# State Passed to step functions and returned by step functions #
#################################################################


@dataclass(frozen=True)
class StepWorkerContext(Generic[R]):
    """
    Base state passed to step functions and returned by step functions.
    """

    # immutable state of the step events at start of the step function execution
    state: StepWorkerState
    # add commands here to mutate the internal worker state after step execution
    returns: Returns[R]


@dataclass(frozen=True)
class StepWorkerState:
    """
    State passed to step functions and returned by step functions.
    """

    step_name: str
    collected_events: dict[str, list[Event]]
    collected_waiters: list[StepWorkerWaiter]

    def _deepcopy(self) -> StepWorkerState:
        return StepWorkerState(
            step_name=self.step_name,
            collected_events={k: list(v) for k, v in self.collected_events.items()},
            collected_waiters=[dataclasses.replace(x) for x in self.collected_waiters],
        )


@dataclass()
class StepWorkerWaiter(Generic[EventType]):
    """
    Any current waiters for events that are or are not resolved. Upon resolution, step should provide a delete waiter command.
    """

    # the waiter id
    waiter_id: str
    # original event to replay once the condition is met
    event: Event
    # the type of event that is being waited for
    waiting_for_event: type[EventType]
    # the requirements for the waiting event to consider it met
    requirements: dict[str, Any]
    # requirements are not required to be serializable. Flag used during deserialization to re-ping the step function for the requirements
    has_requirements: bool
    # set to true when the waiting event has been resolved, such that the step can retrieve it
    resolved_event: EventType | None


@dataclass()
class Returns(Generic[R]):
    """
    Mutate to add return values to the step function. These are only executed after the
    step function has completed (including errors!)
    """

    return_values: list[StepFunctionResult[R]]


class WaitingForEvent(Exception, Generic[EventType]):
    """
    Raised when a step function is called, waiting for an event, but the event is not yet available.
    Handled by the step worker to instead add a waiter rather than failing. Step is retried with the original event
    once the waiting event is available.
    """

    def __init__(self, add: AddWaiter[EventType]):
        self.add = add
        super().__init__(f"Waiting for event {add.event_type}")

    add: AddWaiter[EventType]


StepWorkerStateContextVar = ContextVar[StepWorkerContext]("step_worker")


###################################
# Data returned by step functions #
###################################


@dataclass
class StepWorkerResult(Generic[R]):
    """
    Returned after a step function has been successfully executed.
    """

    result: R


@dataclass
class StepWorkerFailed(Generic[R]):
    """
    Returned after a step function has failed
    """

    exception: Exception
    failed_at: float


@dataclass
class DeleteWaiter:
    """
    Returned after a waiter condition has been successfully resolved.
    """

    waiter_id: str


@dataclass
class DeleteCollectedEvent:
    """
    Returned after a collected event has been successfully resolved.
    """

    event_id: str


@dataclass
class AddCollectedEvent:
    """
    Returned after a collected event has been added, and is not yet resolved.
    """

    event_id: str
    event: Event


@dataclass
class AddWaiter(Generic[EventType]):
    """
    Returned after a waiter has been added, and is not yet resolved.
    """

    waiter_id: str
    waiter_event: Event | None
    requirements: dict[str, Any]
    timeout: float | None
    event_type: type[EventType]


# A step function result "command" communicates back to the workflow how the step function was resolved
# e.g. are we collecting events, waiting for an event, or just returning a result?
StepFunctionResult = Union[
    StepWorkerResult[R],
    StepWorkerFailed[R],
    AddCollectedEvent,
    DeleteCollectedEvent,
    AddWaiter[Event],
    DeleteWaiter,
]
