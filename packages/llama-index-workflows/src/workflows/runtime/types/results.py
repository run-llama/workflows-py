# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import dataclasses
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator
from workflows.events import Event
from workflows.runtime.types.serialization_helpers import (
    SerializableEvent,
    SerializableEventType,
    SerializableException,
    SerializableOptionalEvent,
)

EventType = TypeVar("EventType", bound=Event)

#################################################################
# State Passed to step functions and returned by step functions #
#################################################################


@dataclass(frozen=True)
class StepWorkerContext:
    """
    Base state passed to step functions and returned by step functions.
    """

    # immutable state of the step events at start of the step function execution
    state: StepWorkerState
    # add commands here to mutate the internal worker state after step execution
    returns: Returns


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
class Returns:
    """
    Mutate to add return values to the step function. These are only executed after the
    step function has completed (including errors!)
    """

    return_values: list[StepFunctionResult]


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


class StepWorkerResult(BaseModel):
    """Returned after a step function has been successfully executed."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    type: Literal["result"] = "result"
    result: SerializableOptionalEvent = None
    step_name: str | None = None


class StepWorkerFailed(BaseModel):
    """Returned after a step function has failed."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    type: Literal["failed"] = "failed"
    exception: SerializableException
    failed_at: float


class DeleteWaiter(BaseModel):
    """Returned after a waiter condition has been successfully resolved."""

    model_config = ConfigDict(frozen=True)
    type: Literal["delete_waiter"] = "delete_waiter"
    waiter_id: str


class DeleteCollectedEvent(BaseModel):
    """Returned after a collected event has been successfully resolved."""

    model_config = ConfigDict(frozen=True)
    type: Literal["delete_collected"] = "delete_collected"
    event_id: str


class AddCollectedEvent(BaseModel):
    """Returned after a collected event has been added, and is not yet resolved."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    type: Literal["add_collected"] = "add_collected"
    event_id: str
    event: SerializableEvent


class AddWaiter(BaseModel, Generic[EventType]):
    """Returned after a waiter has been added, and is not yet resolved."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    type: Literal["add_waiter"] = "add_waiter"
    waiter_id: str
    waiter_event: SerializableOptionalEvent = None
    requirements: dict[str, Any] = {}
    timeout: float | None = None
    event_type: SerializableEventType
    has_requirements: bool = False

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        # Always serialize requirements as {} and record whether they existed
        data["has_requirements"] = bool(self.requirements)
        data["requirements"] = {}
        return data

    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, data: Any, handler: Any) -> AddWaiter:
        if isinstance(data, dict):
            # Strip has_requirements before validation (it's computed)
            data = dict(data)
            data.pop("has_requirements", None)
        return handler(data)


# A step function result "command" communicates back to the workflow how the step function was resolved
# e.g. are we collecting events, waiting for an event, or just returning a result?
StepFunctionResult = Union[
    StepWorkerResult,
    StepWorkerFailed,
    AddCollectedEvent,
    DeleteCollectedEvent,
    AddWaiter[Event],
    DeleteWaiter,
]
