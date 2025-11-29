# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

from _collections_abc import dict_items, dict_keys, dict_values
from enum import Enum
from typing import Any, Literal, Type

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_serializer,
)


class DictLikeModel(BaseModel):
    """
    Base Pydantic model class that mimics a dict interface for dynamic fields.

    Known, typed fields behave like regular Pydantic attributes. Any extra
    keyword arguments are stored in an internal dict and can be accessed through
    both attribute and mapping semantics. This hybrid model enables flexible
    event payloads while preserving validation for declared fields.

    PrivateAttr:
        _data (dict[str, Any]): Underlying Python dict for dynamic fields.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _data: dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, **params: Any):
        """
        __init__.

        NOTE: fields and private_attrs are pulled from params by name.
        """
        # extract and set fields, private attrs and remaining shove in _data
        fields = {}
        private_attrs = {}
        data = {}
        for k, v in params.items():
            if k in self.__class__.model_fields:
                fields[k] = v
            elif k in self.__private_attributes__:
                private_attrs[k] = v
            else:
                data[k] = v
        super().__init__(**fields)
        for private_attr, value in private_attrs.items():
            super().__setattr__(private_attr, value)
        if data:
            self._data.update(data)

    def __getattr__(self, __name: str) -> Any:
        if (
            __name in self.__private_attributes__
            or __name in self.__class__.model_fields
        ):
            return super().__getattr__(__name)  # type: ignore
        else:
            if __name not in self._data:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{__name}'"
                )
            return self._data[__name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__private_attributes__ or name in self.__class__.model_fields:
            super().__setattr__(name, value)
        else:
            self._data.__setitem__(name, value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> "dict_keys[str, Any]":
        return self._data.keys()

    def values(self) -> "dict_values[str, Any]":
        return self._data.values()

    def items(self) -> "dict_items[str, Any]":
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Any:
        return iter(self._data)

    def to_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._data

    def __bool__(self) -> bool:
        """Make test `if event:` pass on Event instances."""
        return True

    @model_serializer(mode="wrap")
    def custom_model_dump(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        # include _data in serialization
        if self._data:
            data["_data"] = self._data
        return data


class Event(DictLikeModel):
    """
    Base class for all workflow events.

    Events are light-weight, serializable payloads passed between steps.
    They support both attribute and mapping access to dynamic fields.

    Examples:
        Subclassing with typed fields:

        ```python
        from pydantic import Field

        class CustomEv(Event):
            score: int = Field(ge=0)

        e = CustomEv(score=10)
        print(e.score)
        ```

    See Also:
        - [StartEvent][workflows.events.StartEvent]
        - [StopEvent][workflows.events.StopEvent]
        - [InputRequiredEvent][workflows.events.InputRequiredEvent]
        - [HumanResponseEvent][workflows.events.HumanResponseEvent]
    """

    def __init__(self, **params: Any):
        super().__init__(**params)


class StartEvent(Event):
    """Implicit entry event sent to kick off a `Workflow.run()`."""


class StopEvent(Event):
    """Terminal event that signals the workflow has completed.

    The `result` property contains the return value of the workflow run. When a
    custom stop event subclass is used, the workflow result is that event
    instance itself.

    Examples:
        ```python
        # default stop event: result holds the value
        return StopEvent(result={"answer": 42})
        ```

        Subclassing to provide a custom result:

        ```python
        class MyStopEv(StopEvent):
            pass

        @step
        async def my_step(self, ctx: Context, ev: StartEvent) -> MyStopEv:
            return MyStopEv(result={"answer": 42})
    """

    _result: Any = PrivateAttr(default=None)

    def __init__(self, result: Any = None, **kwargs: Any) -> None:
        # forces the user to provide a result
        super().__init__(_result=result, **kwargs)

    def _get_result(self) -> Any:
        """This can be overridden by subclasses to return the desired result."""
        return self._result

    @property
    def result(self) -> Any:
        return self._get_result()

    @model_serializer(mode="wrap")
    def custom_model_dump(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        # include _result in serialization for base StopEvent
        if self._result is not None:
            data["result"] = self._result
        return data

    def __repr__(self) -> str:
        dict_items = {**self._data, **self.model_dump()}
        # Format as key=value pairs
        parts = [f"{k}={v!r}" for k, v in dict_items.items()]
        dict_str = ", ".join(parts)
        return f"{self.__class__.__name__}({dict_str})"

    def __str__(self) -> str:
        return str(self._result)


class WorkflowFailedEvent(StopEvent):
    """Stop event emitted when a workflow fails due to an unhandled exception."""

    status: Literal["failed"] = Field(
        default="failed", description="Terminal state indicator."
    )
    step_name: str | None = Field(
        default=None, description="Name of the step that raised the exception."
    )
    error_type: str = Field(description="Exception type that caused the failure.")
    message: str | None = Field(
        default=None, description="String representation of the exception."
    )
    attempts: int | None = Field(
        default=None,
        description="Attempt count consumed for the failing step when retries exhausted.",
    )

    @classmethod
    def from_exception(
        cls,
        *,
        exception: Exception,
        step_name: str | None = None,
        attempts: int | None = None,
    ) -> "WorkflowFailedEvent":
        """Build a failure stop event from an exception instance."""
        message = str(exception).strip() or None
        return cls(
            step_name=step_name,
            error_type=type(exception).__name__,
            message=message,
            attempts=attempts,
        )


class WorkflowCancelledEvent(StopEvent):
    """Stop event emitted when a workflow run is cancelled."""

    status: Literal["cancelled"] = Field(
        default="cancelled", description="Terminal state indicator."
    )
    reason: str = Field(
        default="Run cancelled by user.",
        description="Human-readable reason for the cancellation.",
    )
    cancelled_by: str | None = Field(
        default="user",
        description="Identifier describing who initiated the cancellation.",
    )

    @classmethod
    def user_requested(
        cls, reason: str | None = None, cancelled_by: str = "user"
    ) -> "WorkflowCancelledEvent":
        """Build a cancellation event triggered by a user request."""
        final_reason = reason or "Run cancelled by user."
        return cls(reason=final_reason, cancelled_by=cancelled_by)


class WorkflowTimedOutEvent(StopEvent):
    """Stop event emitted when a workflow exceeds its allotted timeout."""

    status: Literal["timed_out"] = Field(
        default="timed_out", description="Terminal state indicator."
    )
    timeout_seconds: float = Field(
        description="Configured timeout threshold that was exceeded."
    )
    active_steps: list[str] = Field(
        default_factory=list,
        description="Names of steps that were still running when the timeout fired.",
    )
    message: str | None = Field(
        default=None,
        description="Detailed timeout message (typically mirrors the raised exception).",
    )

    @classmethod
    def build(
        cls,
        *,
        timeout_seconds: float,
        active_steps: list[str] | None = None,
        message: str | None = None,
    ) -> "WorkflowTimedOutEvent":
        """Build a timeout stop event with helpful metadata."""
        normalized_steps = sorted(active_steps or [])
        return cls(
            timeout_seconds=timeout_seconds,
            active_steps=normalized_steps,
            message=message,
        )


class InputRequiredEvent(Event):
    """Emitted when human input is required to proceed.

    Automatically written to the event stream if returned from a step.

    If returned from a step, it does not need to be consumed by other steps and will pass validation.
    It's expected that the caller will respond to this event and send back a [HumanResponseEvent][workflows.events.HumanResponseEvent].

    Use this directly or subclass it.

    Typical flow: a step returns `InputRequiredEvent`, callers consume it from
    the stream and send back a [HumanResponseEvent][workflows.events.HumanResponseEvent].

    Examples:
        ```python
        from workflows.events import InputRequiredEvent, HumanResponseEvent

        class HITLWorkflow(Workflow):
            @step
            async def my_step(self, ev: StartEvent) -> InputRequiredEvent:
                return InputRequiredEvent(prefix="What's your name? ")

            @step
            async def my_step(self, ev: HumanResponseEvent) -> StopEvent:
                return StopEvent(result=ev.response)
        ```
    """


class HumanResponseEvent(Event):
    """Carries a human's response for a prior input request.

    If consumed by a step and not returned by another, it will still pass validation.

    Examples:
        ```python
        from workflows.events import InputRequiredEvent, HumanResponseEvent

        class HITLWorkflow(Workflow):
            @step
            async def my_step(self, ev: StartEvent) -> InputRequiredEvent:
                return InputRequiredEvent(prefix="What's your name? ")

            @step
            async def my_step(self, ev: HumanResponseEvent) -> StopEvent:
                return StopEvent(result=ev.response)
        ```
    """


class InternalDispatchEvent(Event):
    """
    InternalDispatchEvent is a special event type that exposes processes running inside workflow, even if the user did not explicitly expose them by setting, e.g., `ctx.write_event_to_stream(`.

    Examples:
        ```python
        wf = ExampleWorkflow()
        handler = wf.run(message="Hello, who are you?")

        async for ev in handler.stream_event(expose_internal=True):
            if isinstance(ev, InternalDispatchEvent):
                print(type(ev), ev)
        ```
    """

    pass


class StepState(Enum):
    # is enqueued, but no capacity yet available to run
    PREPARING = "preparing"
    # is running now on a worker. Skips PREPARING if there is capacity available.
    RUNNING = "running"
    # is no longer running.
    NOT_RUNNING = "not_running"


class StepStateChanged(InternalDispatchEvent):
    """
    StepStateChanged is a special event type that exposes internal changes in the state of the event, including whether the step is running or in progress, what worker it is running on and what events it takes as input and output, as well as changes in the workflow state.

    Attributes:
        name (str): Name of the step
        step_state (StepState): State of the step ("running", "not_running", "in_progress", "not_in_progress", "exited")
        worker_id (str): ID of the worker that the step is running on
        input_event_name (str): Name of the input event
        output_event_name (Optional[str]): Name of the output event
        context_state (dict[str, Any]): Snapshot of the current workflow state
    """

    name: str = Field(description="Name of the step")
    step_state: StepState = Field(
        description="State of the step ('running', 'not_running', 'in_progress', 'not_in_progress', 'exited')"
    )
    worker_id: str = Field(description="ID of the worker that the step is running on")
    input_event_name: str = Field(description="Name of the input event")
    output_event_name: str | None = Field(
        description="Name of the output event", default=None
    )


EventType = Type[Event]
