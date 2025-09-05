# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

from _collections_abc import dict_items, dict_keys, dict_values
from typing import Any, Type, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_serializer,
)
from enum import Enum


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
    PREPARING = "preparing"
    RUNNING = "running"
    IN_PROGRESS = "in_progress"
    NOT_RUNNING = "not_running"
    NOT_IN_PROGRESS = "not_in_progress"
    EXITED = "exited"


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
    output_event_name: Optional[str] = Field(
        description="Name of the output event", default=None
    )
    context_state: Optional[dict[str, Any]] = Field(
        description="Snapshot of the current workflow state", default=None
    )


class EventsQueueChanged(InternalDispatchEvent):
    """
    A special event that reports the state of internal queues.

    Attributes:
        name (str): Name of the queue
        size (int): Size of the queue
    """

    name: str = Field(description="Name of the queue")
    size: int = Field(description="Size of the queue")


EventType = Type[Event]
