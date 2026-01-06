# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from typing import Any, cast

import pytest
from pydantic import PrivateAttr
from workflows.context import JsonSerializer
from workflows.events import (
    Event,
    StopEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
    WorkflowTimedOutEvent,
)


class _TestEvent(Event):
    param: str
    _private_param_1: str = PrivateAttr()
    _private_param_2: str = PrivateAttr(default_factory=str)


class _TestEvent2(Event):
    """
    Custom Test Event.

    Private Attrs:
        _private_param: doesn't get modified during construction
        _modified_private_param: gets processed before being set
    """

    _private_param: int = PrivateAttr()
    _modified_private_param: int = PrivateAttr()

    def __init__(self, _modified_private_param: int, **params: Any):
        super().__init__(**params)
        self._modified_private_param = _modified_private_param * 2


def test_event_init_basic() -> None:
    evt = Event(a=1, b=2, c="c")

    assert evt.a == 1
    assert evt.b == 2
    assert evt.c == "c"
    assert evt["a"] == evt.a
    assert evt["b"] == evt.b
    assert evt["c"] == evt.c
    assert evt.keys() == {"a": 1, "b": 2, "c": "c"}.keys()


def test_custom_event_with_fields_and_private_params() -> None:
    evt = _TestEvent(a=1, param="test_param", _private_param_1="test_private_param_1")  # type: ignore

    assert evt.a == 1
    assert evt["a"] == evt.a
    assert evt.param == "test_param"
    assert evt._data == {"a": 1}
    assert evt._private_param_1 == "test_private_param_1"
    assert evt._private_param_2 == ""


def test_custom_event_override_init() -> None:
    evt = _TestEvent2(a=1, b=2, _private_param=2, _modified_private_param=2)

    assert evt.a == 1
    assert evt.b == 2
    assert evt._data == {"a": 1, "b": 2}
    assert evt._private_param == 2
    assert evt._modified_private_param == 4


def test_event_missing_key() -> None:
    ev = _TestEvent(param="bar")
    with pytest.raises(AttributeError):
        ev.wrong_key


def test_event_not_a_field() -> None:
    ev = _TestEvent(param="foo", not_a_field="bar")  # type: ignore
    assert ev._data["not_a_field"] == "bar"
    ev.not_a_field = "baz"
    assert ev._data["not_a_field"] == "baz"
    ev["not_a_field"] = "barbaz"
    assert ev._data["not_a_field"] == "barbaz"
    assert ev.get("not_a_field") == "barbaz"


def test_event_dict_api() -> None:
    ev = _TestEvent(param="foo")
    assert len(ev) == 0
    ev["a_new_key"] = "bar"
    assert len(ev) == 1
    assert list(ev.values()) == ["bar"]
    k, v = next(iter(ev.items()))
    assert k == "a_new_key"
    assert v == "bar"
    assert next(iter(ev)) == "a_new_key"
    assert ev.to_dict() == {"a_new_key": "bar"}


def test_event_serialization() -> None:
    ev = _TestEvent(param="foo", not_a_field="bar")  # type: ignore
    serializer = JsonSerializer()
    serialized_ev = serializer.serialize(ev)
    deseriazlied_ev = serializer.deserialize(serialized_ev)

    assert type(deseriazlied_ev).__name__ == type(ev).__name__
    deseriazlied_ev = cast(
        _TestEvent,
        deseriazlied_ev,
    )
    assert ev.param == deseriazlied_ev.param
    assert ev._data == deseriazlied_ev._data


def test_bool() -> None:
    assert bool(_TestEvent(param="foo")) is True


def test_stop_event_serialization() -> None:
    ev = StopEvent(result="foo")
    data_dict = ev.model_dump()
    assert data_dict == {"result": "foo"}

    serializer = JsonSerializer()
    serialized_ev = serializer.serialize(ev)
    deseriazlied_ev = serializer.deserialize(serialized_ev)

    assert type(deseriazlied_ev).__name__ == type(ev).__name__
    deseriazlied_ev = cast(
        StopEvent,
        deseriazlied_ev,
    )
    assert ev.result == deseriazlied_ev.result


class CustomStopEvent(StopEvent):
    foo: str
    bar: int


def test_custom_stop_event_serialization() -> None:
    ev = CustomStopEvent(foo="foo", bar=42)
    data_dict = ev.model_dump()
    assert data_dict == {"foo": "foo", "bar": 42}

    serializer = JsonSerializer()
    serialized_ev = serializer.serialize(ev)
    deserialized_ev = serializer.deserialize(serialized_ev)

    assert type(deserialized_ev).__name__ == type(ev).__name__
    deserialized_ev = cast(
        CustomStopEvent,
        deserialized_ev,
    )
    assert ev.foo == deserialized_ev.foo
    assert ev.bar == deserialized_ev.bar


def test_stop_event_repr() -> None:
    ev = StopEvent(foo="foo", result=42)
    assert repr(ev) == "StopEvent(foo='foo', result=42)"


def test_custom_stop_event_repr_no_result() -> None:
    ev = CustomStopEvent(foo="foo", bar=42)
    rep = repr(ev)
    assert rep == "CustomStopEvent(foo='foo', bar=42)"


# Tests for workflow termination event subclasses


def test_workflow_termination_events_are_stop_events() -> None:
    """Verify workflow termination events are subclasses of StopEvent."""
    assert issubclass(WorkflowTimedOutEvent, StopEvent)
    assert issubclass(WorkflowCancelledEvent, StopEvent)
    assert issubclass(WorkflowFailedEvent, StopEvent)


def test_workflow_timed_out_event() -> None:
    """Test WorkflowTimedOutEvent creation and attributes."""
    ev = WorkflowTimedOutEvent(timeout=30.0, active_steps=["step1", "step2"])
    assert ev.timeout == 30.0
    assert ev.active_steps == ["step1", "step2"]
    assert isinstance(ev, StopEvent)


def test_workflow_timed_out_event_empty_active_steps() -> None:
    """Test WorkflowTimedOutEvent with no active steps."""
    ev = WorkflowTimedOutEvent(timeout=5.0, active_steps=[])
    assert ev.timeout == 5.0
    assert ev.active_steps == []


def test_workflow_timed_out_event_serialization() -> None:
    """Test WorkflowTimedOutEvent serialization and deserialization."""
    ev = WorkflowTimedOutEvent(timeout=30.0, active_steps=["step1", "step2"])
    data_dict = ev.model_dump()
    assert data_dict == {"timeout": 30.0, "active_steps": ["step1", "step2"]}

    serializer = JsonSerializer()
    serialized_ev = serializer.serialize(ev)
    deserialized_ev = serializer.deserialize(serialized_ev)

    assert type(deserialized_ev).__name__ == type(ev).__name__
    deserialized_ev = cast(WorkflowTimedOutEvent, deserialized_ev)
    assert ev.timeout == deserialized_ev.timeout
    assert ev.active_steps == deserialized_ev.active_steps


def test_workflow_timed_out_event_repr() -> None:
    """Test WorkflowTimedOutEvent string representation."""
    ev = WorkflowTimedOutEvent(timeout=10.0, active_steps=["my_step"])
    rep = repr(ev)
    assert "WorkflowTimedOutEvent" in rep
    assert "timeout=10.0" in rep
    assert "active_steps=['my_step']" in rep


def test_workflow_cancelled_event() -> None:
    """Test WorkflowCancelledEvent creation."""
    ev = WorkflowCancelledEvent()
    assert isinstance(ev, StopEvent)


def test_workflow_cancelled_event_serialization() -> None:
    """Test WorkflowCancelledEvent serialization and deserialization."""
    ev = WorkflowCancelledEvent()
    data_dict = ev.model_dump()
    assert data_dict == {}

    serializer = JsonSerializer()
    serialized_ev = serializer.serialize(ev)
    deserialized_ev = serializer.deserialize(serialized_ev)

    assert type(deserialized_ev).__name__ == type(ev).__name__


def test_workflow_cancelled_event_repr() -> None:
    """Test WorkflowCancelledEvent string representation."""
    ev = WorkflowCancelledEvent()
    rep = repr(ev)
    assert rep == "WorkflowCancelledEvent()"


def test_workflow_failed_event() -> None:
    """Test WorkflowFailedEvent creation and attributes."""
    ev = WorkflowFailedEvent(
        step_name="my_step",
        exception_type="builtins.ValueError",
        exception_message="Something went wrong",
        traceback="Traceback (most recent call last):\n  ...\nValueError: Something went wrong\n",
    )
    assert ev.step_name == "my_step"
    assert ev.exception_type == "builtins.ValueError"
    assert ev.exception_message == "Something went wrong"
    assert "ValueError" in ev.traceback
    assert isinstance(ev, StopEvent)


def test_workflow_failed_event_serialization() -> None:
    """Test WorkflowFailedEvent serialization and deserialization."""
    ev = WorkflowFailedEvent(
        step_name="failing_step",
        exception_type="builtins.RuntimeError",
        exception_message="Test failure",
        traceback="Traceback...\nRuntimeError: Test failure\n",
    )
    data_dict = ev.model_dump()
    assert data_dict == {
        "step_name": "failing_step",
        "exception_type": "builtins.RuntimeError",
        "exception_message": "Test failure",
        "traceback": "Traceback...\nRuntimeError: Test failure\n",
    }

    serializer = JsonSerializer()
    serialized_ev = serializer.serialize(ev)
    deserialized_ev = serializer.deserialize(serialized_ev)

    assert type(deserialized_ev).__name__ == type(ev).__name__
    deserialized_ev = cast(WorkflowFailedEvent, deserialized_ev)
    assert ev.step_name == deserialized_ev.step_name
    assert ev.exception_type == deserialized_ev.exception_type
    assert ev.exception_message == deserialized_ev.exception_message
    assert ev.traceback == deserialized_ev.traceback


def test_workflow_failed_event_repr() -> None:
    """Test WorkflowFailedEvent string representation."""
    ev = WorkflowFailedEvent(
        step_name="my_step",
        exception_type="builtins.ValueError",
        exception_message="error msg",
        traceback="...",
    )
    rep = repr(ev)
    assert "WorkflowFailedEvent" in rep
    assert "step_name='my_step'" in rep
    assert "exception_type='builtins.ValueError'" in rep
    assert "exception_message='error msg'" in rep


def test_workflow_failed_event_with_nested_exception_type() -> None:
    """Test WorkflowFailedEvent with a qualified exception type name."""
    ev = WorkflowFailedEvent(
        step_name="api_step",
        exception_type="http.client.HTTPException",
        exception_message="Connection refused",
        traceback="Traceback...",
    )
    assert ev.exception_type == "http.client.HTTPException"
