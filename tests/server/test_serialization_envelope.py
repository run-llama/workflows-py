# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from workflows.context.serializers import JsonSerializer
from workflows.events import (
    Event,
    StopEvent,
    StepStateChanged,
    StepState,
)
from workflows.server.serialization import build_event_envelope


def test_envelope_user_defined_event() -> None:
    class MyEvent(Event):
        x: int

    ev = MyEvent(x=1)
    env = build_event_envelope(ev, JsonSerializer())

    # Back-compat and metadata (use safe access for TypedDict optional keys)
    assert env.get("__is_pydantic", False) is True
    assert isinstance(env.get("value", {}), dict)
    mro = env.get("mro", [])
    assert isinstance(mro, list)
    # User-defined event
    assert env.get("origin", "") == "user"
    # MRO should include our class and base Event
    assert "tests.server.test_serialization_envelope.MyEvent" in mro
    assert "workflows.events.Event" in mro


def test_envelope_builtin_stop_event() -> None:
    ev = StopEvent()
    env = build_event_envelope(ev, JsonSerializer())

    assert env.get("__is_pydantic", False) is True
    assert isinstance(env.get("value", {}), dict)
    assert env.get("origin", "") == "builtin"
    mro = env.get("mro", [])
    assert "workflows.events.StopEvent" in mro


def test_envelope_stop_event_subclass() -> None:
    class MyStop(StopEvent):
        pass

    ev = MyStop()
    env = build_event_envelope(ev, JsonSerializer())

    assert env.get("__is_pydantic", False) is True
    assert isinstance(env.get("value", {}), dict)
    # Subclass is user-defined
    assert env.get("origin", "") == "user"
    # Must include base StopEvent in MRO
    mro = env.get("mro", [])
    assert "workflows.events.StopEvent" in mro
    assert "tests.server.test_serialization_envelope.MyStop" in mro


def test_envelope_internal_event() -> None:
    ev = StepStateChanged(
        name="s",
        step_state=StepState.PREPARING,
        worker_id="w1",
        input_event_name="X",
    )
    env = build_event_envelope(ev, JsonSerializer())

    assert env.get("__is_pydantic", False) is True
    assert isinstance(env.get("value", {}), dict)
    assert env.get("origin", "") == "builtin"
    # Internal event MRO contains specific class and base Event
    mro = env.get("mro", [])
    assert "workflows.events.StepStateChanged" in mro
    assert "workflows.events.Event" in mro
