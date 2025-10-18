# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from workflows.events import (
    Event,
    StopEvent,
    StepStateChanged,
    StepState,
)
from workflows.protocol.serializable_events import EventEnvelopeWithMetadata


def test_envelope_user_defined_event() -> None:
    class MyEvent(Event):
        x: int

    ev = MyEvent(x=1)
    env = EventEnvelopeWithMetadata.from_event(ev).model_dump()

    assert isinstance(env.get("value", {}), dict)
    types = env.get("types")
    assert types is None
    # User-defined event
    assert env.get("type", "") == "MyEvent"


def test_envelope_builtin_stop_event() -> None:
    ev = StopEvent()
    env = EventEnvelopeWithMetadata.from_event(ev).model_dump()

    assert isinstance(env.get("value", {}), dict)
    types = env.get("types")
    assert types is None
    assert env.get("type", "") == "StopEvent"


def test_envelope_stop_event_subclass() -> None:
    class MyStop(StopEvent):
        pass

    ev = MyStop()
    env = EventEnvelopeWithMetadata.from_event(ev).model_dump()

    assert isinstance(env.get("value", {}), dict)
    # Subclass is user-defined
    assert env.get("type", "") == "MyStop"
    # Must include base StopEvent in MRO
    types = env.get("types")
    assert types is not None
    assert "StopEvent" in types


def test_envelope_internal_event() -> None:
    ev = StepStateChanged(
        name="s",
        step_state=StepState.PREPARING,
        worker_id="w1",
        input_event_name="X",
    )
    env = EventEnvelopeWithMetadata.from_event(ev).model_dump()

    assert isinstance(env.get("value", {}), dict)
    assert env.get("type", "") == "StepStateChanged"
    # Internal event types contains specific class and base Event
    types = env.get("types")
    assert types is not None
    assert "InternalDispatchEvent" in types
