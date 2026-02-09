# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import json
import time

from pydantic import TypeAdapter
from workflows.events import Event, StartEvent, StopEvent
from workflows.runtime.types.results import (
    AddCollectedEvent,
    AddWaiter,
    DeleteCollectedEvent,
    DeleteWaiter,
    StepWorkerFailed,
    StepWorkerResult,
)
from workflows.runtime.types.serialization_helpers import (
    _deserialize_event,
    _deserialize_event_type,
    _deserialize_exception,
    _serialize_event,
    _serialize_event_type,
    _serialize_exception,
)
from workflows.runtime.types.ticks import (
    TickAddEvent,
    TickCancelRun,
    TickPublishEvent,
    TickStepResult,
    TickTimeout,
    WorkflowTick,
)


class MyEvent(Event):
    value: str = "hello"


# -- Serialization helper roundtrip tests --


def test_event_roundtrip() -> None:
    event = MyEvent(value="world")
    serialized = _serialize_event(event)
    result = _deserialize_event(serialized)
    assert isinstance(result, MyEvent)
    assert result.value == "world"


def test_exception_roundtrip() -> None:
    exc = ValueError("something went wrong")
    serialized = _serialize_exception(exc)
    result = _deserialize_exception(serialized)
    assert isinstance(result, ValueError)
    assert str(result) == "something went wrong"


def test_exception_roundtrip_unimportable() -> None:
    CustomError = type("CustomError", (Exception,), {})
    exc = CustomError("oops")
    serialized = _serialize_exception(exc)
    result = _deserialize_exception(serialized)
    assert type(result) is Exception
    assert str(result) == "oops"


def test_event_type_roundtrip() -> None:
    serialized = _serialize_event_type(MyEvent)
    result = _deserialize_event_type(serialized)
    assert result is MyEvent


# -- Tick roundtrip tests --


def test_tick_add_event_round_trip() -> None:
    tick = TickAddEvent(
        event=StartEvent(),
        step_name="my_step",
        attempts=3,
        first_attempt_at=1234567890.0,
    )
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickAddEvent.model_validate(roundtripped)

    assert isinstance(result, TickAddEvent)
    assert isinstance(result.event, StartEvent)
    assert result.step_name == "my_step"
    assert result.attempts == 3
    assert result.first_attempt_at == 1234567890.0


def test_tick_publish_event_round_trip() -> None:
    tick = TickPublishEvent(event=MyEvent(value="world"))
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickPublishEvent.model_validate(roundtripped)

    assert isinstance(result, TickPublishEvent)
    assert isinstance(result.event, MyEvent)
    assert result.event.value == "world"


def test_tick_cancel_run_round_trip() -> None:
    tick = TickCancelRun()
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickCancelRun.model_validate(roundtripped)

    assert isinstance(result, TickCancelRun)


def test_tick_timeout_round_trip() -> None:
    tick = TickTimeout(timeout=30.5)
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickTimeout.model_validate(roundtripped)

    assert isinstance(result, TickTimeout)
    assert result.timeout == 30.5


def test_tick_step_result_with_event_result() -> None:
    event = MyEvent(value="trigger")
    worker_result = StepWorkerResult(result=StopEvent(result="done"))
    tick = TickStepResult(
        step_name="process",
        worker_id=42,
        event=event,
        result=[worker_result],
    )
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickStepResult.model_validate(roundtripped)

    assert isinstance(result, TickStepResult)
    assert result.step_name == "process"
    assert result.worker_id == 42
    assert isinstance(result.event, MyEvent)
    assert result.event.value == "trigger"
    assert len(result.result) == 1
    r = result.result[0]
    assert isinstance(r, StepWorkerResult)
    assert isinstance(r.result, StopEvent)
    assert r.result.result == "done"


def test_tick_step_result_with_none_result() -> None:
    tick = TickStepResult(
        step_name="process",
        worker_id=1,
        event=StartEvent(),
        result=[StepWorkerResult(result=None)],
    )
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickStepResult.model_validate(roundtripped)

    assert isinstance(result, TickStepResult)
    r = result.result[0]
    assert isinstance(r, StepWorkerResult)
    assert r.result is None


def test_tick_step_result_with_failed_value_error() -> None:
    failed_at = time.time()
    tick = TickStepResult(
        step_name="broken_step",
        worker_id=7,
        event=StartEvent(),
        result=[
            StepWorkerFailed(
                exception=ValueError("something went wrong"), failed_at=failed_at
            )
        ],
    )
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickStepResult.model_validate(roundtripped)

    assert isinstance(result, TickStepResult)
    r = result.result[0]
    assert isinstance(r, StepWorkerFailed)
    assert isinstance(r.exception, ValueError)
    assert str(r.exception) == "something went wrong"
    assert r.failed_at == failed_at


def test_tick_step_result_with_failed_unimportable_exception() -> None:
    CustomError = type("CustomError", (Exception,), {})
    failed_at = time.time()
    tick = TickStepResult(
        step_name="broken_step",
        worker_id=8,
        event=StartEvent(),
        result=[StepWorkerFailed(exception=CustomError("oops"), failed_at=failed_at)],
    )
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickStepResult.model_validate(roundtripped)

    assert isinstance(result, TickStepResult)
    r = result.result[0]
    assert isinstance(r, StepWorkerFailed)
    assert type(r.exception) is Exception
    assert str(r.exception) == "oops"
    assert r.failed_at == failed_at


def test_tick_step_result_with_add_collected_event() -> None:
    tick = TickStepResult(
        step_name="collector",
        worker_id=2,
        event=StartEvent(),
        result=[AddCollectedEvent(event_id="evt-1", event=MyEvent(value="collected"))],
    )
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickStepResult.model_validate(roundtripped)

    assert isinstance(result, TickStepResult)
    r = result.result[0]
    assert isinstance(r, AddCollectedEvent)
    assert r.event_id == "evt-1"
    assert isinstance(r.event, MyEvent)
    assert r.event.value == "collected"


def test_tick_step_result_with_delete_collected_event() -> None:
    tick = TickStepResult(
        step_name="collector",
        worker_id=3,
        event=StartEvent(),
        result=[DeleteCollectedEvent(event_id="evt-2")],
    )
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickStepResult.model_validate(roundtripped)

    assert isinstance(result, TickStepResult)
    r = result.result[0]
    assert isinstance(r, DeleteCollectedEvent)
    assert r.event_id == "evt-2"


def test_tick_step_result_with_add_waiter() -> None:
    tick = TickStepResult(
        step_name="waiter_step",
        worker_id=4,
        event=StartEvent(),
        result=[
            AddWaiter(
                waiter_id="w-1",
                waiter_event=MyEvent(value="waiting"),
                requirements={"key": "value"},
                timeout=60.0,
                event_type=MyEvent,
            )
        ],
    )
    serialized = tick.model_dump(mode="json")

    # Verify the serialized form captures has_requirements correctly
    waiter_data = serialized["result"][0]
    assert waiter_data["has_requirements"] is True
    assert waiter_data["requirements"] == {}

    roundtripped = json.loads(json.dumps(serialized))
    result = TickStepResult.model_validate(roundtripped)

    assert isinstance(result, TickStepResult)
    r = result.result[0]
    assert isinstance(r, AddWaiter)
    assert r.waiter_id == "w-1"
    assert isinstance(r.waiter_event, MyEvent)
    assert r.waiter_event.value == "waiting"
    # Requirements are always {} after deserialization
    assert r.requirements == {}
    assert r.timeout == 60.0
    assert r.event_type is MyEvent


def test_tick_step_result_with_delete_waiter() -> None:
    tick = TickStepResult(
        step_name="cleanup",
        worker_id=5,
        event=StartEvent(),
        result=[DeleteWaiter(waiter_id="w-2")],
    )
    serialized = tick.model_dump(mode="json")
    roundtripped = json.loads(json.dumps(serialized))
    result = TickStepResult.model_validate(roundtripped)

    assert isinstance(result, TickStepResult)
    r = result.result[0]
    assert isinstance(r, DeleteWaiter)
    assert r.waiter_id == "w-2"


# -- WorkflowTick discriminated union tests --


def test_workflow_tick_discriminated_union_roundtrip() -> None:
    """Verify that WorkflowTick TypeAdapter can roundtrip all tick types."""
    adapter = TypeAdapter(WorkflowTick)

    ticks = [
        TickAddEvent(event=StartEvent(), step_name="s"),
        TickPublishEvent(event=MyEvent(value="x")),
        TickCancelRun(),
        TickTimeout(timeout=10.0),
        TickStepResult(
            step_name="s",
            worker_id=0,
            event=StartEvent(),
            result=[StepWorkerResult(result=None)],
        ),
    ]
    for tick in ticks:
        dumped = adapter.dump_python(tick, mode="json")
        roundtripped = json.loads(json.dumps(dumped))
        restored = adapter.validate_python(roundtripped)
        assert type(restored) is type(tick)
