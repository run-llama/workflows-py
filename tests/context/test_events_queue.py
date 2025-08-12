import pytest
from typing import Sequence

from workflows.events import Event
from workflows.context.events_queue import EventsQueue
from workflows.context.serializers import JsonSerializer


class SomeEvent(Event):
    data: str


class SomeOtherEvent(Event):
    other_data: str


@pytest.fixture()
def events() -> Sequence[Event]:
    return [
        SomeEvent(data="hello"),
        SomeEvent(data="ciao"),
        SomeOtherEvent(other_data="other hello"),
        SomeOtherEvent(other_data="other ciao"),
        SomeOtherEvent(other_data="extra data point"),
    ]


def test_basic_operations(events: Sequence[Event]) -> None:
    queue = EventsQueue()
    queue.put(events)
    assert queue.get(SomeEvent) == 2
    assert queue.get(SomeOtherEvent) == 3


def test_serialization_deserialization(events: Sequence[Event]) -> None:
    # tests serialization/deserialization patterns as done in context
    queue = EventsQueue()
    queue.put(events)
    try:
        a = JsonSerializer().serialize(queue._queue)
        success = True
    except Exception:
        a = ""
        success = False
    assert success
    try:
        b = JsonSerializer().deserialize(a)
        success = True
    except Exception:
        b = {}
        success = False
    assert success
    queue1 = EventsQueue(queue=b)
    queue1.put(events)
    assert queue1.get(SomeEvent) == 4
    assert queue1.get(SomeOtherEvent) == 6
