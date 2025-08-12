from collections import defaultdict

from workflows.events import Event
from typing import Sequence, Type, Dict, Any, Optional


class EventsQueue:
    def __init__(self, queue: Optional[Dict[Any, int]] = None) -> None:
        self._queue = queue or defaultdict(int)

    def put(self, events: Sequence[Event]) -> None:
        for event in events:
            self._queue[str(type(event))] += 1

    def get(self, event_type: Type[Event]) -> int:
        return self._queue[str(event_type)]
