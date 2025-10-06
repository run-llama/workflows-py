from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import json
from typing import Any

from workflows.context.context_types import SerializedContext
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.events import Event


@dataclass
class WorkflowBrokerState:
    """Asyncio-native, serializable broker state for a single workflow run.

    Owned and mutated by ``WorkflowBroker``; excludes the user state store
    (owned by ``Context``). See ``SerializedContext`` for the wire format.
    """

    # True while a run is active; cleared on completion/timeout/cancel
    is_running: bool = False

    # Event stream for observers; broker writes lifecycle/system events here
    # A None item may be used as a sentinel for end-of-stream
    streaming_queue: asyncio.Queue[Event | None] = field(
        default_factory=lambda: asyncio.Queue()
    )

    # Per-step and waiter inbound queues: step name / waiter ID -> asyncio.Queue
    # Workers consume from these to drive step execution
    queues: dict[str, asyncio.Queue[Event]] = field(default_factory=dict)

    # Buffers used by collect_events: buffer_id -> fully.qualified.Event -> [Event]
    # Enables waiting for specific combinations of events
    event_buffers: dict[str, dict[str, list[Event]]] = field(default_factory=dict)

    # Step name -> list of input events currently being processed by that step
    # Also used to seed queues on deserialization so in-flight work can resume
    in_progress: dict[str, list[Event]] = field(default_factory=dict)

    # (step_name, input_event_class_name) recorded when a step outputs an event
    # Consumed by drawing/visualization utilities to reconstruct edges
    accepted_events: list[tuple[str, str]] = field(default_factory=list)

    # Append-only log of all events dispatched by the broker in order
    broker_log: list[Event] = field(default_factory=list)

    # Active waiter IDs to suppress duplicate waiter events (e.g., UI prompts)
    waiting_ids: set[str] = field(default_factory=set)

    def to_serialized(self, serializer: BaseSerializer) -> SerializedContext:
        return SerializedContext(
            state={},  # blank to be filled in separately by the state manager
            streaming_queue=_serialize_queue(self.streaming_queue, serializer),
            queues={k: _serialize_queue(v, serializer) for k, v in self.queues.items()},
            event_buffers={
                k: {
                    inner_k: [serializer.serialize(ev) for ev in inner_v]
                    for inner_k, inner_v in v.items()
                }
                for k, v in self.event_buffers.items()
            },
            in_progress={
                k: [serializer.serialize(ev) for ev in v]
                for k, v in self.in_progress.items()
            },
            accepted_events=self.accepted_events,
            broker_log=[serializer.serialize(ev) for ev in self.broker_log],
            is_running=self.is_running,
            waiting_ids=list(self.waiting_ids),
        )

    @classmethod
    def from_serialized(
        cls, data: SerializedContext, serializer: BaseSerializer | None = None
    ) -> WorkflowBrokerState:
        serializer = serializer or JsonSerializer()

        streaming_queue = _deserialize_queue(data.streaming_queue, serializer)

        event_buffers: dict[str, dict[str, list[Event]]] = {}
        for buffer_id, type_events_map in data.event_buffers.items():
            event_buffers[buffer_id] = {}
            for event_type, events_list in type_events_map.items():
                event_buffers[buffer_id][event_type] = [
                    serializer.deserialize(ev) for ev in events_list
                ]

        accepted_events = list(data.accepted_events)
        broker_log = [serializer.deserialize(ev) for ev in data.broker_log]

        # load back up whatever was in the queue as well as the events whose steps
        # were in progress when the serialization of the Context took place
        queues = {
            k: _deserialize_queue(
                v, serializer, prefix_queue_objs=data.in_progress.get(k, [])
            )
            for k, v in data.queues.items()
        }
        in_progress = defaultdict(list)
        for k, v in data.in_progress.items():
            in_progress[k] = [serializer.deserialize(ev) for ev in v]

        # restore waiting ids for hitl
        waiting_ids = set(data.waiting_ids) if data.waiting_ids else set()

        is_running = data.is_running
        return cls(
            is_running=is_running,
            streaming_queue=streaming_queue,
            queues=queues,
            event_buffers=event_buffers,
            in_progress=in_progress,
            accepted_events=accepted_events,
            broker_log=broker_log,
            waiting_ids=waiting_ids,
        )


def _deserialize_queue(
    queue_str: str,
    serializer: BaseSerializer,
    prefix_queue_objs: list[Any] = [],
) -> asyncio.Queue:
    queue_objs = json.loads(queue_str)
    queue_objs = prefix_queue_objs + queue_objs
    queue: asyncio.Queue = asyncio.Queue()
    for obj in queue_objs:
        event_obj = serializer.deserialize(obj)
        queue.put_nowait(event_obj)
    return queue


def _serialize_queue(queue: asyncio.Queue, serializer: BaseSerializer) -> str:
    # get the queue items that are defined, without waiting or draining
    queue_items = list(queue._queue)  # type: ignore
    queue_objs = [serializer.serialize(obj) for obj in queue_items]
    return json.dumps(queue_objs)
