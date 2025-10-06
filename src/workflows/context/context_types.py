from typing import Any
from pydantic import BaseModel, Field


class SerializedContext(BaseModel):
    """
    Internal shape of serialized context. Will be serialized to a json dict before returning. External applications should treat this as opaque.
    """

    # Serialized state store payload produced by InMemoryStateStore.to_dict(serializer).
    # Shape:
    #   {
    #     "state_type": str,            # class name of the model (e.g. "DictState" or custom model)
    #     "state_module": str,          # module path of the model
    #     "state_data": ...             # see below
    #   }
    # For DictState: state_data = {"_data": {key: serialized_value_str}}, where each value is the
    # serializer-encoded string (e.g. JSON string from JsonSerializer.serialize).
    # For typed Pydantic models: state_data is a serializer-encoded string containing JSON for a dict with
    # discriminator fields (e.g. {"__is_pydantic": true, "value": <model_dump>, "qualified_name": <module.Class>}).
    state: dict[str, Any] = Field(default_factory=dict)

    # Streaming queue contents used by the event stream. This is a JSON string representing a list
    # of serializer-encoded events (each element is a string as returned by BaseSerializer.serialize).
    # Example: '["<serialized_event>", "<serialized_event>"]'.
    streaming_queue: str = Field(default="[]")

    # Per-step (and waiter) inbound event queues. Maps queue name -> JSON string representing a list
    # of serializer-encoded events (same format as streaming_queue).
    queues: dict[str, str] = Field(default_factory=dict)

    # Buffered events used by Context.collect_events. Maps buffer_id -> { fully.qualified.EventType: [serialized_event_str, ...] }.
    # Each inner list element is a serializer-encoded string for an Event.
    event_buffers: dict[str, dict[str, list[str]]] = Field(default_factory=dict)

    # Events that were in-flight for each step at serialization time. Maps step_name -> [serialized_event_str, ...].
    in_progress: dict[str, list[str]] = Field(default_factory=dict)

    # Pairs recorded when a step produced an output event: (step_name, input_event_class_name).
    # Note: stored as Python tuples here; if JSON-encoded externally they become 2-element lists.
    accepted_events: list[tuple[str, str]] = Field(default_factory=list)

    # Broker log of all dispatched events in order, as serializer-encoded strings.
    broker_log: list[str] = Field(default_factory=list)

    # Whether the workflow was running when serialized.
    is_running: bool = Field(default=False)

    # IDs currently waiting in wait_for_event to suppress duplicate waiter events. These IDs may appear
    # as keys in `queues` (they are used as queue names for waiter-specific queues).
    waiting_ids: list[str] = Field(default_factory=list)
