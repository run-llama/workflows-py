# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Annotated types for Pydantic serialization of tricky tick/result fields.

Provides custom serializers/validators for:
- Events (polymorphic Pydantic models)
- Exceptions (not natively serializable)
- Event types (type[Event] as qualified name strings)
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import PlainSerializer, PlainValidator
from workflows.context.serializers import JsonSerializer
from workflows.context.utils import (
    import_module_from_qualified_name,
)
from workflows.events import Event

_json_serializer = JsonSerializer()


def _serialize_event(event: Event) -> Any:
    return _json_serializer.serialize_value(event)


def _deserialize_event(data: Any) -> Event:
    return _json_serializer.deserialize_value(data)


SerializableEvent = Annotated[
    Event,
    PlainSerializer(_serialize_event, return_type=Any),
    PlainValidator(_deserialize_event),
]


def _serialize_optional_event(event: Event | None) -> Any:
    if event is None:
        return None
    return _json_serializer.serialize_value(event)


def _deserialize_optional_event(data: Any) -> Event | None:
    if data is None:
        return None
    return _json_serializer.deserialize_value(data)


SerializableOptionalEvent = Annotated[
    Optional[Event],
    PlainSerializer(_serialize_optional_event, return_type=Any),
    PlainValidator(_deserialize_optional_event),
]


def _serialize_exception(exc: Exception) -> dict[str, Any]:
    exc_type = type(exc)
    qualified_name = f"{exc_type.__module__}.{exc_type.__qualname__}"
    return {
        "exception_type": qualified_name,
        "exception_message": str(exc),
    }


def _deserialize_exception(data: Any) -> Exception:
    if isinstance(data, Exception):
        return data
    exc_message = data["exception_message"]
    try:
        exc_cls = import_module_from_qualified_name(data["exception_type"])
        return exc_cls(exc_message)
    except (ImportError, AttributeError, ValueError):
        return Exception(exc_message)


SerializableException = Annotated[
    Exception,
    PlainSerializer(_serialize_exception, return_type=dict[str, Any]),
    PlainValidator(_deserialize_exception),
]


def _serialize_event_type(event_type: type[Event]) -> str:
    return f"{event_type.__module__}.{event_type.__qualname__}"


def _deserialize_event_type(data: Any) -> type[Event]:
    if isinstance(data, type):
        return data
    return import_module_from_qualified_name(data)


SerializableEventType = Annotated[
    type[Event],
    PlainSerializer(_serialize_event_type, return_type=str),
    PlainValidator(_deserialize_event_type),
]
