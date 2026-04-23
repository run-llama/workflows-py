# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""DEPRECATED — back-compat serialization shims.

Everything in this module exists only to keep older tick/result models
serializable. Do not add new types here and do not reach for these shims
from new code — use ``JsonSerializer`` / ``EventEnvelope`` directly.
All aliases below will be removed once the remaining call sites are
migrated.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import PlainSerializer, PlainValidator
from workflows.context.utils import (
    import_module_from_qualified_name,
)

# DEPRECATED re-exports — kept only so existing imports from this module
# keep working. New code should import these directly from ``workflows.events``.
from workflows.events import (
    Event,
    SerializableEvent,
    SerializableOptionalEvent,
)

__all__ = [
    # DEPRECATED: all of the below are back-compat shims scheduled for removal.
    "SerializableEvent",
    "SerializableOptionalEvent",
    "SerializableException",
    "SerializableEventType",
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


# DEPRECATED: back-compat shim, scheduled for removal. Do not use in new code.
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


# DEPRECATED: back-compat shim, scheduled for removal. Do not use in new code.
SerializableEventType = Annotated[
    type[Event],
    PlainSerializer(_serialize_event_type, return_type=str),
    PlainValidator(_deserialize_event_type),
]
