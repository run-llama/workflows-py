# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

from typing import Any, TypedDict

from workflows.context.serializers import JsonSerializer
from workflows.events import (
    Event,
)


class EventEnvelope(TypedDict):
    value: Any

    # deprecated, use type instead
    qualified_name: str

    # New metadata
    type: str
    types: list[str] | None


def _mro_names(cls: type) -> list[str] | None:
    built_in_mros = ["Event", "DictLikeModel", "BaseModel", "object"]
    names: list[str] = []
    # Skip the class itself by starting from the second MRO entry
    for c in cls.mro()[1:]:
        if c.__name__ in built_in_mros:
            break
        try:
            names.append(c.__name__)
        except Exception:
            continue
    if not names:
        return None
    return names


def build_event_envelope(event: Event, serializer: JsonSerializer) -> EventEnvelope:
    """
    Build a backward-compatible envelope for an Event, preserving existing
    fields (e.g., qualified_name, value) while adding metadata useful for
    type-safe clients.

    """
    # Start with the existing JSON-serializable structure
    base = serializer.serialize_value(event)
    if not isinstance(base, dict):
        raise TypeError(f"Expected dict, got {type(base)}")

    envelope = EventEnvelope(
        value=base.get("value", {}),
        qualified_name=base.get("qualified_name", ""),
        types=_mro_names(type(event)),
        type=type(event).__name__,
    )
    return envelope
