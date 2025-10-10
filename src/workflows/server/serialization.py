# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

from typing import Any, TypedDict, Literal
from enum import Enum

from workflows.context.serializers import JsonSerializer
from workflows.events import (
    Event,
)


class EventOrigin(str, Enum):
    BUILTIN = "builtin"
    USER = "user"


class EventEnvelope(TypedDict):
    # Back-compat fields from serializer
    __is_pydantic: bool
    value: Any

    # deprecated, use mro instead
    qualified_name: str

    # New metadata
    mro: list[str]
    origin: Literal["builtin", "user"]


def _qualified_name_of_class(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def _mro_qualified_names(cls: type) -> list[str]:
    names: list[str] = []
    for c in cls.mro():
        try:
            names.append(_qualified_name_of_class(c))
        except Exception:
            # Best effort; skip if class is unusual
            continue
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
        __is_pydantic=base.get("__is_pydantic", False),
        mro=_mro_qualified_names(type(event)),
        origin=(
            EventOrigin.BUILTIN.value
            if event.__class__.__module__ == "workflows.events"
            else EventOrigin.USER.value
        ),
    )
    return envelope
