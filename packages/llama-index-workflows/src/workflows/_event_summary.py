# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from typing import Any

from workflows.events import Event, StopEvent


def _summarize_value(value: Any, max_val_length: int = 50) -> str:
    if isinstance(value, str):
        if len(value) > max_val_length:
            value = value[:max_val_length] + "..."
        return repr(value)
    if isinstance(value, list):
        if len(value) > 3:
            return f"[{len(value)} items]"
        return repr(value)
    if isinstance(value, dict):
        if len(value) > 3:
            return f"{{{len(value)} keys}}"
        return repr(value)
    return repr(value)


def summarize_event(event: Event, max_length: int = 200) -> str:
    try:
        parts: list[str] = []

        # Special-case StopEvent to include result
        if isinstance(event, StopEvent):
            result = event._result  # noqa: SLF001
            if result is not None:
                parts.append(f"result={_summarize_value(result)}")

        # Declared pydantic fields
        for field_name in event.__class__.model_fields:
            val = getattr(event, field_name, None)
            parts.append(f"{field_name}={_summarize_value(val)}")

        # Dynamic _data entries
        data = event._data  # noqa: SLF001
        for key, val in data.items():
            parts.append(f"{key}={_summarize_value(val)}")

        class_name = event.__class__.__name__
        inner = ", ".join(parts)
        result_str = f"{class_name}({inner})"

        if len(result_str) > max_length:
            result_str = result_str[: max_length - 3] + "..."

        return result_str
    except Exception:
        try:
            return repr(event)
        except Exception:
            return event.__class__.__name__
