# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from typing import Any, TypeVar

try:
    from typing import Union
except ImportError:  # pragma: no cover
    from typing_extensions import Union

from .events import StopEvent

StopEventT = TypeVar("StopEventT", bound=StopEvent)
# TODO: When releasing 1.0, remove support for Any
# and enforce usage of StopEventT
RunResultT = Union[StopEventT, Any]
"""
Type aliases for workflow results.

- `StopEventT`: Generic bound to [StopEvent][workflows.events.StopEvent]
- `RunResultT`: Result type returned by a workflow run. Today it allows either
  a `StopEventT` subclass or `Any` for backward compatibility; future versions
  may restrict this to `StopEventT` only.
"""
