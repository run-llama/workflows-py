# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

from typing import Any, TypeVar

try:
    from typing import Union
except ImportError:
    from typing_extensions import Union

from .events import StopEvent

StopEventT = TypeVar("StopEventT", bound=StopEvent)
# TODO: When releasing 1.0, remove support for Any
# and enforce usage of StopEventT
RunResultT = Union[StopEventT, Any]
