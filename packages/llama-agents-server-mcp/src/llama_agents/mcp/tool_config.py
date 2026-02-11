# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass
class MCPToolConfig:
    """Configuration for exposing a workflow as an MCP tool."""

    mode: Literal["sync", "async"] = "sync"
    description: str | None = None
    tool_name: str | None = None
    render_result: Callable[[Any], str] | None = None
    tags: set[str] | None = field(default=None)
