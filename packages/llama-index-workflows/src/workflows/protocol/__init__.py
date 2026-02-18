# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Re-export protocol types from the optional llama-agents-client package."""

import warnings

warnings.warn(
    "Importing from 'workflows.protocol' is deprecated. "
    "Install 'llama-agents-client' and use "
    "'from llama_agents.client.protocol import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from llama_agents.client.protocol import (
        CancelHandlerResponse,
        HandlerData,
        HandlersListResponse,
        HealthResponse,
        SendEventResponse,
        Status,
        WorkflowEventsListResponse,
        WorkflowGraphResponse,
        WorkflowSchemaResponse,
        WorkflowsListResponse,
        is_status_completed,
    )
except ImportError as e:
    raise ImportError(
        "workflows.protocol requires the 'client' extra. "
        "Install with: pip install 'llama-index-workflows[client]'"
    ) from e

__all__ = [
    "Status",
    "is_status_completed",
    "HandlerData",
    "HandlersListResponse",
    "HealthResponse",
    "WorkflowsListResponse",
    "SendEventResponse",
    "CancelHandlerResponse",
    "WorkflowSchemaResponse",
    "WorkflowEventsListResponse",
    "WorkflowGraphResponse",
]
