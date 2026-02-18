# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Re-export client components from the optional llama-agents-client package."""

import warnings

warnings.warn(
    "Importing from 'workflows.client' is deprecated. "
    "Install 'llama-agents-client' and use "
    "'from llama_agents.client import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from llama_agents.client import WorkflowClient
except ImportError as e:
    raise ImportError(
        "workflows.client requires the 'client' extra. "
        "Install with: pip install 'llama-index-workflows[client]'"
    ) from e

__all__ = ["WorkflowClient"]
