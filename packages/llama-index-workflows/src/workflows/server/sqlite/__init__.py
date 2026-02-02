# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Re-export sqlite components from the optional llama-agents-server package."""

try:
    from llama_agents.server.sqlite.sqlite_workflow_store import SqliteWorkflowStore
except ImportError as e:
    raise ImportError(
        "workflows.server.sqlite requires the 'server' extra. "
        "Install with: pip install 'llama-index-workflows[server]'"
    ) from e

__all__ = ["SqliteWorkflowStore"]
