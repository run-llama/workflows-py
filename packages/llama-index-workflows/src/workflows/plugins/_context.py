# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Context-scoped runtime access."""

from __future__ import annotations

from workflows.runtime.types.plugin import Runtime, _current_runtime


def get_current_runtime() -> Runtime:
    """
    Get the current runtime from context or fall back to basic_runtime.

    Returns the context-scoped runtime if set, otherwise returns basic_runtime.
    """
    # Inline import to avoid circular dependency (basic -> runtime -> workflow)
    from workflows.plugins.basic import basic_runtime

    runtime = _current_runtime.get()
    return runtime if runtime is not None else basic_runtime
