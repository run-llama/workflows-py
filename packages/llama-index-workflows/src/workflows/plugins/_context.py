# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Context-scoped plugin access."""

from __future__ import annotations

from workflows.runtime.types.plugin import Plugin


def get_current_plugin() -> Plugin:
    """
    Get the current plugin from context or fall back to basic_runtime.

    Returns the context-scoped plugin if set, otherwise returns basic_runtime.
    """
    # Inline imports to avoid circular dependency (basic -> plugin -> workflow)
    from workflows.plugins.basic import basic_runtime
    from workflows.runtime.types.plugin import _current_plugin

    plugin = _current_plugin.get()
    return plugin if plugin is not None else basic_runtime
