# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
EventInterceptorDecorator: blocks write_to_event_stream from reaching the
inner runtime while allowing all other operations (ticks, send/recv, close)
to pass through normally.

Used when the ServerRuntimeDecorator already writes events to the workflow
store and forwarding them to the inner runtime (e.g. DBOS) would cause
duplicate writes.
"""

from __future__ import annotations

import logging

from typing_extensions import override
from workflows.events import Event
from workflows.runtime.types.plugin import InternalRunAdapter
from workflows.workflow import Workflow

from .runtime_decorators import (
    BaseInternalRunAdapterDecorator,
    BaseRuntimeDecorator,
)

logger = logging.getLogger(__name__)


class _InterceptorInternalAdapter(BaseInternalRunAdapterDecorator):
    """Internal adapter that swallows write_to_event_stream calls."""

    @override
    async def write_to_event_stream(self, event: Event) -> None:
        # No-op: do NOT forward to inner adapter.
        pass


class EventInterceptorDecorator(BaseRuntimeDecorator):
    """Runtime decorator that prevents published events from reaching the
    inner runtime's event stream.

    All other methods (on_tick, send_event, wait_receive, close, etc.)
    pass through to the inner runtime normally.
    """

    @override
    def get_internal_adapter(self, workflow: Workflow) -> InternalRunAdapter:
        inner = self._decorated.get_internal_adapter(workflow)
        return _InterceptorInternalAdapter(inner)
