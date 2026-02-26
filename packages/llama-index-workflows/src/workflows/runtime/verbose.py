# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
Verbose runtime decorator that prints step activity by observing StepStateChanged events.

Output destination is auto-detected: if the ``"workflows.verbose"`` logger is
configured to emit DEBUG or INFO messages, those levels are used (in that
priority order).  Otherwise output falls back to :func:`print`.
"""

from __future__ import annotations

import logging
from typing import Callable

from workflows.events import Event, StepState, StepStateChanged
from workflows.runtime.runtime_decorators import (
    BaseInternalRunAdapterDecorator,
    BaseRuntimeDecorator,
)
from workflows.runtime.types.plugin import InternalRunAdapter, Runtime
from workflows.workflow import Workflow

verbose_logger = logging.getLogger("workflows.verbose")


def _resolve_output() -> Callable[[str], None]:
    """Pick the best output sink based on current logging configuration."""
    if verbose_logger.isEnabledFor(logging.DEBUG):
        return verbose_logger.debug
    if verbose_logger.isEnabledFor(logging.INFO):
        return verbose_logger.info
    return print


class _VerboseInternalRunAdapter(BaseInternalRunAdapterDecorator):
    """Intercepts write_to_event_stream to print/log step activity."""

    def __init__(
        self,
        decorated: InternalRunAdapter,
        output: Callable[[str], None],
    ) -> None:
        super().__init__(decorated)
        self._output = output

    async def write_to_event_stream(self, event: Event) -> None:
        if isinstance(event, StepStateChanged):
            if event.step_state == StepState.RUNNING:
                self._output(f"Running step {event.name}")
            elif event.step_state == StepState.NOT_RUNNING:
                if event.output_event_name:
                    self._output(
                        f"Step {event.name} produced event {event.output_event_name}"
                    )
                else:
                    self._output(f"Step {event.name} produced no event")
        await super().write_to_event_stream(event)


class VerboseDecorator(BaseRuntimeDecorator):
    """Runtime decorator that prints step starts and completions.

    Output destination is auto-detected at construction time based on the
    ``"workflows.verbose"`` logger's effective level.  If DEBUG or INFO
    messages would be emitted, the logger is used; otherwise falls back to
    :func:`print`.
    """

    def __init__(self, decorated: Runtime) -> None:
        super().__init__(decorated)
        self._output = _resolve_output()

    def get_internal_adapter(self, workflow: Workflow) -> InternalRunAdapter:
        inner = self._decorated.get_internal_adapter(workflow)
        return _VerboseInternalRunAdapter(inner, self._output)
