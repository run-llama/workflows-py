# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
Server runtime decorator: adapter wrapping, event recording, and handler
persistence. Service-level orchestration lives in _service.py.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

from llama_agents.client.protocol.serializable_events import (
    EventEnvelopeWithMetadata,
)
from llama_agents.server._runtime.runtime_decorators import (
    BaseInternalRunAdapterDecorator,
)
from typing_extensions import override
from workflows.events import (
    Event,
    StopEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
    WorkflowTimedOutEvent,
)
from workflows.handler import WorkflowHandler
from workflows.runtime.types.plugin import (
    InternalRunAdapter,
    Runtime,
)
from workflows.workflow import Workflow

from .._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    PersistentHandler,
    Status,
)
from .runtime_decorators import BaseRuntimeDecorator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _ServerInternalRunAdapter
# ---------------------------------------------------------------------------


class _ServerInternalRunAdapter(BaseInternalRunAdapterDecorator):
    """Internal adapter that records every emitted event to the workflow store.

    Handles event recording and terminal-event status updates. Tick
    persistence, state stores, and idle detection live in
    _DurableInternalRunAdapter.
    """

    def __init__(
        self,
        inner: InternalRunAdapter,
        parent: ServerRuntimeDecorator,
    ) -> None:
        super().__init__(inner)
        self._parent = parent
        self._store = parent._store

    @override
    async def write_to_event_stream(self, event: Event) -> None:
        """
        Monitors for writes to the event stream that indicate a workflow has terminated.
        """
        if isinstance(event, WorkflowFailedEvent):
            await self._parent._handle_status_update(
                run_id=self.run_id,
                status="failed",
                error=event.exception_message,
            )
        elif isinstance(event, WorkflowTimedOutEvent):
            await self._parent._handle_status_update(
                run_id=self.run_id,
                status="failed",
                error=f"Workflow timed out after {event.timeout}s",
            )
        elif isinstance(event, WorkflowCancelledEvent):
            await self._parent._handle_status_update(
                run_id=self.run_id, status="cancelled"
            )
        elif isinstance(event, StopEvent):
            await self._parent._handle_status_update(
                run_id=self.run_id,
                status="completed",
                result=event,
            )

        envelope = EventEnvelopeWithMetadata.from_event(event)
        await self._store.append_event(self.run_id, envelope)

        # Forward to inner adapter (e.g. _DurableInternalRunAdapter for idle detection)
        await super().write_to_event_stream(event)


# ---------------------------------------------------------------------------
# ServerRuntimeDecorator -- adapter wrapping, handler persistence,
# status updates, and workflow registry
# ---------------------------------------------------------------------------


class ServerRuntimeDecorator(BaseRuntimeDecorator):
    """
    Runtime decorator that wraps the main runtime to also record events to a configured
    workflow store, for integration with the WorkflowService for querying
    workflow run state.
    """

    def __init__(
        self,
        inner: Runtime,
        store: AbstractWorkflowStore,
        *,
        persistence_backoff: list[float] | None = None,
    ) -> None:
        super().__init__(inner)
        self._store: AbstractWorkflowStore = store
        self._registered_workflows: dict[str, Workflow] = {}
        self._persistence_backoff = (
            list(persistence_backoff) if persistence_backoff is not None else [0.5, 3]
        )

    async def _retry_store_write(self, coro_fn: Callable[[], Awaitable[None]]) -> None:
        """Wrap a store write with retry/backoff."""
        backoffs = list(self._persistence_backoff)
        while True:
            try:
                await coro_fn()
                return
            except Exception as e:
                backoff = backoffs.pop(0) if backoffs else None
                if backoff is None:
                    logger.error(
                        "Store write failed after final attempt",
                        exc_info=True,
                    )
                    raise
                logger.error(f"Store write failed, retrying in {backoff}s: {e}")
                await asyncio.sleep(backoff)

    # ------------------------------------------------------------------
    # Workflow registration
    # ------------------------------------------------------------------

    @override
    def track_workflow(self, workflow: Workflow) -> None:
        # Keep a strong reference — the base WorkflowSet uses weak refs,
        # so without this the workflow can be GC'd before launch().
        self._registered_workflows[workflow.workflow_name] = workflow
        super().track_workflow(workflow)

    @override
    def untrack_workflow(self, workflow: Workflow) -> None:
        self._registered_workflows.pop(workflow.workflow_name, None)
        super().untrack_workflow(workflow)

    def get_workflow(self, name: str) -> Workflow | None:
        return self._registered_workflows.get(name)

    def get_workflow_names(self) -> list[str]:
        return list(self._registered_workflows.keys())

    # ------------------------------------------------------------------
    # Adapter wiring
    # ------------------------------------------------------------------

    async def _handle_status_update(
        self,
        run_id: str,
        status: Status,
        result: StopEvent | None = None,
        error: str | None = None,
    ) -> None:
        """Callback for adapter terminal-event status updates."""
        await self._retry_store_write(
            lambda: self._store.update_handler_status(
                run_id, status=status, result=result, error=error
            )
        )

    def get_internal_adapter(self, workflow: Workflow) -> InternalRunAdapter:
        """Wraps the inner runtime's adapter in _ServerInternalRunAdapter."""
        inner_adapter = self._inner.get_internal_adapter(workflow)
        return _ServerInternalRunAdapter(inner_adapter, self)

    # ------------------------------------------------------------------
    # Handler persistence
    # ------------------------------------------------------------------

    async def run_workflow_handler(
        self,
        handler_id: str,
        workflow_name: str,
        handler: WorkflowHandler,
    ) -> WorkflowHandler:
        """Persist initial handler record to store, then notify decorator chain."""
        started_at = datetime.now(timezone.utc)

        await self._retry_store_write(
            lambda: self._store.update(
                PersistentHandler(
                    handler_id=handler_id,
                    workflow_name=workflow_name,
                    status="running",
                    run_id=handler.run_id,
                    started_at=started_at,
                    updated_at=started_at,
                )
            )
        )

        return handler
