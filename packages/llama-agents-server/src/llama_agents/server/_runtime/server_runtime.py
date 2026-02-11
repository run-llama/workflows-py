# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
Server runtime decorator: the main required runtime decorator for workflows
served by the WorkflowServer. Handles event recording, handler persistence,
and status updates.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from llama_agents.client.protocol.serializable_events import (
    EventEnvelopeWithMetadata,
)
from llama_agents.server._runtime.runtime_decorators import (
    BaseInternalRunAdapterDecorator,
)
from typing_extensions import override
from workflows.context.serializers import BaseSerializer
from workflows.context.state_store import (
    InMemoryStateStore,
    StateStore,
    infer_state_type,
)
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
    WorkflowTimedOutEvent,
)
from workflows.handler import WorkflowHandler
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
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

    Handles event recording and terminal-event status updates.
    """

    def __init__(
        self,
        decorated: InternalRunAdapter,
        runtime: ServerRuntimeDecorator,
        *,
        state_type: type[Any] | None = None,
    ) -> None:
        super().__init__(decorated)
        self._runtime = runtime
        self._store = runtime._store
        self._state_type = state_type
        self._state_store: StateStore[Any] | None = None

    @override
    def get_state_store(self) -> StateStore[Any]:
        if self._state_store is not None:
            return self._state_store
        store = self._store.create_state_store(self.run_id, self._state_type)
        # Seed with initial context state if provided at run start
        initial = self._runtime._initial_state.pop(self.run_id, None)
        if initial is not None:
            serialized_state, serializer = initial
            self._seed_state(store, serialized_state, serializer)
        self._state_store = store
        return store

    def _seed_state(
        self,
        store: StateStore[Any],
        serialized_state: dict[str, Any],
        serializer: BaseSerializer,
    ) -> None:
        """Seed a state store from serialized state data.

        Dispatches to the appropriate seeding mechanism based on store type.
        InMemory stores are seeded directly. SQL-backed stores handle both
        their own serialized references (optimized copy) and InMemory format.
        """
        from .._store.postgres_state_store import PostgresStateStore
        from .._store.sqlite.sqlite_state_store import SqliteStateStore

        if isinstance(store, InMemoryStateStore):
            try:
                seed_store = InMemoryStateStore.from_dict(serialized_state, serializer)
                store._state = seed_store._state
            except Exception:
                logger.warning("Failed to seed InMemoryStateStore", exc_info=True)
        elif isinstance(store, SqliteStateStore):
            store._seed_from_serialized(serialized_state, serializer)
        elif isinstance(store, PostgresStateStore):
            store._pending_seed = (serialized_state, serializer)
        else:
            logger.warning(
                "Unknown state store type %s, skipping seed", type(store).__name__
            )

    @override
    async def write_to_event_stream(self, event: Event) -> None:
        """Record events to the workflow store, skipping duplicates on replay.

        During replay, non-terminal events are skipped to avoid duplicates
        (they were already persisted in the original run). Terminal events
        are always written because the TaskJournal records step completion
        *before* events are published — if a crash occurs between the journal
        write and the event write, the terminal event was never persisted and
        must be re-emitted so that ``subscribe_events()`` can detect workflow
        completion. Status updates (``_handle_status_update``) are idempotent,
        so writing them again is safe; a duplicate ``append_event`` for a
        terminal event is harmless.

        The inner adapter forwarding always happens so that runtime-level
        concerns (e.g. idle detection, DBOS stream) still function during
        replay.
        """
        replaying = self.is_replaying()
        is_terminal = isinstance(
            event,
            (
                StopEvent,
                WorkflowFailedEvent,
                WorkflowTimedOutEvent,
                WorkflowCancelledEvent,
            ),
        )

        if not replaying or is_terminal:
            if isinstance(event, WorkflowFailedEvent):
                await self._runtime._handle_status_update(
                    run_id=self.run_id,
                    status="failed",
                    error=event.exception_message,
                )
            elif isinstance(event, WorkflowTimedOutEvent):
                await self._runtime._handle_status_update(
                    run_id=self.run_id,
                    status="failed",
                    error=f"Workflow timed out after {event.timeout}s",
                )
            elif isinstance(event, WorkflowCancelledEvent):
                await self._runtime._handle_status_update(
                    run_id=self.run_id, status="cancelled"
                )
            elif isinstance(event, StopEvent):
                await self._runtime._handle_status_update(
                    run_id=self.run_id,
                    status="completed",
                    result=event,
                )

            envelope = EventEnvelopeWithMetadata.from_event(event)
            await self._store.append_event(self.run_id, envelope)

        # Always forward to inner adapter (e.g. idle detection, DBOS stream)
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
        decorated: Runtime,
        store: AbstractWorkflowStore,
        *,
        persistence_backoff: list[float] | None = None,
    ) -> None:
        super().__init__(decorated)
        self._store: AbstractWorkflowStore = store
        self._registered_workflows: dict[str, Workflow] = {}
        self._initial_state: dict[str, Any] = {}
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

    @override
    def run_workflow(
        self,
        run_id: str,
        workflow: Workflow,
        init_state: BrokerState,
        start_event: StartEvent | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> ExternalRunAdapter:
        # Intercept serialized state: we handle seeding ourselves in get_state_store
        # so non-InMemory formats don't leak to the base runtime.
        passthrough_state = serialized_state
        if serialized_state and serializer:
            self._initial_state[run_id] = (serialized_state, serializer)
            store_type = serialized_state.get("store_type")
            if store_type is not None and store_type != "in_memory":
                passthrough_state = None
        return super().run_workflow(
            run_id,
            workflow,
            init_state,
            start_event=start_event,
            serialized_state=passthrough_state,
            serializer=serializer,
        )

    def get_internal_adapter(self, workflow: Workflow) -> InternalRunAdapter:
        """Wraps the inner runtime's adapter in _ServerInternalRunAdapter."""
        inner_adapter = self._decorated.get_internal_adapter(workflow)
        state_type = infer_state_type(workflow)
        return _ServerInternalRunAdapter(inner_adapter, self, state_type=state_type)

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
