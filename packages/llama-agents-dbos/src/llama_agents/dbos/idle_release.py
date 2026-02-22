# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""DBOSIdleReleaseDecorator and supporting adapters.

Wraps an EventInterceptorDecorator to add idle detection, memory release via
TickCancelRun, and reload-on-demand via continue-as-new (new run_id).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from datetime import datetime, timezone
from typing import Any

from llama_agents.server._keyed_lock import KeyedLock
from llama_agents.server._runtime.persistence_runtime import TickPersistenceDecorator
from llama_agents.server._runtime.runtime_decorators import (
    BaseExternalRunAdapterDecorator,
    BaseInternalRunAdapterDecorator,
    BaseRuntimeDecorator,
)
from llama_agents.server._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
)
from typing_extensions import override
from workflows.context.context_types import SerializedContext
from workflows.context.serializers import JsonSerializer
from workflows.context.state_store import create_in_memory_payload, infer_state_type
from workflows.events import Event, WorkflowIdleEvent
from workflows.runtime.control_loop import rebuild_state_from_ticks
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    WaitResult,
    WaitResultTick,
)
from workflows.runtime.types.ticks import (
    TickCancelRun,
    WorkflowTick,
    WorkflowTickAdapter,
)
from workflows.utils import _nanoid as nanoid
from workflows.workflow import Workflow

logger = logging.getLogger(__name__)


class _DBOSIdleReleaseInternalRunAdapter(BaseInternalRunAdapterDecorator):
    """Internal adapter that detects idle events and schedules release."""

    def __init__(
        self,
        decorated: InternalRunAdapter,
        runtime: DBOSIdleReleaseDecorator,
        store: AbstractWorkflowStore,
    ) -> None:
        super().__init__(decorated)
        self._runtime = runtime
        self._store = store

    @override
    async def wait_receive(
        self,
        timeout_seconds: float | None = None,
    ) -> WaitResult:
        result = await super().wait_receive(timeout_seconds)
        if isinstance(result, WaitResultTick):
            self._runtime._cancel_deferred_release(self.run_id)
        return result

    @override
    async def write_to_event_stream(self, event: Event) -> None:
        if isinstance(event, WorkflowIdleEvent):
            idle_since = datetime.now(timezone.utc)
            await self._store.update_handler_status(
                self.run_id, status="running", idle_since=idle_since
            )
        await super().write_to_event_stream(event)
        if isinstance(event, WorkflowIdleEvent):
            self._runtime._schedule_deferred_release(self.run_id)


class DBOSIdleReleaseExternalRunAdapter(BaseExternalRunAdapterDecorator):
    """Proxy adapter that adds reload-on-demand for idle-released DBOS handlers.

    The inner adapter is resolved lazily because ``get_external_adapter`` is
    sync but reload (continue-as-new) is async.
    """

    def __init__(self, runtime: DBOSIdleReleaseDecorator, run_id: str) -> None:
        # Intentionally skip super().__init__ -- _decorated is a lazy property.
        self._runtime = runtime
        self._run_id = run_id

    @property  # type: ignore[override]
    def _decorated(self) -> ExternalRunAdapter:
        return self._runtime._decorated.get_external_adapter(self._run_id)

    @_decorated.setter
    def _decorated(self, value: ExternalRunAdapter) -> None:
        pass

    @property
    def run_id(self) -> str:
        return self._run_id

    @override
    async def send_event(self, tick: WorkflowTick) -> None:
        async with self._runtime._reload_lock(self.run_id):
            handlers = await self._runtime._store.query(
                HandlerQuery(run_id_in=[self.run_id])
            )
            if len(handlers) == 1 and handlers[0].idle_since is not None:
                new_run_id = await self._runtime._ensure_active_run_locked(self.run_id)
                self._run_id = new_run_id
            await self._decorated.send_event(tick)


class DBOSIdleReleaseDecorator(BaseRuntimeDecorator):
    """Runtime decorator for idle detection, release via TickCancelRun,
    and reload via continue-as-new for DBOS-backed workflows.

    Must wrap an EventInterceptorDecorator (or compatible runtime) that
    wraps a DBOSRuntime.
    """

    def __init__(
        self,
        decorated: BaseRuntimeDecorator,
        store: AbstractWorkflowStore,
        idle_timeout: float = 60.0,
        tick_persistence: TickPersistenceDecorator | None = None,
    ) -> None:
        super().__init__(decorated)
        self._store = store
        self._reload_lock = KeyedLock()
        self._deferred_release_tasks: dict[str, asyncio.Task[None]] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._idle_timeout = idle_timeout
        self._idle_releasing: set[str] = set()
        if tick_persistence is None:
            raise ValueError("tick_persistence is required")
        self._tick_persistence: TickPersistenceDecorator = tick_persistence

    def _spawn_task(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _schedule_deferred_release(self, run_id: str) -> None:
        """Cancel any existing timer for run_id and schedule a new one."""
        if run_id in self._idle_releasing:
            return
        self._cancel_deferred_release(run_id)
        task = self._spawn_task(self._deferred_release(run_id))
        self._deferred_release_tasks[run_id] = task

    def _cancel_deferred_release(self, run_id: str) -> None:
        """Cancel a pending deferred release timer for run_id, if any."""
        task = self._deferred_release_tasks.pop(run_id, None)
        if task is not None and not task.done():
            task.cancel()

    @override
    def get_internal_adapter(self, workflow: Workflow) -> InternalRunAdapter:
        inner_adapter = self._decorated.get_internal_adapter(workflow)
        return _DBOSIdleReleaseInternalRunAdapter(inner_adapter, self, self._store)

    @override
    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        return DBOSIdleReleaseExternalRunAdapter(self, run_id)

    async def _deferred_release(self, run_id: str) -> None:
        """Wait for idle_timeout then release the handler if still idle."""
        await asyncio.sleep(self._idle_timeout)
        self._deferred_release_tasks.pop(run_id, None)
        await self._release_idle_handler(run_id)

    async def _release_idle_handler(self, run_id: str) -> None:
        """Release an idle handler by sending TickCancelRun."""
        async with self._reload_lock(run_id):
            handlers = await self._store.query(HandlerQuery(run_id_in=[run_id]))
            if len(handlers) != 1 or handlers[0].idle_since is None:
                return
            elapsed = (
                datetime.now(timezone.utc) - handlers[0].idle_since
            ).total_seconds()
            if elapsed < self._idle_timeout:
                return

            self._idle_releasing.add(run_id)
            await self._decorated.get_external_adapter(run_id).send_event(
                TickCancelRun()
            )
            logger.info(f"Released idle DBOS handler [run_id={run_id}]")

    async def _broker_state_from_ticks(
        self, workflow: Workflow, run_id: str
    ) -> BrokerState:
        """Rebuild BrokerState from persisted ticks without converting to Context."""
        stored_ticks = await self._store.get_ticks(run_id)
        serializer = JsonSerializer()

        legacy_ctx = self._tick_persistence._get_legacy_ctx(run_id)

        if legacy_ctx:
            self._tick_persistence._seed_legacy_state(run_id, legacy_ctx)
            parsed = SerializedContext.from_dict_auto(legacy_ctx)
            init_state = BrokerState.from_serialized(parsed, workflow, serializer)
        else:
            init_state = BrokerState.from_workflow(workflow)

        if stored_ticks:
            ticks = [
                WorkflowTickAdapter.validate_python(st.tick_data) for st in stored_ticks
            ]
            init_state = rebuild_state_from_ticks(init_state, ticks)

        return init_state

    async def _ensure_active_run_locked(self, run_id: str) -> str:
        """Resume a workflow that was previously idle-released using continue-as-new.

        Called under the reload lock. Rebuilds state from ticks, creates a new
        run_id, and starts a fresh workflow run.

        Returns the new run_id.
        """
        self._cancel_deferred_release(run_id)

        # Look up handler to get workflow_name
        handlers = await self._store.query(HandlerQuery(run_id_in=[run_id]))
        if len(handlers) != 1:
            raise ValueError(
                f"Expected 1 handler for run {run_id}, got {len(handlers)}"
            )
        handler = handlers[0]

        workflow = self._tick_persistence.get_tracked_workflow(handler.workflow_name)
        if workflow is None:
            raise ValueError(f"Workflow {handler.workflow_name} not found")

        # Rebuild BrokerState from persisted ticks
        init_state = await self._broker_state_from_ticks(workflow, run_id)

        # Carry over state from old run's state store
        serializer = JsonSerializer()
        serialized_state: dict[str, Any] | None = None
        state_type = infer_state_type(workflow)
        if state_type is not None:
            try:
                old_state_store = self._store.create_state_store(
                    run_id, state_type=state_type
                )
                state = await old_state_store.get_state()
                if state is not None:
                    payload = create_in_memory_payload(state, serializer)
                    serialized_state = payload.model_dump()
            except Exception:
                logger.warning(
                    f"Failed to carry over state from run {run_id}", exc_info=True
                )

        # Generate new run_id
        new_run_id = nanoid()

        # Start new workflow run
        self._decorated.run_workflow(
            new_run_id,
            workflow,
            init_state,
            serialized_state=serialized_state,
            serializer=serializer,
        )

        # Update handler in store: new run_id, clear idle, set running
        updated_handler = PersistentHandler(
            handler_id=handler.handler_id,
            workflow_name=handler.workflow_name,
            status="running",
            run_id=new_run_id,
            error=handler.error,
            result=handler.result,
            started_at=handler.started_at,
            updated_at=datetime.now(timezone.utc),
            completed_at=handler.completed_at,
            idle_since=None,
        )
        await self._store.update(updated_handler)

        self._idle_releasing.discard(run_id)

        logger.info(
            f"Resumed DBOS workflow via continue-as-new "
            f"[old_run_id={run_id}, new_run_id={new_run_id}]"
        )
        return new_run_id
