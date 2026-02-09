# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""DurableDecorator and _DurableInternalRunAdapter.

Wraps a basic runtime to add local durability: tick persistence, idle
detection, reload-on-demand, and active-run tracking.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Coroutine
from datetime import datetime, timezone
from typing import Any

from typing_extensions import override
from workflows import Context
from workflows.context.context_types import SerializedContext
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.context.state_store import (
    InMemoryStateStore,
    StateStore,
    infer_state_type,
)
from workflows.events import (
    Event,
    StartEvent,
    WorkflowIdleEvent,
)
from workflows.runtime.control_loop import rebuild_state_from_ticks
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    V2RuntimeCompatibilityShim,
    as_snapshottable_adapter,
)
from workflows.runtime.types.ticks import WorkflowTick, WorkflowTickAdapter
from workflows.workflow import Workflow

from .._keyed_lock import KeyedLock
from .._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
    as_legacy_context_store,
)
from .._store.sqlite.sqlite_state_store import SqliteStateStore
from .runtime_decorators import (
    BaseInternalRunAdapterDecorator,
    BaseRuntimeDecorator,
)

logger = logging.getLogger(__name__)


class _DurableInternalRunAdapter(BaseInternalRunAdapterDecorator):
    """Internal adapter that adds tick persistence, state-store wiring,
    and idle detection on top of a base adapter.

    Forwards all events to the inner adapter via super() so that the next
    layer in the decorator chain (e.g. _ServerInternalRunAdapter) can also
    process them.
    """

    def __init__(
        self,
        inner: InternalRunAdapter,
        outer: DurableDecorator,
        store: AbstractWorkflowStore,
        *,
        state_type: type[Any] | None = None,
    ) -> None:
        super().__init__(inner)
        self._main_runtime = outer
        self._store = store
        self._inner_snapshottable = as_snapshottable_adapter(inner)
        self._state_type = state_type
        self._state_store: StateStore[Any] | None = None

    @override
    def get_state_store(self) -> StateStore[Any]:
        if self._state_store is not None:
            return self._state_store
        store = self._store.create_state_store(self.run_id, self._state_type)
        # Seed with initial context state if provided at run start
        initial = self._main_runtime._initial_state.pop(self.run_id, None)
        if initial is not None and isinstance(store, InMemoryStateStore):
            store._state = initial
        self._state_store = store
        return store

    @override
    async def on_tick(self, tick: WorkflowTick) -> None:
        # Forward to inner adapter so in-memory tick list stays populated
        await super().on_tick(tick)
        # Persist to store
        tick_data = WorkflowTickAdapter.dump_python(tick, mode="json")
        try:
            await self._store.append_tick(self.run_id, tick_data)
        except Exception:
            logger.exception(
                "Failed to persist tick for run %s",
                self.run_id,
            )

    @override
    async def write_to_event_stream(self, event: Event) -> None:
        if isinstance(event, WorkflowIdleEvent):
            idle_since = datetime.now(timezone.utc)
            await self._store.update_handler_status(
                self.run_id, status="running", idle_since=idle_since
            )
        # Forward to inner adapter (base class forwards to _inner)
        await super().write_to_event_stream(event)
        # Schedule release after event is fully written (fire-and-forget to
        # avoid self-cancellation since we're inside the handler's task)
        if isinstance(event, WorkflowIdleEvent):
            skip = self._main_runtime._skip_idle_release.get(self.run_id, 0)
            if skip > 0:
                self._main_runtime._skip_idle_release[self.run_id] = skip - 1
            else:
                self._main_runtime._spawn_task(
                    self._main_runtime._release_idle_handler(self.run_id)
                )


class DurableExternalRunAdapter(ExternalRunAdapter):
    """External adapter that adds reload-on-demand to a base adapter.

    The inner adapter is resolved lazily because an idle-released handler
    needs to be reloaded first (which creates new queues in the basic runtime).
    """

    def __init__(self, outer: DurableDecorator, run_id: str) -> None:
        self._outer = outer
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def _get_inner(self) -> ExternalRunAdapter:
        return self._outer._inner.get_external_adapter(self._run_id)

    @override
    async def send_event(self, tick: WorkflowTick) -> None:
        # Serialize send/reload with idle-release. This prevents a stale
        # release task from aborting the run while an external event is being
        # delivered.
        async with self._outer._reload_lock(self._run_id):
            # When reloading from an idle state, the workflow will immediately
            # go idle again before the event reaches it. Increment the skip
            # counter so the first idle event after reload doesn't trigger a
            # release.
            if self._run_id not in self._outer._active_run_ids:
                counter = self._outer._skip_idle_release
                counter[self._run_id] = counter.get(self._run_id, 0) + 2
                await self._outer._ensure_active_run_locked(self._run_id)
            else:
                # Mark as non-idle before delivering input so a previously
                # queued release task can detect activity and no-op.
                await self._outer._store.update_handler_status(
                    self._run_id, idle_since=None
                )
            await self._get_inner().send_event(tick)

    def stream_published_events(self) -> Any:
        return self._get_inner().stream_published_events()

    async def close(self) -> None:
        await self._get_inner().close()

    async def get_result(self) -> Any:
        return await self._get_inner().get_result()

    async def cancel(self) -> None:
        await self._get_inner().cancel()

    def get_state_store(self) -> Any:
        return self._get_inner().get_state_store()


class DurableDecorator(BaseRuntimeDecorator):
    """Runtime decorator that adds durability to a basic runtime for server mode.

    Manages active runs, reload-on-demand from persisted ticks,
    and start/stop lifecycle for resuming workflows.

    For now, only supports a single node runtime. If multiple nodes were to share a state store,
    sending_events will not route correctly (possibly creating duplicate workflow runs), and workflow
    recovery will compete and create duplicate workflow runs as well.
    """

    def __init__(
        self,
        inner: Any,
        store: AbstractWorkflowStore,
    ) -> None:
        super().__init__(inner)
        self._store = store
        self._reload_lock = KeyedLock()
        self._active_run_ids: set[str] = set()
        self._skip_idle_release: dict[str, int] = {}
        self._initial_state: dict[str, Any] = {}
        self._workflows_by_name: dict[str, Workflow] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self.resume_task: asyncio.Task[None] | None = None
        self.stop_task: asyncio.Task[None] | None = None

    def _spawn_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

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
        self._active_run_ids.add(run_id)
        if serialized_state and serializer:
            try:
                seed_store = InMemoryStateStore.from_dict(serialized_state, serializer)
                self._initial_state[run_id] = seed_store._state
            except Exception:
                pass
        return super().run_workflow(
            run_id,
            workflow,
            init_state,
            start_event=start_event,
            serialized_state=serialized_state,
            serializer=serializer,
        )

    @override
    def get_internal_adapter(self, workflow: Workflow) -> InternalRunAdapter:
        inner_adapter = self._inner.get_internal_adapter(workflow)
        state_type = infer_state_type(workflow)
        return _DurableInternalRunAdapter(
            inner_adapter,
            self,
            self._store,
            state_type=state_type,
        )

    @override
    def track_workflow(self, workflow: Workflow) -> None:
        self._workflows_by_name[workflow.workflow_name] = workflow
        super().track_workflow(workflow)

    @override
    def untrack_workflow(self, workflow: Workflow) -> None:
        self._workflows_by_name.pop(workflow.workflow_name, None)
        super().untrack_workflow(workflow)

    @override
    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        return DurableExternalRunAdapter(self, run_id)

    async def _release_idle_handler(self, run_id: str) -> None:
        """Release an idle handler from memory. Called via asyncio.create_task."""
        async with self._reload_lock(run_id):
            handlers = await self._store.query(HandlerQuery(run_id_in=[run_id]))
            if len(handlers) != 1 or handlers[0].idle_since is None:
                return
            if run_id not in self._active_run_ids:
                return
            self._active_run_ids.discard(run_id)
            self._abort_inner_run(run_id)
            logger.info(f"Released idle handler [run_id={run_id}] from memory")

    def _abort_inner_run(self, run_id: str) -> None:
        """Cancel the inner runtime's control loop task for a run and clean up."""
        try:
            inner_adapter = self._inner.get_external_adapter(run_id)
        except Exception:
            return
        # TODO - support without the shim
        if isinstance(inner_adapter, V2RuntimeCompatibilityShim):
            inner_adapter.abort()
        else:
            raise ValueError(f"Inner adapter {inner_adapter} does not support abort")

    async def _ensure_active_run(self, run_id: str) -> None:
        if run_id in self._active_run_ids:
            return
        async with self._reload_lock(run_id):
            await self._ensure_active_run_locked(run_id)

    async def _ensure_active_run_locked(self, run_id: str) -> None:
        if run_id in self._active_run_ids:
            return
        handlers = await self._store.query(HandlerQuery(run_id_in=[run_id]))
        if len(handlers) != 1:
            raise ValueError(
                f"Expected 1 handler for run {run_id}, got {len(handlers)}"
            )
        handler = handlers[0]
        workflow = self._workflows_by_name.get(handler.workflow_name)
        if workflow is None:
            raise ValueError(f"Workflow {handler.workflow_name} not found")
        context = await self._context_from_ticks(workflow, run_id)
        workflow.run(ctx=context, run_id=run_id)
        self._active_run_ids.add(run_id)
        # Clear idle_since so the handler doesn't appear idle after reload
        await self._store.update_handler_status(run_id, idle_since=None)
        logger.info(
            f"Reloaded workflow [handler_id={handler.handler_id}, run_id={run_id}] from persistence"
        )

    async def _context_from_ticks(
        self, workflow: Workflow, run_id: str
    ) -> Context | None:
        """Rebuild a Context from persisted ticks (and legacy ctx if available)."""
        stored_ticks = await self._store.get_ticks(run_id)
        serializer = JsonSerializer()

        # Try to load legacy ctx data for back-compat migration
        legacy_ctx = self._get_legacy_ctx(run_id)

        if not stored_ticks and not legacy_ctx:
            return None

        if legacy_ctx:
            # Seed user state from old ctx if state table has no data yet
            self._seed_legacy_state(run_id, legacy_ctx)

            # Use old broker state as init state for tick replay
            parsed = SerializedContext.from_dict_auto(legacy_ctx)
            init_state = BrokerState.from_serialized(parsed, workflow, serializer)
        else:
            init_state = BrokerState.from_workflow(workflow)

        if stored_ticks:
            ticks = [
                WorkflowTickAdapter.validate_python(st.tick_data) for st in stored_ticks
            ]
            init_state = rebuild_state_from_ticks(init_state, ticks)

        serialized = init_state.to_serialized(serializer)
        return Context.from_dict(
            workflow=workflow, data=serialized.model_dump(), serializer=serializer
        )

    def _get_legacy_ctx(self, run_id: str) -> dict[str, Any] | None:
        """Get legacy ctx data from the store if it supports it."""
        legacy_store = as_legacy_context_store(self._store)
        if legacy_store is None:
            return None
        try:
            return legacy_store.get_legacy_ctx(run_id)
        except Exception:
            logger.warning(
                "Failed to read legacy ctx for run %s", run_id, exc_info=True
            )
            return None

    def _seed_legacy_state(self, run_id: str, legacy_ctx: dict[str, Any]) -> None:
        """Migrate user state from old ctx into the state table if not already present."""
        try:
            parsed = SerializedContext.from_dict_auto(legacy_ctx)
        except Exception:
            logger.warning(
                "Failed to parse legacy ctx for state migration, run %s", run_id
            )
            return

        state_data = parsed.state
        if not state_data:
            return

        # Check if state already exists for this run
        state_store = self._store.create_state_store(run_id)
        # SqliteStateStore uses _load_state which creates a default if missing,
        # so we check directly if a row exists
        if not isinstance(state_store, SqliteStateStore):
            return

        conn = sqlite3.connect(state_store._db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM state WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is not None:
                return  # State already exists, don't overwrite
        finally:
            conn.close()

        state_store._write_in_memory_state(state_data)

    @override
    def launch(self) -> None:
        super().launch()
        self.resume_task = self._spawn_task(
            self._on_server_start(self._workflows_by_name)
        )

    async def _on_server_start(self, registered_workflows: dict[str, Workflow]) -> None:
        """Resume previously running (non-idle) workflows from persistence."""
        handlers = await self._store.query(
            HandlerQuery(
                status_in=["running"],
                workflow_name_in=list(registered_workflows.keys()),
                is_idle=False,
            )
        )
        for persistent in handlers:
            workflow = registered_workflows.get(persistent.workflow_name)
            if workflow is None:
                continue
            if persistent.run_id is None:
                logger.error(f"Run ID is required for handler {persistent.handler_id}")
                continue
            run_id = persistent.run_id
            if run_id in self._active_run_ids:
                continue
            try:
                context = await self._context_from_ticks(workflow, run_id)
                workflow.run(ctx=context, run_id=run_id)
                self._active_run_ids.add(run_id)
            except Exception as e:
                logger.error(
                    f"Failed to resume handler {persistent.handler_id} for workflow {persistent.workflow_name}: {e}"
                )
                try:
                    now = datetime.now(timezone.utc)
                    await self._store.update(
                        PersistentHandler(
                            handler_id=persistent.handler_id,
                            workflow_name=persistent.workflow_name,
                            status="failed",
                            run_id=persistent.run_id,
                            error=str(e),
                            result=None,
                            started_at=persistent.started_at,
                            updated_at=now,
                            completed_at=now,
                        )
                    )
                except Exception:
                    pass
                continue

    @override
    def destroy(self) -> None:
        super().destroy()
        if self.resume_task is not None:
            try:
                self.resume_task.cancel()
            except Exception:
                pass
        if self.stop_task is not None:
            try:
                self.stop_task.cancel()
            except Exception:
                pass
        self.stop_task = self._spawn_task(self._on_server_stop())

    async def _on_server_stop(self) -> None:
        """Cancel all active runs."""
        run_ids = list(self._active_run_ids)
        logger.info(f"Shutting down. Cancelling {len(run_ids)} handlers.")
        for run_id in run_ids:
            self._abort_inner_run(run_id)
        self._active_run_ids.clear()
