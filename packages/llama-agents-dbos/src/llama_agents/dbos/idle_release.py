# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""DBOSIdleReleaseDecorator and supporting adapters.

Wraps an EventInterceptorDecorator to add idle detection, memory release via
DBOS cancel_workflow, and reload-on-demand via DBOS resume_workflow.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa
from llama_agents.server._keyed_lock import KeyedLock
from llama_agents.server._runtime.runtime_decorators import (
    BaseExternalRunAdapterDecorator,
    BaseInternalRunAdapterDecorator,
    BaseRuntimeDecorator,
)
from llama_agents.server._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
)
from typing_extensions import override
from workflows.events import Event, WorkflowIdleEvent
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    WaitResult,
    WaitResultTick,
)
from workflows.runtime.types.ticks import WorkflowTick
from workflows.workflow import Workflow

from dbos import DBOS
from dbos._context import _dbos_context_var
from dbos._dbos import _get_dbos_instance
from dbos._sys_db import SystemSchema

from .runtime import _IO_STREAM_TICK_TOPIC

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
        self._was_cancelled = False

    @override
    async def wait_receive(
        self,
        timeout_seconds: float | None = None,
    ) -> WaitResult:
        result = await super().wait_receive(timeout_seconds)
        if isinstance(result, WaitResultTick):
            self._runtime._cancel_deferred_release(self.run_id)
            self._runtime._suppress_idle_release.discard(self.run_id)
        return result

    @override
    async def close(self) -> None:
        if self._was_cancelled:
            # Skip sending _DBOSInternalShutdown — the workflow was cancelled
            # by idle release. Sending shutdown would create a stale
            # notification that the resumed workflow would consume.
            return
        await super().close()

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
    sync but reload (resume_workflow) is async.
    """

    def __init__(self, runtime: DBOSIdleReleaseDecorator, run_id: str) -> None:
        # Intentionally skip super().__init__ — _decorated is a lazy property.
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
                await self._runtime._ensure_active_run_locked(self.run_id)
            await self._decorated.send_event(tick)


class DBOSIdleReleaseDecorator(BaseRuntimeDecorator):
    """Runtime decorator for idle detection, release via cancel_workflow,
    and reload via resume_workflow for DBOS-backed workflows.

    Must wrap an EventInterceptorDecorator (or compatible runtime) that
    wraps a DBOSRuntime.
    """

    def __init__(
        self,
        decorated: BaseRuntimeDecorator,
        store: AbstractWorkflowStore,
        idle_timeout: float = 60.0,
    ) -> None:
        super().__init__(decorated)
        self._store = store
        self._reload_lock = KeyedLock()
        self._deferred_release_tasks: dict[str, asyncio.Task[None]] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._idle_timeout = idle_timeout
        # Track active internal adapters by run_id so we can mark them
        # as cancelled during idle release (prevents stale shutdown notifications).
        self._active_adapters: dict[str, _DBOSIdleReleaseInternalRunAdapter] = {}
        # Run IDs where idle release is suppressed because a resume is in
        # progress and a tick is pending delivery. Cleared when the tick is
        # consumed by wait_receive.
        self._suppress_idle_release: set[str] = set()

    def _spawn_task(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _schedule_deferred_release(self, run_id: str) -> None:
        """Cancel any existing timer for run_id and schedule a new one."""
        if run_id in self._suppress_idle_release:
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
        adapter = _DBOSIdleReleaseInternalRunAdapter(inner_adapter, self, self._store)
        self._active_adapters[adapter.run_id] = adapter
        return adapter

    @override
    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        return DBOSIdleReleaseExternalRunAdapter(self, run_id)

    async def _deferred_release(self, run_id: str) -> None:
        """Wait for idle_timeout then release the handler if still idle."""
        await asyncio.sleep(self._idle_timeout)
        self._deferred_release_tasks.pop(run_id, None)
        await self._release_idle_handler(run_id)

    async def _release_idle_handler(self, run_id: str) -> None:
        """Release an idle handler: cancel in DBOS and let control loop die naturally."""
        async with self._reload_lock(run_id):
            handlers = await self._store.query(HandlerQuery(run_id_in=[run_id]))
            if len(handlers) != 1 or handlers[0].idle_since is None:
                return
            elapsed = (
                datetime.now(timezone.utc) - handlers[0].idle_since
            ).total_seconds()
            if elapsed < self._idle_timeout:
                return

            # Mark the adapter as cancelled so close() skips sending
            # _DBOSInternalShutdown (which would create a stale notification).
            adapter = self._active_adapters.pop(run_id, None)
            if adapter is not None:
                adapter._was_cancelled = True

            # Clear the DBOS workflow context so these operations don't get
            # journaled as steps of the (about-to-be-cancelled) workflow.
            # Without this, cancel_workflow would be replayed on resume and
            # send_async would see the cancelled context and raise.
            token = _dbos_context_var.set(None)
            try:
                # Cancel in DBOS DB — marks the workflow as cancelled
                try:
                    await DBOS.cancel_workflow_async(run_id)
                except Exception:
                    logger.warning(
                        f"Failed to cancel DBOS workflow [run_id={run_id}]",
                        exc_info=True,
                    )
                    return

                # Wake the recv thread by sending a None notification.
                # recv returns None → WaitResultTimeout, the loop continues,
                # and the next recv detects CANCELLED status and exits cleanly.
                await DBOS.send_async(run_id, None, topic=_IO_STREAM_TICK_TOPIC)
            finally:
                _dbos_context_var.reset(token)

            logger.info(f"Released idle DBOS handler [run_id={run_id}]")

    async def _ensure_active_run_locked(self, run_id: str) -> None:
        """Resume a DBOS workflow that was previously idle-released.

        Called under the reload lock. Uses DBOS resume_workflow_async to
        restart the workflow from its last completed step with the same run_id.
        The caller must have already verified the handler is idle via the store.
        """
        self._cancel_deferred_release(run_id)
        # Suppress idle release scheduling during replay — the resumed workflow
        # will re-emit WorkflowIdleEvent while rebuilding state, which would
        # schedule a new timer that fires before the tick is consumed.
        # Cleared when wait_receive sees the tick.
        self._suppress_idle_release.add(run_id)

        # Delete stale notifications left over from the cancel cycle.
        # The original workflow's recv consumed its notification live (DELETE),
        # but DBOS recv replay returns cached results WITHOUT deleting from DB.
        # Any notifications inserted after the original recv (e.g. the wake-up
        # None from _release_idle_handler) persist and would be consumed by the
        # resumed workflow's fresh recv instead of the real tick.
        dbos_inst = _get_dbos_instance()
        with dbos_inst._sys_db.engine.begin() as conn:
            conn.execute(
                sa.delete(SystemSchema.notifications).where(
                    SystemSchema.notifications.c.destination_uuid == run_id,
                    SystemSchema.notifications.c.topic == _IO_STREAM_TICK_TOPIC,
                )
            )

        token = _dbos_context_var.set(None)
        try:
            await DBOS.resume_workflow_async(run_id)
        finally:
            _dbos_context_var.reset(token)
        logger.info(f"Resumed DBOS workflow [run_id={run_id}]")
        await self._store.update_handler_status(run_id, idle_since=None)
