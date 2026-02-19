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
from workflows.events import Event, StartEvent, WorkflowIdleEvent
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
)
from workflows.runtime.types.ticks import WorkflowTick
from workflows.workflow import Workflow

from dbos import DBOS

from .runtime import _IO_STREAM_TICK_TOPIC, _DBOSInternalShutdown

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
    async def write_to_event_stream(self, event: Event) -> None:
        if isinstance(event, WorkflowIdleEvent):
            idle_since = datetime.now(timezone.utc)
            await self._store.update_handler_status(
                self.run_id, status="running", idle_since=idle_since
            )
        await super().write_to_event_stream(event)
        if isinstance(event, WorkflowIdleEvent):
            self._runtime._spawn_task(self._runtime._deferred_release(self.run_id))


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
        self._active_run_ids: set[str] = set()
        self._internal_adapters: dict[str, InternalRunAdapter] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        self.stop_task: asyncio.Task[None] | None = None
        self._idle_timeout = idle_timeout

    def _spawn_task(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
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
        serializer: Any | None = None,
    ) -> ExternalRunAdapter:
        self._active_run_ids.add(run_id)
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
        inner_adapter = self._decorated.get_internal_adapter(workflow)
        wrapped = _DBOSIdleReleaseInternalRunAdapter(inner_adapter, self, self._store)
        # Track by run_id so we can close() on release
        self._internal_adapters[wrapped.run_id] = wrapped
        return wrapped

    @override
    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        return DBOSIdleReleaseExternalRunAdapter(self, run_id)

    async def _deferred_release(self, run_id: str) -> None:
        """Wait for idle_timeout then release the handler if still idle."""
        await asyncio.sleep(self._idle_timeout)
        await self._release_idle_handler(run_id)

    async def _release_idle_handler(self, run_id: str) -> None:
        """Release an idle handler: cancel in DBOS and close the control loop."""
        async with self._reload_lock(run_id):
            handlers = await self._store.query(HandlerQuery(run_id_in=[run_id]))
            if len(handlers) != 1 or handlers[0].idle_since is None:
                return
            elapsed = (
                datetime.now(timezone.utc) - handlers[0].idle_since
            ).total_seconds()
            if elapsed < self._idle_timeout:
                return
            if run_id not in self._active_run_ids:
                return

            self._active_run_ids.discard(run_id)

            # Cancel in DBOS DB — marks the workflow as cancelled
            try:
                await DBOS.cancel_workflow_async(run_id)
            except Exception:
                logger.warning(
                    f"Failed to cancel DBOS workflow [run_id={run_id}]",
                    exc_info=True,
                )
                # Re-add to active since cancel failed
                self._active_run_ids.add(run_id)
                return

            # Close the internal adapter to wake blocked recv and kill control loop
            await self._close_internal_adapter(run_id)
            logger.info(f"Released idle DBOS handler [run_id={run_id}]")

    async def _close_internal_adapter(self, run_id: str) -> None:
        """Close the internal adapter for a run, waking any blocked recv."""
        adapter = self._internal_adapters.pop(run_id, None)
        if adapter is not None:
            try:
                await adapter.close()
            except Exception:
                logger.debug(
                    f"Error closing internal adapter [run_id={run_id}]",
                    exc_info=True,
                )
        else:
            # Fallback: send shutdown signal directly via DBOS
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: DBOS.send(
                        run_id,
                        _DBOSInternalShutdown(),
                        topic=_IO_STREAM_TICK_TOPIC,
                    ),
                )
            except Exception:
                logger.debug(
                    f"Error sending shutdown signal [run_id={run_id}]",
                    exc_info=True,
                )

    async def _ensure_active_run_locked(self, run_id: str) -> None:
        """Resume a DBOS workflow that was previously idle-released.

        Called under the reload lock. Uses DBOS resume_workflow_async to
        restart the workflow from its last completed step with the same run_id.
        The caller must have already verified the handler is idle via the store.
        """
        # Resume the DBOS workflow — restarts from last completed step
        await DBOS.resume_workflow_async(run_id)

        self._active_run_ids.add(run_id)
        await self._store.update_handler_status(run_id, idle_since=None)
        logger.info(f"Resumed DBOS workflow [run_id={run_id}]")

    @override
    def destroy(self) -> None:
        super().destroy()
        if self.stop_task is not None:
            try:
                self.stop_task.cancel()
            except Exception:
                pass
        self.stop_task = self._spawn_task(self._on_server_stop())

    async def _on_server_stop(self) -> None:
        """Cancel all active runs on shutdown."""
        run_ids = list(self._active_run_ids)
        logger.info(f"Shutting down. Cancelling {len(run_ids)} DBOS handlers.")
        for run_id in run_ids:
            try:
                await DBOS.cancel_workflow_async(run_id)
            except Exception:
                logger.debug(
                    f"Error cancelling DBOS workflow on shutdown [run_id={run_id}]",
                    exc_info=True,
                )
            await self._close_internal_adapter(run_id)
        self._active_run_ids.clear()
