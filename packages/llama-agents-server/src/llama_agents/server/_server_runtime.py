# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
Server runtime decorator: owns checkpointing, idle detection, suspend/resume,
and startup resumption for workflows served by WorkflowServer.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from workflows.events import (
    Event,
    StartEvent,
    StepState,
    StepStateChanged,
    StopEvent,
    WorkflowIdleEvent,
)
from workflows.runtime.types.named_task import NamedTask
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
)
from workflows.runtime.types.ticks import WorkflowTick

from .abstract_workflow_store import (
    AbstractWorkflowStore,
    PersistentHandler,
    Status,
)
from .keyed_lock import KeyedLock
from .runtime_decorators import (
    BaseExternalRunAdapterDecorator,
    BaseInternalRunAdapterDecorator,
    BaseRuntimeDecorator,
)

logger = logging.getLogger()

# Type alias for context provider callbacks
ContextProvider = Callable[[], dict[str, Any] | None]


class ServerRuntimeDecorator(BaseRuntimeDecorator):
    """Runtime decorator that adds server-layer lifecycle management.

    Wraps each workflow's runtime at registration time to provide:
    - Checkpointing via after_step_completed() on the internal adapter
    - Idle detection via wait_for_next_task() on the internal adapter
    - Suspend/resume for idle release
    - Startup resumption of previously running workflows
    """

    def __init__(
        self,
        inner: Any,
        *,
        store: AbstractWorkflowStore,
        persistence_backoff: list[float] | None = None,
        idle_release_timeout: timedelta | None = timedelta(seconds=10),
    ) -> None:
        super().__init__(inner)
        self._store = store
        self._persistence_backoff = (
            list(persistence_backoff) if persistence_backoff else [0.5, 3]
        )
        self._idle_release_timeout = idle_release_timeout
        self._reload_lock = KeyedLock()

        # Active runs keyed by handler_id
        self._active_runs: dict[str, ServerExternalAdapter] = {}
        # Per-run context providers (weakrefs to avoid preventing GC)
        self._context_providers: dict[str, weakref.ref[Any]] = {}
        # Per-run metadata
        self._run_metadata: dict[str, _RunMetadata] = {}

    def run_workflow(
        self,
        run_id: str,
        workflow: Any,
        init_state: Any,
        start_event: StartEvent | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: Any = None,
    ) -> ExternalRunAdapter:
        inner_adapter = self._inner.run_workflow(
            run_id,
            workflow,
            init_state,
            start_event=start_event,
            serialized_state=serialized_state,
            serializer=serializer,
        )
        server_external = ServerExternalAdapter(
            inner_adapter,
            runtime_decorator=self,
        )
        return server_external

    def get_internal_adapter(self, workflow: Any) -> InternalRunAdapter:
        inner = self._inner.get_internal_adapter(workflow)
        return ServerInternalAdapter(inner, runtime_decorator=self)

    def register_run(
        self,
        handler_id: str,
        server_external: ServerExternalAdapter,
        *,
        workflow_name: str,
        workflow: Any = None,
    ) -> None:
        """Register a run after workflow.run() returns."""
        self._active_runs[handler_id] = server_external
        self._run_metadata[handler_id] = _RunMetadata(
            handler_id=handler_id,
            workflow_name=workflow_name,
            started_at=datetime.now(timezone.utc),
            workflow=workflow,
        )

    def register_context_provider(
        self,
        run_id: str,
        provider: ContextProvider,
    ) -> None:
        """Register a callback to get ctx.to_dict() for checkpointing."""
        # Store as-is; we handle missing providers gracefully
        self._context_providers[run_id] = weakref.ref(provider)  # type: ignore[arg-type]

    def register_context_provider_strong(
        self,
        run_id: str,
        provider: ContextProvider,
    ) -> None:
        """Register a context provider without weakref (for lambdas/closures)."""
        # Store a wrapper that keeps a strong reference
        self._context_providers[run_id] = _StrongRef(provider)  # type: ignore[assignment]

    def get_context_dict(self, run_id: str) -> dict[str, Any] | None:
        """Get the current context dict for a run."""
        ref = self._context_providers.get(run_id)
        if ref is None:
            return None
        provider = ref() if callable(ref) else None
        if provider is None:
            return None
        # provider is the actual callable
        if callable(provider):
            return provider()
        return None

    def get_run_metadata(self, handler_id: str) -> _RunMetadata | None:
        return self._run_metadata.get(handler_id)

    def get_active_run(self, handler_id: str) -> ServerExternalAdapter | None:
        return self._active_runs.get(handler_id)

    def unregister_run(self, handler_id: str) -> None:
        """Remove a run from active tracking."""
        self._active_runs.pop(handler_id, None)
        self._context_providers.pop(handler_id, None)
        self._run_metadata.pop(handler_id, None)

    @property
    def store(self) -> AbstractWorkflowStore:
        return self._store

    @property
    def active_runs(self) -> dict[str, ServerExternalAdapter]:
        return self._active_runs

    @property
    def reload_lock(self) -> KeyedLock:
        return self._reload_lock


class _StrongRef:
    """Wrapper that mimics weakref.ref but holds a strong reference."""

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __call__(self) -> Any:
        return self._obj


class _RunMetadata:
    """Per-run metadata tracked by the server runtime decorator."""

    def __init__(
        self,
        handler_id: str,
        workflow_name: str,
        started_at: datetime,
        workflow: Any = None,
    ) -> None:
        self.handler_id = handler_id
        self.workflow_name = workflow_name
        self.started_at = started_at
        self.updated_at = started_at
        self.completed_at: datetime | None = None
        self.idle_since: datetime | None = None
        self.workflow = workflow


class ServerInternalAdapter(BaseInternalRunAdapterDecorator):
    """Internal adapter decorator that handles checkpointing and idle detection."""

    def __init__(
        self,
        inner: InternalRunAdapter,
        *,
        runtime_decorator: ServerRuntimeDecorator,
    ) -> None:
        super().__init__(inner)
        self._runtime = runtime_decorator

    async def write_to_event_stream(self, event: Event) -> None:
        """Intercept events written to the stream for idle detection."""
        await self._inner.write_to_event_stream(event)

        handler_id = self._find_handler_id()
        if handler_id is None:
            return

        meta = self._runtime.get_run_metadata(handler_id)
        if meta is None:
            return

        if isinstance(event, WorkflowIdleEvent):
            meta.idle_since = datetime.now(timezone.utc)
            server_ext = self._runtime.get_active_run(handler_id)
            if server_ext is not None:
                server_ext._notify_idle()
        elif isinstance(event, StepStateChanged) and event.step_state == StepState.RUNNING:
            if meta.idle_since is not None:
                meta.idle_since = None
                server_ext = self._runtime.get_active_run(handler_id)
                if server_ext is not None:
                    server_ext._notify_active()

    async def after_step_completed(self) -> None:
        """Checkpoint after each step completes."""
        await self._inner.after_step_completed()
        handler_id = self._find_handler_id()
        if handler_id is None:
            return
        await self._checkpoint(handler_id)

    def _find_handler_id(self) -> str | None:
        """Find the handler_id for this adapter's run_id."""
        run_id = self.run_id
        for hid, ext in self._runtime.active_runs.items():
            if ext.run_id == run_id:
                return hid
        return None

    async def _checkpoint(self, handler_id: str) -> None:
        """Persist handler state with retry/backoff."""
        meta = self._runtime.get_run_metadata(handler_id)
        if meta is None:
            return

        ctx_dict = self._runtime.get_context_dict(handler_id)
        if ctx_dict is None:
            logger.debug(
                f"Skipping checkpoint for {handler_id}: context provider not available"
            )
            return

        meta.updated_at = datetime.now(timezone.utc)

        # Determine status from the external adapter
        server_ext = self._runtime.get_active_run(handler_id)
        status: Status = "running"
        error: str | None = None
        result: StopEvent | None = None

        if server_ext is not None:
            try:
                inner_done = server_ext._inner_done()
                if inner_done:
                    try:
                        result = server_ext._inner_result()
                        status = "completed"
                        meta.completed_at = meta.updated_at
                    except asyncio.CancelledError:
                        status = "cancelled"
                        meta.completed_at = meta.updated_at
                    except Exception as e:
                        status = "failed"
                        error = str(e)
                        meta.completed_at = meta.updated_at
            except Exception:
                pass

        persistent = PersistentHandler(
            handler_id=handler_id,
            workflow_name=meta.workflow_name,
            status=status,
            run_id=self.run_id,
            error=error,
            result=result,
            started_at=meta.started_at,
            updated_at=meta.updated_at,
            completed_at=meta.completed_at,
            idle_since=meta.idle_since,
            ctx=ctx_dict,
        )

        backoffs = list(self._runtime._persistence_backoff)
        while True:
            try:
                await self._runtime.store.update(persistent)
                return
            except Exception as e:
                backoff = backoffs.pop(0) if backoffs else None
                if backoff is None:
                    logger.error(
                        f"Failed to checkpoint handler {handler_id} after final attempt.",
                        exc_info=True,
                    )
                    # Cancel the run on fatal checkpoint failure
                    server_ext = self._runtime.get_active_run(handler_id)
                    if server_ext is not None:
                        try:
                            await server_ext.cancel()
                        except Exception:
                            pass
                    raise
                logger.error(
                    f"Failed to checkpoint handler {handler_id}. "
                    f"Retrying in {backoff}s: {e}"
                )
                await asyncio.sleep(backoff)


class ServerExternalAdapter(BaseExternalRunAdapterDecorator):
    """External adapter decorator that handles suspend/resume and idle management."""

    def __init__(
        self,
        inner: ExternalRunAdapter,
        *,
        runtime_decorator: ServerRuntimeDecorator,
    ) -> None:
        super().__init__(inner)
        self._runtime = runtime_decorator
        self._idle_release_timer: asyncio.Task[None] | None = None
        self._suspended = False

    def _notify_idle(self) -> None:
        """Called by ServerInternalAdapter when workflow becomes idle."""
        self._start_idle_release_timer()

    def _notify_active(self) -> None:
        """Called by ServerInternalAdapter when workflow becomes active."""
        self._cancel_idle_release_timer()

    def _start_idle_release_timer(self) -> None:
        """Start a timer to suspend this handler after the idle timeout."""
        if self._runtime._idle_release_timeout is None:
            return
        self._cancel_idle_release_timer()
        timeout = self._runtime._idle_release_timeout.total_seconds()

        async def release_after_timeout() -> None:
            try:
                await asyncio.sleep(timeout)
                handler_id = self._find_handler_id()
                if handler_id is not None:
                    await self._suspend(handler_id)
            except asyncio.CancelledError:
                pass

        self._idle_release_timer = asyncio.create_task(release_after_timeout())

    def _cancel_idle_release_timer(self) -> None:
        """Cancel any pending idle release timer."""
        if self._idle_release_timer is not None:
            self._idle_release_timer.cancel()
            self._idle_release_timer = None

    async def _suspend(self, handler_id: str) -> None:
        """Suspend the inner run, keeping this adapter alive."""
        async with self._runtime.reload_lock(handler_id):
            if self._suspended:
                return
            # Final checkpoint
            meta = self._runtime.get_run_metadata(handler_id)
            ctx_dict = self._runtime.get_context_dict(handler_id)
            if meta is not None and ctx_dict is not None:
                meta.updated_at = datetime.now(timezone.utc)
                persistent = PersistentHandler(
                    handler_id=handler_id,
                    workflow_name=meta.workflow_name,
                    status="running",
                    run_id=self.run_id,
                    started_at=meta.started_at,
                    updated_at=meta.updated_at,
                    idle_since=meta.idle_since,
                    ctx=ctx_dict,
                )
                try:
                    await self._runtime.store.update(persistent)
                except Exception:
                    logger.error(
                        f"Failed to checkpoint before suspend for {handler_id}",
                        exc_info=True,
                    )
            # Mark suspended before cancelling so that the _on_complete
            # callback (which awaits stop_event_result) sees the flag and
            # skips its final persist — otherwise it would overwrite the
            # "running" status we just checkpointed.
            self._suspended = True
            # Cancel the inner run
            try:
                await self._inner.cancel()
            except Exception:
                pass
            # Clear context provider since inner run is gone
            self._runtime._context_providers.pop(handler_id, None)

        logger.info(f"Suspended idle workflow {handler_id}")

    async def send_event(self, tick: WorkflowTick) -> None:
        """Send event, resuming if suspended."""
        if self._suspended:
            handler_id = self._find_handler_id()
            if handler_id is not None:
                await self._resume(handler_id)
        # Mark active to cancel idle timer
        self._cancel_idle_release_timer()
        await self._inner.send_event(tick)

    async def _resume(self, handler_id: str) -> None:
        """Resume from persistence."""
        async with self._runtime.reload_lock(handler_id):
            if not self._suspended:
                return
            meta = self._runtime.get_run_metadata(handler_id)
            if meta is None:
                logger.warning(f"Cannot resume {handler_id}: no metadata")
                return
            # Query store for latest state
            from .abstract_workflow_store import HandlerQuery

            found = await self._runtime.store.query(
                HandlerQuery(handler_id_in=[handler_id])
            )
            if not found:
                logger.warning(f"Cannot resume {handler_id}: not in store")
                return
            handler_data = found[0]
            if handler_data.status != "running":
                logger.warning(
                    f"Cannot resume {handler_id}: status is {handler_data.status}"
                )
                return

            workflow = meta.workflow
            if workflow is None:
                logger.warning(f"Cannot resume {handler_id}: no workflow ref")
                return

            from workflows.context import Context
            from workflows.runtime.types.internal_state import BrokerState

            ctx = Context.from_dict(workflow=workflow, data=handler_data.ctx)
            init_state = BrokerState.from_workflow(workflow)

            # Launch new inner run
            inner_adapter = self._runtime._inner.run_workflow(
                run_id=self.run_id,
                workflow=workflow,
                init_state=init_state,
                serialized_state=handler_data.ctx.get("state"),
            )
            self._inner = inner_adapter
            self._suspended = False
            meta.idle_since = None

            # Re-register context provider
            handler_ctx = ctx

            def _get_ctx() -> dict[str, Any] | None:
                try:
                    return handler_ctx.to_dict()
                except Exception:
                    return None

            self._runtime.register_context_provider_strong(handler_id, _get_ctx)

        logger.info(f"Resumed workflow {handler_id}")

    def _inner_done(self) -> bool:
        """Check if the inner adapter's task is done."""
        if self._suspended:
            return False
        from workflows.runtime.types.plugin import as_v2_runtime_compatibility_shim

        shim = as_v2_runtime_compatibility_shim(self._inner)
        if shim is not None:
            return not shim.is_running
        return False

    def _inner_result(self) -> StopEvent:
        """Get the result from the inner adapter (blocking-safe only when done)."""
        from workflows.runtime.types.plugin import as_v2_runtime_compatibility_shim

        shim = as_v2_runtime_compatibility_shim(self._inner)
        if shim is not None:
            result = shim.get_result_or_none()
            if result is not None:
                return result
        raise RuntimeError("Result not available")

    def _find_handler_id(self) -> str | None:
        """Find the handler_id for this adapter."""
        for hid, ext in self._runtime.active_runs.items():
            if ext is self:
                return hid
        return None

    @property
    def is_suspended(self) -> bool:
        return self._suspended
