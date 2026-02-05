# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from llama_index_instrumentation.dispatcher import instrument_tags
from workflows import Context, Workflow
from workflows.events import Event, StartEvent, StopEvent
from workflows.handler import WorkflowHandler
from workflows.utils import _nanoid as nanoid

from ._server_runtime import ServerExternalAdapter, ServerRuntimeDecorator
from .abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
)
from .keyed_lock import KeyedLock
from .memory_workflow_store import MemoryWorkflowStore

logger = logging.getLogger()


class _WorkflowService:
    """Workflow registration, startup, and event registry management.

    Delegates handler lifecycle (checkpointing, idle release, resume) to
    ServerRuntimeDecorator. This layer handles workflow registration, run
    initiation, context provider hookup, and event type registries.
    """

    def __init__(
        self,
        *,
        workflow_store: AbstractWorkflowStore | None = None,
        persistence_backoff: list[float] = [0.5, 3],
        idle_release_timeout: timedelta | None = timedelta(seconds=10),
    ) -> None:
        self._workflows: dict[str, Workflow] = {}
        self._additional_events: dict[str, list[type[Event]] | None] = {}
        self._workflow_store = (
            workflow_store if workflow_store is not None else MemoryWorkflowStore()
        )
        self._persistence_backoff = list(persistence_backoff)
        self._idle_release_timeout = idle_release_timeout
        self._reload_lock = KeyedLock()
        # Per-workflow runtime decorators
        self._decorators: dict[str, ServerRuntimeDecorator] = {}

    def add_workflow(
        self,
        name: str,
        workflow: Workflow,
        additional_events: list[type[Event]] | None = None,
    ) -> None:
        self._workflows[name] = workflow
        if additional_events is not None:
            self._additional_events[name] = additional_events
        # Wrap the workflow's runtime with the server decorator
        decorator = ServerRuntimeDecorator(
            workflow._runtime,
            store=self._workflow_store,
            persistence_backoff=self._persistence_backoff,
            idle_release_timeout=self._idle_release_timeout,
        )
        workflow._runtime = decorator
        self._decorators[name] = decorator

    async def start(self) -> None:
        """Resume previously running (non-idle) workflows from persistence."""
        handlers = await self._workflow_store.query(
            HandlerQuery(
                status_in=["running"],
                workflow_name_in=list(self._workflows.keys()),
                is_idle=False,
            )
        )
        for persistent in handlers:
            workflow = self._workflows[persistent.workflow_name]
            try:
                await self.start_workflow(
                    workflow_name=persistent.workflow_name,
                    workflow=workflow,
                    handler_id=persistent.handler_id,
                    context=Context.from_dict(workflow=workflow, data=persistent.ctx),
                )
            except Exception as e:
                logger.error(
                    f"Failed to resume handler {persistent.handler_id} for workflow {persistent.workflow_name}: {e}"
                )
                try:
                    now = datetime.now(timezone.utc)
                    await self._workflow_store.update(
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
                            ctx=persistent.ctx,
                        )
                    )
                except Exception:
                    pass
                continue

    async def stop(self) -> None:
        """Shut down all active workflow runs."""
        # Collect all active runs across all decorators
        all_runs: list[tuple[str, ServerExternalAdapter]] = []
        for decorator in self._decorators.values():
            all_runs.extend(list(decorator.active_runs.items()))
        logger.info(
            f"Shutting down Workflow server. Cancelling {len(all_runs)} handlers."
        )

        async def _cancel_run(
            handler_id: str, adapter: ServerExternalAdapter
        ) -> None:
            try:
                adapter._cancel_idle_release_timer()
                await adapter.cancel()
            except Exception:
                pass

        await asyncio.gather(
            *[_cancel_run(hid, ext) for hid, ext in all_runs]
        )
        for decorator in self._decorators.values():
            decorator._active_runs.clear()
            decorator._context_providers.clear()
            decorator._run_metadata.clear()

    async def start_workflow(
        self,
        workflow_name: str,
        workflow: Workflow,
        handler_id: str,
        start_event: StartEvent | None = None,
        context: Context | None = None,
    ) -> ServerExternalAdapter:
        """Start a workflow and return the server external adapter."""
        decorator = self._decorators.get(workflow_name)
        if decorator is None:
            raise RuntimeError(f"Workflow '{workflow_name}' not registered")

        with instrument_tags({"handler_id": handler_id}):
            handler: WorkflowHandler[Any] = workflow.run(
                ctx=context,
                start_event=start_event,
            )

            # The external adapter returned by workflow.run() is already wrapped
            # by ServerRuntimeDecorator.run_workflow() → ServerExternalAdapter
            server_ext = self._get_server_external_adapter(handler)

            # Register with the decorator
            decorator.register_run(
                handler_id,
                server_ext,
                workflow_name=workflow_name,
                workflow=workflow,
            )

            # Register context provider for checkpointing
            handler_ctx = handler.ctx

            def _get_ctx() -> dict[str, Any] | None:
                try:
                    return handler_ctx.to_dict()
                except Exception:
                    return None

            decorator.register_context_provider_strong(handler_id, _get_ctx)

            # Initial checkpoint (with retry matching step-checkpoint backoff)
            ctx_dict = _get_ctx()
            if ctx_dict is not None:
                meta = decorator.get_run_metadata(handler_id)
                if meta is not None:
                    persistent = PersistentHandler(
                        handler_id=handler_id,
                        workflow_name=workflow_name,
                        status="running",
                        run_id=handler.run_id,
                        started_at=meta.started_at,
                        updated_at=meta.updated_at,
                        ctx=ctx_dict,
                    )
                    try:
                        await self._persist_with_retry(persistent)
                    except Exception:
                        # Clean up: cancel the run and unregister
                        try:
                            await server_ext.cancel()
                        except Exception:
                            pass
                        decorator.unregister_run(handler_id)
                        raise

            # Set up completion callback to do final checkpoint and unregister
            async def _on_complete() -> None:
                # If the adapter was suspended, the inner run was cancelled
                # intentionally — don't treat it as a real completion.
                if server_ext.is_suspended:
                    return

                result = None
                error: str | None = None
                status: str = "completed"
                try:
                    result = await handler.stop_event_result()
                except asyncio.CancelledError:
                    # Re-check: suspend may have happened during the await
                    if server_ext.is_suspended:
                        return
                    status = "cancelled"
                except Exception as e:
                    status = "failed"
                    error = str(e)

                # Final checkpoint with completed status
                now = datetime.now(timezone.utc)
                meta = decorator.get_run_metadata(handler_id)
                if meta is not None:
                    ctx_dict = _get_ctx() or {}
                    meta.completed_at = now
                    meta.updated_at = now
                    persistent = PersistentHandler(
                        handler_id=handler_id,
                        workflow_name=workflow_name,
                        status=status,
                        run_id=handler.run_id,
                        error=error,
                        result=result if isinstance(result, StopEvent) else None,
                        started_at=meta.started_at,
                        updated_at=now,
                        completed_at=now,
                        ctx=ctx_dict,
                    )
                    try:
                        await self._workflow_store.update(persistent)
                    except Exception:
                        pass

                decorator.unregister_run(handler_id)

            asyncio.create_task(_on_complete())

            return server_ext

    def _get_server_external_adapter(
        self, handler: WorkflowHandler[Any]
    ) -> ServerExternalAdapter:
        """Extract the ServerExternalAdapter from a WorkflowHandler."""
        adapter = handler._external_adapter
        if isinstance(adapter, ServerExternalAdapter):
            return adapter
        # It should be wrapped by the decorator
        raise RuntimeError(
            "WorkflowHandler's external adapter is not a ServerExternalAdapter. "
            "Is the workflow's runtime wrapped by ServerRuntimeDecorator?"
        )

    def get_decorator(self, workflow_name: str) -> ServerRuntimeDecorator | None:
        """Get the server runtime decorator for a workflow."""
        return self._decorators.get(workflow_name)

    def get_adapter_for_handler(
        self, handler_id: str
    ) -> ServerExternalAdapter | None:
        """Find the active server external adapter for a handler ID."""
        for decorator in self._decorators.values():
            adapter = decorator.get_active_run(handler_id)
            if adapter is not None:
                return adapter
        return None

    def get_workflow_name_for_handler(self, handler_id: str) -> str | None:
        """Get the workflow name for an active handler."""
        for decorator in self._decorators.values():
            meta = decorator.get_run_metadata(handler_id)
            if meta is not None:
                return meta.workflow_name
        return None

    async def try_reload_handler(
        self, handler_id: str
    ) -> tuple[ServerExternalAdapter | None, PersistentHandler | None]:
        """Attempt to reload a released handler from persistence.

        Returns (adapter, persistent_data). The persistent data is returned
        so callers can inspect it without re-querying the store.

        Uses a per-handler lock to prevent concurrent reloads from creating
        duplicate workflow instances.
        """
        # Fast path: already active (no lock needed)
        adapter = self.get_adapter_for_handler(handler_id)
        if adapter is not None:
            return adapter, None

        async with self._reload_lock(handler_id):
            # Re-check after acquiring lock
            adapter = self.get_adapter_for_handler(handler_id)
            if adapter is not None:
                return adapter, None

            found = await self._workflow_store.query(
                HandlerQuery(handler_id_in=[handler_id])
            )
            if not found:
                return None, None

            handler_data = found[0]
            if handler_data.status != "running":
                return None, handler_data

            workflow = self._workflows.get(handler_data.workflow_name)
            if workflow is None:
                logger.warning(
                    f"Cannot reload {handler_id}: workflow {handler_data.workflow_name} not registered"
                )
                return None, handler_data

            try:
                context = Context.from_dict(workflow=workflow, data=handler_data.ctx)
                ext = await self.start_workflow(
                    workflow_name=handler_data.workflow_name,
                    workflow=workflow,
                    handler_id=handler_id,
                    context=context,
                )
                logger.info(f"Reloaded workflow {handler_id} from persistence")
                return ext, handler_data
            except Exception as e:
                logger.error(f"Failed to reload handler {handler_id}: {e}")
                raise

    async def _persist_with_retry(self, persistent: PersistentHandler) -> None:
        """Persist handler state with retry/backoff matching step-checkpoint logic."""
        backoffs = list(self._persistence_backoff)
        while True:
            try:
                await self._workflow_store.update(persistent)
                return
            except Exception as e:
                backoff = backoffs.pop(0) if backoffs else None
                if backoff is None:
                    raise
                logger.error(
                    f"Failed to persist handler {persistent.handler_id}. "
                    f"Retrying in {backoff}s: {e}"
                )
                await asyncio.sleep(backoff)

    def _get_context_dict(self, handler_id: str) -> dict[str, Any] | None:
        """Get context dict for a handler from any decorator."""
        for decorator in self._decorators.values():
            ctx_dict = decorator.get_context_dict(handler_id)
            if ctx_dict is not None:
                return ctx_dict
        return None

    def event_registry(self, workflow_name: str) -> dict[str, type[Event]]:
        items = {e.__name__: e for e in self._workflows[workflow_name].events}
        items.update(
            {
                e.__name__: e
                for e in self._additional_events.get(workflow_name, None) or []
            }
        )
        return items

    def prepare_run_params(
        self,
        workflow_name: str,
        start_event_data: dict | None,
        context_data: dict | None,
        handler_id: str | None,
        run_kwargs: dict | None,
    ) -> tuple[str, dict | None]:
        """Prepare and validate run parameters from already-parsed request data."""
        if run_kwargs and start_event_data is None:
            start_event_data = run_kwargs
        handler_id = handler_id or nanoid()
        return handler_id, start_event_data
