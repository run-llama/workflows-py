# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from llama_index_instrumentation.dispatcher import instrument_tags
from workflows import Context, Workflow
from workflows.events import Event, StartEvent
from workflows.handler import WorkflowHandler
from workflows.utils import _nanoid as nanoid

from ._handler import _NamedWorkflow, _WorkflowHandler
from .abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
)
from .keyed_lock import KeyedLock
from .memory_workflow_store import MemoryWorkflowStore

logger = logging.getLogger()


class _WorkflowService:
    """Handler lifecycle, persistence, and event registry management.

    This layer owns the _handlers dict, _workflows dict, _reload_lock,
    and all lifecycle methods. It has no knowledge of HTTP.
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
        self._handlers: dict[str, _WorkflowHandler] = {}
        self._workflow_store = (
            workflow_store if workflow_store is not None else MemoryWorkflowStore()
        )
        self._persistence_backoff = list(persistence_backoff)
        self._idle_release_timeout = idle_release_timeout
        self._reload_lock = KeyedLock()

    def add_workflow(
        self,
        name: str,
        workflow: Workflow,
        additional_events: list[type[Event]] | None = None,
    ) -> None:
        self._workflows[name] = workflow
        if additional_events is not None:
            self._additional_events[name] = additional_events

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
                    workflow=_NamedWorkflow(
                        name=persistent.workflow_name, workflow=workflow
                    ),
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
        logger.info(
            f"Shutting down Workflow server. Cancelling {len(self._handlers)} handlers."
        )
        await asyncio.gather(
            *[self.close_handler(handler) for handler in list(self._handlers.values())]
        )
        self._handlers.clear()

    async def start_workflow(
        self,
        workflow: _NamedWorkflow,
        handler_id: str,
        start_event: StartEvent | None = None,
        context: Context | None = None,
        idle_since: datetime | None = None,
    ) -> _WorkflowHandler:
        """Start a workflow and return a wrapper for the handler."""
        with instrument_tags({"handler_id": handler_id}):
            handler = workflow.workflow.run(
                ctx=context,
                start_event=start_event,
            )
            wrapper = await self.run_workflow_handler(
                handler_id, workflow.name, handler, idle_since=idle_since
            )
            return wrapper

    async def run_workflow_handler(
        self,
        handler_id: str,
        workflow_name: str,
        handler: WorkflowHandler,
        idle_since: datetime | None = None,
    ) -> _WorkflowHandler:
        """Create a wrapper for the handler and start streaming events."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        started_at = datetime.now(timezone.utc)

        wrapper = _WorkflowHandler(
            run_handler=handler,
            queue=queue,
            task=None,
            consumer_mutex=asyncio.Lock(),
            handler_id=handler_id,
            workflow_name=workflow_name,
            started_at=started_at,
            updated_at=started_at,
            completed_at=None,
            _workflow_store=self._workflow_store,
            _persistence_backoff=self._persistence_backoff,
            _idle_release_timeout=self._idle_release_timeout,
            _on_idle_release=self.release_handler,
        )
        wrapper.idle_since = idle_since
        # Initial checkpoint before registration; fail fast if persistence is unavailable
        await wrapper.checkpoint()
        # Now register and start streaming
        self._handlers[handler_id] = wrapper

        async def on_finish() -> None:
            self._handlers.pop(handler_id, None)

        wrapper.start_streaming(on_finish=on_finish)

        return wrapper

    async def close_handler(self, handler: _WorkflowHandler) -> None:
        """Close and cleanup a handler."""
        await handler.cancel_handlers_and_tasks()
        self._handlers.pop(handler.handler_id, None)

    async def release_handler(self, wrapper: _WorkflowHandler) -> None:
        """Release an idle handler from memory, keeping it in persistence."""
        handler_id = wrapper.handler_id

        async with self._reload_lock(handler_id):
            current = self._handlers.get(handler_id)
            if current is not None and current is not wrapper:
                logger.debug(
                    f"Skipping release checkpoint for {handler_id}: "
                    "handler was already reloaded"
                )
                wrapper._cancel_idle_release_timer(skip_checkpoint=True)
                await wrapper.cancel_handlers_and_tasks()
                return

            self._handlers.pop(handler_id, None)
            wrapper._cancel_idle_release_timer()

            try:
                await wrapper.checkpoint()
            finally:
                await wrapper.cancel_handlers_and_tasks()

        logger.info(f"Released idle workflow {handler_id} from memory")

    async def try_reload_handler(
        self, handler_id: str
    ) -> tuple[_WorkflowHandler | None, PersistentHandler | None]:
        """Attempt to reload a released handler from persistence.

        Uses per-handler locking to prevent concurrent reloads from creating
        duplicate workflow instances.

        Returns (wrapper, persistent_data). The persistent data is returned
        so callers can inspect it without re-querying the store.
        """
        async with self._reload_lock(handler_id):
            if handler_id in self._handlers:
                return self._handlers[handler_id], None

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
                wrapper = await self.start_workflow(
                    workflow=_NamedWorkflow(
                        name=handler_data.workflow_name, workflow=workflow
                    ),
                    handler_id=handler_id,
                    context=context,
                    idle_since=handler_data.idle_since,
                )

                if wrapper.idle_since is not None:
                    wrapper._start_idle_release_timer()

                logger.info(f"Reloaded workflow {handler_id} from persistence")
                return wrapper, handler_data
            except Exception as e:
                logger.error(f"Failed to reload handler {handler_id}: {e}")
                raise

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
        """Prepare and validate run parameters from already-parsed request data.

        Returns (handler_id, start_event_data) where handler_id may be generated
        if not provided.
        """
        if run_kwargs and start_event_data is None:
            start_event_data = run_kwargs
        handler_id = handler_id or nanoid()
        return handler_id, start_event_data
