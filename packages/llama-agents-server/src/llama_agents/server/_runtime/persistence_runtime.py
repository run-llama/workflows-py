# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""PersistenceDecorator and _PersistenceInternalRunAdapter.

Wraps a basic runtime to add tick persistence and auto-restart on server
start.  Does NOT handle idle detection or reload-on-demand — those live in
IdleReleaseDecorator.
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
from workflows.events import StartEvent
from workflows.runtime.control_loop import rebuild_state_from_ticks
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
)
from workflows.runtime.types.ticks import WorkflowTick, WorkflowTickAdapter
from workflows.workflow import Workflow

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


class _PersistenceInternalRunAdapter(BaseInternalRunAdapterDecorator):
    """Internal adapter that persists ticks to the workflow store."""

    def __init__(
        self,
        inner: InternalRunAdapter,
        store: AbstractWorkflowStore,
    ) -> None:
        super().__init__(inner)
        self._store = store

    @override
    async def on_tick(self, tick: WorkflowTick) -> None:
        await super().on_tick(tick)
        tick_data = WorkflowTickAdapter.dump_python(tick, mode="json")
        try:
            await self._store.append_tick(self.run_id, tick_data)
        except Exception:
            logger.exception(
                "Failed to persist tick for run %s",
                self.run_id,
            )


class PersistenceDecorator(BaseRuntimeDecorator):
    """Runtime decorator for tick persistence and auto-restart.

    Manages workflow tracking, tick persistence via internal adapter,
    and resuming previously running workflows on server start.
    """

    def __init__(
        self,
        inner: Any,
        store: AbstractWorkflowStore,
    ) -> None:
        super().__init__(inner)
        self._store = store
        self._workflows_by_name: dict[str, Workflow] = {}
        self._active_run_ids: set[str] = set()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self.resume_task: asyncio.Task[None] | None = None

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
        return _PersistenceInternalRunAdapter(inner_adapter, self._store)

    @override
    def track_workflow(self, workflow: Workflow) -> None:
        self._workflows_by_name[workflow.workflow_name] = workflow
        super().track_workflow(workflow)

    @override
    def untrack_workflow(self, workflow: Workflow) -> None:
        self._workflows_by_name.pop(workflow.workflow_name, None)
        super().untrack_workflow(workflow)

    def get_tracked_workflow(self, name: str) -> Workflow | None:
        """Look up a tracked workflow by name (used by IdleReleaseDecorator)."""
        return self._workflows_by_name.get(name)

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
                context = await self.context_from_ticks(workflow, run_id)
                workflow.run(ctx=context, run_id=run_id)
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

    async def context_from_ticks(
        self, workflow: Workflow, run_id: str
    ) -> Context | None:
        """Rebuild a Context from persisted ticks (and legacy ctx if available)."""
        stored_ticks = await self._store.get_ticks(run_id)
        serializer = JsonSerializer()

        legacy_ctx = self._get_legacy_ctx(run_id)

        if not stored_ticks and not legacy_ctx:
            return None

        if legacy_ctx:
            self._seed_legacy_state(run_id, legacy_ctx)
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

        state_store = self._store.create_state_store(run_id)
        if not isinstance(state_store, SqliteStateStore):
            return

        conn = sqlite3.connect(state_store._db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM state WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is not None:
                return
        finally:
            conn.close()

        state_store._write_in_memory_state(state_data)
