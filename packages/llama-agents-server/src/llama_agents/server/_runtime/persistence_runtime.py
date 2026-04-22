# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""TickPersistenceDecorator, PersistenceDecorator, and _PersistenceInternalRunAdapter.

TickPersistenceDecorator provides tick persistence, workflow tracking, and
context_from_ticks.  PersistenceDecorator extends it with auto-restart on
server start.  Neither handles idle detection — that lives in
IdleReleaseDecorator.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import Any, Callable

from typing_extensions import override
from workflows import Context
from workflows.context.context_types import SerializedContext
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.events import StartEvent, StopEvent
from workflows.runtime.control_loop import rebuild_state_from_ticks_stream
from workflows.runtime.runtime_decorators import (
    BaseInternalRunAdapterDecorator,
    BaseRuntimeDecorator,
)
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    Runtime,
)
from workflows.runtime.types.results import StepWorkerFailed, StepWorkerResult
from workflows.runtime.types.ticks import (
    TickCancelRun,
    TickStepResult,
    TickTimeout,
    WorkflowTick,
    WorkflowTickAdapter,
)
from workflows.workflow import Workflow

from .._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    Status,
    as_legacy_context_store,
    stream_workflow_ticks,
)
from .._store.sqlite.sqlite_state_store import SqliteStateStore

logger = logging.getLogger(__name__)


@dataclass
class _TerminalInfo:
    status: Status
    result: StopEvent | None = None
    error: str | None = None


def _classify_tick(
    tick: WorkflowTick, current: _TerminalInfo | None
) -> _TerminalInfo | None:
    """Return updated terminal info if *tick* is a terminal transition.

    The last terminal-looking tick wins; retries produce earlier
    StepWorkerFailed entries that are superseded by later successful results.
    """
    if isinstance(tick, TickStepResult):
        for step_result in tick.result:
            if isinstance(step_result, StepWorkerResult) and isinstance(
                step_result.result, StopEvent
            ):
                return _TerminalInfo(status="completed", result=step_result.result)
            if isinstance(step_result, StepWorkerFailed):
                return _TerminalInfo(status="failed", error=str(step_result.exception))
    elif isinstance(tick, TickTimeout):
        return _TerminalInfo(
            status="failed", error=f"Workflow timed out after {tick.timeout}s"
        )
    elif isinstance(tick, TickCancelRun):
        return _TerminalInfo(status="cancelled")
    return current


class _PersistenceInternalRunAdapter(BaseInternalRunAdapterDecorator):
    """Internal adapter that persists ticks to the workflow store."""

    def __init__(
        self,
        decorated: InternalRunAdapter,
        store: AbstractWorkflowStore,
    ) -> None:
        super().__init__(decorated)
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

    @override
    async def after_tick(self, tick: WorkflowTick) -> None:
        await super().after_tick(tick)
        if not isinstance(tick, TickStepResult):
            return
        tick_data = WorkflowTickAdapter.dump_python(tick, mode="json")
        try:
            await self._store.after_tick(self.run_id, tick_data)
        except Exception:
            logger.exception(
                "Failed to gather pending writes for run %s",
                self.run_id,
            )


class TickPersistenceDecorator(BaseRuntimeDecorator):
    """Runtime decorator for tick persistence and workflow tracking.

    Provides tick storage via internal adapter, workflow tracking by name,
    and context_from_ticks for rebuilding state from persisted ticks.
    """

    def __init__(
        self,
        decorated: Runtime,
        store: AbstractWorkflowStore,
    ) -> None:
        super().__init__(decorated)
        self._store = store
        self._workflows_by_name: dict[str, Workflow] = {}
        self._active_run_ids: set[str] = set()

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
        inner_adapter = self._decorated.get_internal_adapter(workflow)
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

    async def context_from_ticks(
        self,
        workflow: Workflow,
        run_id: str,
        *,
        on_tick: Callable[[WorkflowTick], None] | None = None,
    ) -> Context | None:
        """Rebuild a Context from persisted ticks (and legacy ctx if available).

        If *on_tick* is provided, it is called once per tick as the stream is
        consumed — lets callers classify terminal transitions in the same
        pass that rebuilds state.
        """
        serializer = JsonSerializer()
        legacy_ctx = self._get_legacy_ctx(run_id)

        tick_stream = stream_workflow_ticks(self._store, run_id)
        try:
            first_tick = await tick_stream.__anext__()
        except StopAsyncIteration:
            first_tick = None

        if first_tick is None and not legacy_ctx:
            return None

        if legacy_ctx:
            self._seed_legacy_state(run_id, legacy_ctx)
            parsed = SerializedContext.from_dict_auto(legacy_ctx)
            init_state = BrokerState.from_serialized(parsed, workflow, serializer)
        else:
            init_state = BrokerState.from_workflow(workflow)

        if first_tick is not None:

            async def _with_first() -> AsyncIterator[WorkflowTick]:
                if on_tick is not None:
                    on_tick(first_tick)
                yield first_tick
                async for tick in tick_stream:
                    if on_tick is not None:
                        on_tick(tick)
                    yield tick

            init_state = await rebuild_state_from_ticks_stream(
                init_state, _with_first()
            )

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
                "SELECT 1 FROM workflow_state WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is not None:
                return
        finally:
            conn.close()

        state_store._write_in_memory_state(state_data)


class PersistenceDecorator(TickPersistenceDecorator):
    """Runtime decorator that extends TickPersistenceDecorator with auto-restart.

    Resumes previously running workflows on server start.
    """

    def __init__(
        self,
        decorated: Runtime,
        store: AbstractWorkflowStore,
    ) -> None:
        super().__init__(decorated, store)
        self._background_tasks: set[asyncio.Task[None]] = set()
        self.resume_task: asyncio.Task[None] | None = None

    def _spawn_task(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @override
    async def launch(self) -> None:
        await super().launch()
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
                terminal: _TerminalInfo | None = None

                def _observe(tick: WorkflowTick) -> None:
                    nonlocal terminal
                    terminal = _classify_tick(tick, terminal)

                context = await self.context_from_ticks(
                    workflow, run_id, on_tick=_observe
                )

                if context is None:
                    # A fresh-start attempt here would build a StartEvent from
                    # empty kwargs and loop on every boot for workflows with
                    # required fields. Mark failed so the handler stops being
                    # picked up by the next resume query.
                    logger.warning(
                        "No replayable state for handler %s (workflow %s); "
                        "marking as failed",
                        persistent.handler_id,
                        persistent.workflow_name,
                    )
                    await self._store.update_handler_status(
                        run_id,
                        status="failed",
                        error="no ticks to replay on resume — handler was never persisted past creation",
                    )
                    continue

                if not context.is_running:
                    info = terminal or _TerminalInfo(
                        status="failed",
                        error="workflow terminated without a recognizable terminal event",
                    )
                    logger.warning(
                        "Replay for handler %s (workflow %s) terminated as %s; "
                        "finalizing without resume",
                        persistent.handler_id,
                        persistent.workflow_name,
                        info.status,
                    )
                    await self._store.update_handler_status(
                        run_id,
                        status=info.status,
                        result=info.result,
                        error=info.error,
                    )
                    continue

                workflow.run(ctx=context, run_id=run_id)
            except Exception as e:
                logger.error(
                    f"Failed to resume handler {persistent.handler_id} for workflow {persistent.workflow_name}: {e}"
                )
                try:
                    await self._store.update_handler_status(
                        run_id, status="failed", error=str(e)
                    )
                except Exception:
                    logger.exception(
                        "Failed to mark resume-failed handler %s as failed",
                        persistent.handler_id,
                    )
                continue

    @override
    async def destroy(self) -> None:
        await super().destroy()
        if self.resume_task is not None:
            try:
                self.resume_task.cancel()
            except Exception:
                pass
