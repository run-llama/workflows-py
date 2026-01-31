# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
DBOS Runtime for durable workflow execution.

This module provides the DBOSRuntime class for running LlamaIndex workflows
with durable execution backed by DBOS.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator

from llama_index_instrumentation.dispatcher import active_instrument_tags
from workflows.context.serializers import BaseSerializer
from workflows.context.state_store import StateStore
from workflows.events import Event, StartEvent, StopEvent
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    RegisteredWorkflow,
    Runtime,
    WaitResult,
    WaitResultTick,
    WaitResultTimeout,
)
from workflows.runtime.types.step_function import (
    StepWorkerFunction,
    as_step_worker_functions,
    create_workflow_run_function,
)
from workflows.runtime.types.ticks import WorkflowTick
from workflows.runtime.workflow_tracker import WorkflowTracker
from workflows.workflow import Workflow

from dbos import DBOS, SetWorkflowID

from .sql_state_store import (
    PostgresStateStore,
    SqliteStateStore,
    SqlStateStore,
    create_state_store,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# Very long timeout for unbounded waits - encourages workflow to sleep.
# DBOS's default 60s is too short and gets recorded to event logs.
_UNBOUNDED_WAIT_TIMEOUT_SECONDS = 60 * 60 * 24  # 1 day


@dataclass
class _DBOSInternalShutdown:
    """Internal signal sent via DBOS.send to wake blocked recv for shutdown."""

    pass


def _durable_time() -> float:
    """Get current timestamp.

    Note: This is intentionally NOT decorated with @DBOS.step() because:
    1. The timestamp durability comes from being recorded in the workflow's output,
       not from step-level recording.
    2. Using @DBOS.step() here causes a race condition: it temporarily sets
       step context on the shared DBOSContext, which causes concurrent worker
       tasks to see is_step=True. This makes their DBOS.step wrappers bypass
       durability (because ctx.is_workflow() returns False when is_step is True).
    """
    return time.time()


class DBOSRuntime(Runtime):
    """
    DBOS-backed workflow runtime for durable execution.

    Workflows are registered at launch() time with stable names,
    enabling distributed workers and recovery.

    State is persisted to the database using SQL state stores,
    enabling state recovery across process restarts.
    """

    def __init__(self, polling_interval_sec: float = 1.0) -> None:
        self._tracker = WorkflowTracker()
        self._dbos_launched = False
        self._tasks: list[asyncio.Task[None]] = []
        self._polling_interval_sec = polling_interval_sec
        self._sql_engine: Engine | None = None
        self._migrations_run = False

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.append(task)
        task.add_done_callback(self._tasks.remove)

    def track_workflow(self, workflow: Workflow) -> None:
        """Track a workflow for registration at launch time.

        If launch() was already called, registers the workflow immediately.
        This allows late registration for testing scenarios.
        """
        if self._dbos_launched:
            # Already launched - register immediately
            registered = self.register(workflow)
            self._tracker.set_registered(workflow, registered)
        else:
            self._tracker.add(workflow)

    def get_registered(self, workflow: Workflow) -> RegisteredWorkflow | None:
        """Get the registered workflow if available."""
        return self._tracker.get_registered(workflow)

    def register(self, workflow: Workflow) -> RegisteredWorkflow:
        """
        Wrap workflow with DBOS decorators.

        Called at launch() time for each tracked workflow.
        Uses workflow.workflow_name for stable DBOS registration names.
        """
        # Use workflow's name directly
        name = workflow.workflow_name

        # Create DBOS-wrapped control loop with stable name
        @DBOS.workflow(name=f"{name}.control_loop")
        async def _dbos_control_loop(
            init_state: BrokerState,
            start_event: StartEvent | None = None,
            tags: dict[str, Any] = {},
        ) -> StopEvent:
            workflow_run_fn = create_workflow_run_function(workflow)
            return await workflow_run_fn(init_state, start_event, tags)

        # Wrap steps with stable names
        wrapped_steps: dict[str, StepWorkerFunction[Any]] = {
            step_name: DBOS.step(name=f"{name}.{step_name}")(step)
            for step_name, step in as_step_worker_functions(workflow).items()
        }

        return RegisteredWorkflow(
            workflow=workflow, workflow_run_fn=_dbos_control_loop, steps=wrapped_steps
        )

    def _get_sql_engine(self) -> Engine:
        """Get the SQLAlchemy engine from DBOS for state storage.

        Uses DBOS's app database if configured, otherwise falls back to sys database.

        Returns:
            SQLAlchemy Engine for state storage.

        Raises:
            RuntimeError: If no database is available.
        """
        if self._sql_engine is not None:
            return self._sql_engine

        from dbos._dbos import _get_dbos_instance

        dbos = _get_dbos_instance()

        # Try app database first, fall back to system database
        app_db = dbos._app_db
        if app_db is not None:
            self._sql_engine = app_db.engine
            return self._sql_engine

        # Fall back to system database
        sys_db = dbos._sys_db
        self._sql_engine = sys_db.engine
        return self._sql_engine

    def _run_state_store_migrations(self) -> None:
        """Run migrations for SQL state stores.

        Called once at launch() time to create the workflow_state table.
        """
        if self._migrations_run:
            return

        from .sql_state_store import create_state_store

        engine = self._get_sql_engine()
        # Create a temporary store to trigger migrations
        temp_store = create_state_store(run_id="_migration_check", engine=engine)
        temp_store._ensure_initialized()
        self._migrations_run = True
        logger.info("SQL state store migrations completed")

    def run_workflow(
        self,
        run_id: str,
        workflow: Workflow,
        init_state: BrokerState,
        start_event: StartEvent | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
        adapter_state: dict[str, Any] | None = None,
    ) -> ExternalRunAdapter:
        """Set up a workflow run with SQL-backed state storage.

        State is persisted to the database, enabling recovery across
        process restarts and distributed execution.
        """
        from workflows.context.serializers import JsonSerializer
        from workflows.context.state_store import infer_state_type

        from .sql_state_store import create_state_store

        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before running workflows."
            )

        registered = self.get_registered(workflow)
        if registered is None:
            raise RuntimeError(
                "DBOSRuntime workflows must be registered before running. Did you forget to call runtime.launch()?"
            )

        # Create SQL state store for durable state persistence
        active_serializer = serializer or JsonSerializer()
        engine = self._get_sql_engine()

        # Infer state type from workflow step configs
        state_type = infer_state_type(registered.workflow)

        # Create SQL state store
        state_store = create_state_store(
            run_id=run_id,
            engine=engine,
            state_type=state_type,
            serializer=active_serializer,
        )
        _dbos_state_stores[run_id] = state_store

        async def _run_workflow() -> None:
            with SetWorkflowID(run_id):
                try:
                    await DBOS.start_workflow_async(
                        registered.workflow_run_fn,
                        init_state,
                        start_event,
                        active_instrument_tags.get(),
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to submit work to DBOS for {run_id} with start event: {start_event} and init state: {init_state}. Error: {e}",
                        exc_info=True,
                    )
                    raise e

        # fire and forget
        self._track_task(asyncio.create_task(_run_workflow()))

        # Return adapter - execution is handled by broker for now
        return ExternalDBOSAdapter(run_id, self._polling_interval_sec)

    def get_internal_adapter(self) -> InternalRunAdapter:
        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before running workflows."
            )
        run_id = DBOS.workflow_id
        if run_id is None:
            raise RuntimeError(
                "No current run id. Must be called within a workflow run."
            )
        return InternalDBOSAdapter(run_id)

    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before running workflows."
            )
        return ExternalDBOSAdapter(run_id, self._polling_interval_sec)

    def launch(self) -> None:
        """
        Launch DBOS and register all tracked workflows.

        Must be called before running any workflows.
        Runs SQL state store migrations to create the workflow_state table.
        """
        if self._dbos_launched:
            return  # Already launched

        # Register each pending workflow with DBOS
        for workflow in self._tracker.get_pending():
            # Get steps from the workflow instance (includes class + added steps)

            # Register with DBOS (this applies decorators)
            registered = self.register(workflow)
            if registered is not None:
                self._tracker.set_registered(workflow, registered)

        # Mark as launched (no more workflows can be added)
        self._tracker.mark_launched()

        # Launch DBOS runtime
        DBOS.launch()
        self._dbos_launched = True

        # Run SQL state store migrations after DBOS is launched
        self._run_state_store_migrations()

    def destroy(self, destroy_dbos: bool = True) -> None:
        """Clean up DBOS runtime resources.

        Args:
            destroy_dbos: If True (default), also calls DBOS.destroy().
                Set to False when DBOS lifecycle is managed externally
                (e.g., shared across multiple runtimes in tests).
        """
        self._tracker.clear()
        self._dbos_launched = False
        self._sql_engine = None
        self._migrations_run = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if destroy_dbos:
            DBOS.destroy()


# State stores by run_id
# TODO: Add cleanup mechanism for completed workflows
_dbos_state_stores: dict[str, StateStore[Any]] = {}


_IO_STREAM_PUBLISHED_EVENTS_NAME = "published_events"
_IO_STREAM_TICK_TOPIC = "ticks"


class InternalDBOSAdapter(InternalRunAdapter):
    """
    Internal DBOS adapter for the workflow control loop.

    - send_event sends ticks via DBOS.send (using run_in_executor to escape step context)
    - wait_receive receives ticks via DBOS.recv_async
    - write_to_event_stream publishes events via DBOS streams
    - get_now returns a durable timestamp
    - close sends shutdown signal to wake blocked recv
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._closed = False

    @property
    def run_id(self) -> str:
        return self._run_id

    async def write_to_event_stream(self, event: Event) -> None:
        try:
            await DBOS.write_stream_async(_IO_STREAM_PUBLISHED_EVENTS_NAME, event)
        except Exception as e:
            # HACK: Ignore duplicate stream writes during replay.
            # DBOS tracks stream offsets internally, and during replay
            # (or certain race conditions), writes may conflict.
            # This is a temporary workaround until proper stream handling is implemented.
            if "UNIQUE constraint failed" in str(e):
                logger.debug(f"Ignoring duplicate stream write: {e}")
            else:
                raise

    async def get_now(self) -> float:
        return _durable_time()

    async def send_event(self, tick: WorkflowTick) -> None:
        # Use run_in_executor to escape DBOS step context.
        # DBOS.send requires workflow context, but when called from a step's
        # background task, we're in step context which fails the check.
        # Running in a thread pool escapes the contextvar-based context.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: DBOS.send(self._run_id, tick, topic=_IO_STREAM_TICK_TOPIC),
        )

    async def wait_receive(
        self,
        timeout_seconds: float | None = None,
    ) -> WaitResult:
        """Wait for tick using DBOS.recv_async with timeout.

        For bounded waits (timeout_seconds specified), uses the specified timeout.
        For unbounded waits (timeout_seconds=None), loops with very long timeouts
        (hours) to encourage the workflow to sleep. Uses shutdown signal from
        close() to exit cleanly - raises CancelledError on shutdown.
        """
        if self._closed:
            raise asyncio.CancelledError("Adapter closed")

        if timeout_seconds is not None:
            # Bounded wait - use specified timeout
            result = await DBOS.recv_async(
                _IO_STREAM_TICK_TOPIC,
                timeout_seconds=timeout_seconds,
            )
            if result is None:
                return WaitResultTimeout()
            if isinstance(result, _DBOSInternalShutdown):
                self._closed = True
                raise asyncio.CancelledError("Adapter closed")
            return WaitResultTick(tick=result)

        # Unbounded wait - loop with very long timeouts, never return TIMEOUT
        while True:
            result = await DBOS.recv_async(
                _IO_STREAM_TICK_TOPIC,
                timeout_seconds=_UNBOUNDED_WAIT_TIMEOUT_SECONDS,
            )
            if result is None:
                # Timeout expired - just loop again (sleep more)
                continue
            if isinstance(result, _DBOSInternalShutdown):
                self._closed = True
                raise asyncio.CancelledError("Adapter closed")
            return WaitResultTick(tick=result)

    async def close(self) -> None:
        """Signal shutdown by sending internal message to wake blocked recv."""
        if self._closed:
            return
        self._closed = True

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: DBOS.send(
                self._run_id, _DBOSInternalShutdown(), topic=_IO_STREAM_TICK_TOPIC
            ),
        )

    def get_state_store(self) -> "StateStore[Any] | None":
        return _dbos_state_stores.get(self._run_id)


class ExternalDBOSAdapter(ExternalRunAdapter):
    """
    External DBOS adapter for workflow interaction.

    - send_event puts ticks into the shared mailbox queue
    - stream_published_events reads from DBOS streams
    - close is a no-op
    """

    def __init__(self, run_id: str, polling_interval_sec: float = 1.0) -> None:
        self._run_id = run_id
        self._polling_interval_sec = polling_interval_sec

    @property
    def run_id(self) -> str:
        """Get the workflow run ID."""
        return self._run_id

    async def send_event(self, tick: WorkflowTick) -> None:
        await DBOS.send_async(self._run_id, tick, topic=_IO_STREAM_TICK_TOPIC)

    async def stream_published_events(self) -> AsyncGenerator[Event, None]:
        await self._ensure_workflow_started()

        async for event in DBOS.read_stream_async(self.run_id, "published_events"):
            yield event

    async def get_result(self) -> StopEvent:
        await self._ensure_workflow_started()
        handle = await DBOS.retrieve_workflow_async(self.run_id)
        return await handle.get_result(polling_interval_sec=self._polling_interval_sec)

    async def _ensure_workflow_started(self) -> None:
        # Wait for the workflow to exist in DBOS before trying to read from its stream.
        workflow_id = self.run_id
        max_wait = 10.0
        poll_interval = 0.01
        max_poll_interval = 1.0
        elapsed = 0.0
        while elapsed < max_wait:
            status = await DBOS.get_workflow_status_async(workflow_id)
            if status is not None:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            poll_interval = min(poll_interval * 2.0, max_poll_interval)
        else:
            raise RuntimeError(
                f"Workflow {workflow_id} did not start within {max_wait} seconds"
            )


__all__ = [
    "DBOSRuntime",
    "InternalDBOSAdapter",
    "ExternalDBOSAdapter",
    "_dbos_state_stores",
    "SqlStateStore",
    "PostgresStateStore",
    "SqliteStateStore",
    "create_state_store",
]
