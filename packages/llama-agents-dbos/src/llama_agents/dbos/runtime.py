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
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, AsyncGenerator, TypedDict, cast

import asyncpg
from llama_index_instrumentation.dispatcher import active_instrument_tags
from pydantic import BaseModel
from typing_extensions import Unpack
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.context.state_store import (
    StateStore,
    deserialize_state_from_dict,
    infer_state_type,
)
from workflows.events import Event, StartEvent, StopEvent
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.named_task import NamedTask
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
from workflows.workflow import Workflow

try:
    from dbos import DBOS, SetWorkflowID
    from dbos._dbos import _get_dbos_instance
except ImportError as e:
    # if 3.9, give a detailed error that dbos is not supported on this version of python
    if sys.version_info.major == 3 and sys.version_info.minor <= 9:
        raise ImportError(
            "dbos is not supported on Python 3.9. Please use Python 3.10 or higher."
            f"Error: {e}"
        ) from e
    raise

from llama_agents.server._runtime.event_interceptor import EventInterceptorDecorator
from llama_agents.server._store.abstract_workflow_store import AbstractWorkflowStore
from llama_agents.server._store.postgres_state_store import PostgresStateStore
from llama_agents.server._store.postgres_workflow_store import (
    PostgresWorkflowStore,
)
from llama_agents.server._store.sqlite.sqlite_state_store import SqliteStateStore
from llama_agents.server._store.sqlite.sqlite_workflow_store import SqliteWorkflowStore
from sqlalchemy.engine import URL as SaURL
from sqlalchemy.engine import Engine

from .journal.crud import JOURNAL_TABLE_NAME, PostgresJournalCrud, SqliteJournalCrud
from .journal.task_journal import TaskJournal

STATE_TABLE_NAME = "workflow_state"

logger = logging.getLogger(__name__)


class DBOSRuntimeConfig(TypedDict, total=False):
    """Configuration options for DBOSRuntime.

    All fields are optional — defaults are resolved at launch time.
    """

    polling_interval_sec: float
    run_migrations_on_launch: bool
    schema: str | None
    state_table_name: str
    journal_table_name: str


DEFAULT_STATE_TABLE_NAME = STATE_TABLE_NAME
DEFAULT_JOURNAL_TABLE_NAME = JOURNAL_TABLE_NAME


def _resolve_schema(config: DBOSRuntimeConfig, engine: Engine) -> str | None:
    """Resolve schema from config, falling back to dialect-based default.

    If "schema" was explicitly provided (even as None), uses that value.
    Otherwise, defaults to "dbos" for PostgreSQL and None for SQLite.
    """
    if "schema" in config:
        return config["schema"]
    is_postgres = engine.dialect.name == "postgresql"
    return "dbos" if is_postgres else None


def _sqlalchemy_url_to_asyncpg_dsn(url: SaURL) -> str:
    """Convert a SQLAlchemy URL to an asyncpg-compatible DSN.

    Strips dialect driver suffixes (e.g. postgresql+psycopg2 -> postgresql)
    and renders the URL as a plain connection string.
    """
    # url is a sqlalchemy.engine.URL object
    # Set the drivername to plain 'postgresql' for asyncpg
    plain_url = url.set(drivername="postgresql")
    return plain_url.render_as_string(hide_password=False)


# Very long timeout for unbounded waits - encourages workflow to sleep.
# DBOS's default 60s is too short and gets recorded to event logs.
_UNBOUNDED_WAIT_TIMEOUT_SECONDS = 60 * 60 * 24  # 1 day


@dataclass
class _DBOSInternalShutdown:
    """Internal signal sent via DBOS.send to wake blocked recv for shutdown."""


@DBOS.step()
def _durable_time() -> float:
    """
    Get current timestamp, wrapped as a DBOS step so that it's snapshotted and replayed
    This could be made more consistent if it got the timestamp from the DB.
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

    def __init__(self, **kwargs: Unpack[DBOSRuntimeConfig]) -> None:
        """Initialize the DBOS runtime.

        Args:
            **kwargs: Configuration options. See DBOSRuntimeConfig for details.
                polling_interval_sec: Interval for polling workflow results. Default 1.0.
                run_migrations_on_launch: Auto-run migrations on launch(). Default True.
                schema: Database schema name. Default: auto-detected at launch
                    ("dbos" for PostgreSQL, None for SQLite). Pass None explicitly
                    to force no schema even on PostgreSQL.
                state_table_name: State table name. Default "workflow_state".
                journal_table_name: Journal table name. Default "workflow_journal".
        """
        super().__init__()
        self.config: DBOSRuntimeConfig = dict(kwargs)  # type: ignore[assignment]

        # Workflow tracking state
        self._tracked_workflows: list[Workflow] = []
        self._tracked_workflow_ids: set[int] = set()  # Track by id for dedup
        self._registered: dict[int, RegisteredWorkflow] = {}  # keyed by id(workflow)

        self._dbos_launched = False
        self._tasks: list[asyncio.Task[None]] = []
        self._sql_engine: Engine | None = None
        self._migrations_run = False

        # Native driver resources (resolved at launch time)
        self._pool: asyncpg.Pool | None = None
        self._dsn: str | None = None  # asyncpg DSN for lazy pool creation
        self._db_path: str | None = None  # sqlite path
        self._schema: str | None = None
        self._workflow_store: AbstractWorkflowStore | None = None

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
            self._registered[id(workflow)] = registered
        else:
            wf_id = id(workflow)
            if wf_id not in self._tracked_workflow_ids:
                self._tracked_workflows.append(workflow)
                self._tracked_workflow_ids.add(wf_id)

    def get_registered(self, workflow: Workflow) -> RegisteredWorkflow | None:
        """Get the registered workflow if available."""
        return self._registered.get(id(workflow))

    def register(self, workflow: Workflow) -> RegisteredWorkflow:
        """
        Wrap workflow with DBOS decorators.

        Called at launch() time for each tracked workflow.
        Uses workflow.workflow_name for stable DBOS registration names.
        Idempotent: returns existing registration if already registered.
        """
        # Return existing registration if already registered
        existing = self._registered.get(id(workflow))
        if existing is not None:
            return existing

        # Use workflow's name directly
        name = workflow.workflow_name

        # Create DBOS-wrapped control loop with stable name
        @DBOS.workflow(name=f"{name}.control_loop")
        async def _dbos_control_loop(
            init_state: BrokerState,
            start_event: StartEvent | None = None,
            tags: dict[str, Any] = {},
        ) -> StopEvent:
            # Eagerly resolve the asyncpg pool so the adapter can use it
            # synchronously in get_state_store / is_replaying.
            if self._dsn is not None:
                await self._ensure_pool()
            workflow_run_fn = create_workflow_run_function(workflow)
            return await workflow_run_fn(init_state, start_event, tags)

        # Wrap steps with stable names
        wrapped_steps: dict[str, StepWorkerFunction] = {
            step_name: DBOS.step(name=f"{name}.{step_name}")(step)
            for step_name, step in as_step_worker_functions(workflow).items()
        }

        registered = RegisteredWorkflow(
            workflow=workflow, workflow_run_fn=_dbos_control_loop, steps=wrapped_steps
        )
        self._registered[id(workflow)] = registered
        return registered

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

    async def _ensure_pool(self) -> asyncpg.Pool:
        """Get or lazily create the asyncpg connection pool.

        Only valid for postgres dialect. Raises RuntimeError for sqlite.
        """
        if self._pool is not None:
            return self._pool
        if self._dsn is None:
            raise RuntimeError(
                "No asyncpg DSN configured. Either not launched or using sqlite dialect."
            )
        self._pool = await asyncpg.create_pool(dsn=self._dsn)
        return self._pool

    def run_migrations(self) -> None:
        """Run database migrations for all workflow tables.

        Uses the file-based migration system to create/update workflow store,
        state, and journal tables. Idempotent - safe to call multiple times.

        Can be called explicitly before launch() when run_migrations_on_launch=False,
        allowing for custom migration timing (e.g., during application startup).

        Requires DBOS to be launched first (calls _get_sql_engine internally).
        """
        if self._migrations_run:
            return

        engine = self._get_sql_engine()
        schema = _resolve_schema(self.config, engine)

        if engine.dialect.name == "postgresql":
            dsn = _sqlalchemy_url_to_asyncpg_dsn(engine.url)
            PostgresWorkflowStore.run_migrations_sync(dsn, schema=schema)
        else:
            db_path = str(engine.url.database) if engine.url.database else ":memory:"
            SqliteWorkflowStore.run_migrations(db_path)

        self._migrations_run = True
        logger.info("Database migrations completed")

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

        Args:
            run_id: Unique identifier for this workflow run.
            workflow: The workflow to run.
            init_state: Initial broker state for the control loop.
            start_event: Optional start event to kick off the workflow.
            serialized_state: Optional pre-populated state from InMemoryStateStore.to_dict().
                If provided, this state is written to the database before the workflow
                starts, allowing workflows to begin with pre-set initial values.
            serializer: Serializer for state data. Defaults to JsonSerializer.
            adapter_state: Optional adapter state (unused for DBOS).
        """
        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before running workflows."
            )

        registered = self.get_registered(workflow)
        if registered is None:
            raise RuntimeError(
                "DBOSRuntime workflows must be registered before running. Did you forget to call runtime.launch()?"
            )

        # Capture values needed in the async task closure
        active_serializer = serializer or JsonSerializer()

        async def _run_workflow() -> None:
            with SetWorkflowID(run_id):
                # Write initial state to DB before starting workflow (non-blocking to caller)
                if serialized_state:
                    if self._dsn is not None:
                        pool = await self._ensure_pool()
                        store: StateStore[Any] = PostgresStateStore(
                            pool=pool,
                            run_id=run_id,
                            state_type=infer_state_type(workflow),
                            serializer=active_serializer,
                            schema=self._schema,
                        )
                    elif self._db_path is not None:
                        store = SqliteStateStore(
                            db_path=self._db_path,
                            run_id=run_id,
                            state_type=infer_state_type(workflow),
                            serializer=active_serializer,
                        )
                    else:
                        raise RuntimeError("No pool or db_path configured.")
                    # Deserialize and save the initial state
                    state = deserialize_state_from_dict(
                        serialized_state,
                        active_serializer,
                        state_type=infer_state_type(workflow),
                    )
                    await store.set_state(state)

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

        # Create startup task and pass to adapter so it can await workflow readiness
        startup_task = asyncio.create_task(_run_workflow())
        self._track_task(startup_task)

        return ExternalDBOSAdapter(
            run_id,
            self.config.get("polling_interval_sec", 1.0),
            startup_task,
        )

    def get_internal_adapter(self, workflow: Workflow) -> InternalRunAdapter:
        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before running workflows."
            )
        run_id = DBOS.workflow_id
        if run_id is None:
            raise RuntimeError(
                "No current run id. Must be called within a workflow run."
            )

        # Infer state_type from the workflow for typed state support
        state_type = infer_state_type(workflow)

        engine = self._get_sql_engine()
        return InternalDBOSAdapter(
            run_id,
            engine,
            state_type,
            schema=self._schema,
            state_table_name=self.config.get(
                "state_table_name", DEFAULT_STATE_TABLE_NAME
            ),
            journal_table_name=self.config.get(
                "journal_table_name", DEFAULT_JOURNAL_TABLE_NAME
            ),
            ensure_pool=self._ensure_pool if self._dsn is not None else None,
            pool=self._pool,
            db_path=self._db_path,
        )

    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before running workflows."
            )
        return ExternalDBOSAdapter(run_id, self.config.get("polling_interval_sec", 1.0))

    def create_workflow_store(self) -> AbstractWorkflowStore:
        """Return the cached workflow store, creating it on first call.

        Detects the engine dialect and creates the appropriate store:
        - PostgreSQL: PostgresWorkflowStore using asyncpg with LISTEN/NOTIFY
        - SQLite: SqliteWorkflowStore using raw sqlite3

        Must be called after launch(). The store is cached and closed on
        destroy().
        """
        if self._workflow_store is not None:
            return self._workflow_store

        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before creating workflow store."
            )

        engine = self._get_sql_engine()
        schema = _resolve_schema(self.config, engine)

        if engine.dialect.name == "postgresql":
            dsn = _sqlalchemy_url_to_asyncpg_dsn(engine.url)
            logger.info("Using PostgresWorkflowStore (asyncpg) for workflow storage")
            self._workflow_store = PostgresWorkflowStore(
                dsn=dsn,
                schema=schema,
            )
            return self._workflow_store

        # SQLite — use SqliteWorkflowStore
        db_path = str(engine.url.database) if engine.url.database else ":memory:"
        logger.info("Using SqliteWorkflowStore for workflow storage")
        self._workflow_store = SqliteWorkflowStore(db_path=db_path, auto_migrate=False)
        return self._workflow_store

    def build_server_runtime(self) -> Runtime:
        """Build the decorator chain for use with WorkflowServer.

        Wraps the DBOS runtime with:
        - EventInterceptorDecorator (blocks events from reaching DBOS streams)

        DBOS handles persistence and resumption internally, and idle detection
        is not supported (would require cancelling and resuming a new workflow).

        The returned runtime should be passed as the ``runtime`` argument
        to ``WorkflowServer``.
        """
        return EventInterceptorDecorator(self)

    def launch(self) -> None:
        """
        Launch DBOS and register all tracked workflows.

        Must be called before running any workflows.
        Runs database migrations unless run_migrations_on_launch=False.
        """
        if self._dbos_launched:
            return  # Already launched

        # Register each pending workflow with DBOS
        for workflow in self._tracked_workflows:
            # Register with DBOS (this applies decorators)
            registered = self.register(workflow)
            self._registered[id(workflow)] = registered

        # Launch DBOS runtime
        DBOS.launch()
        self._dbos_launched = True

        # Resolve native driver config from SQLAlchemy engine
        engine = self._get_sql_engine()
        self._schema = _resolve_schema(self.config, engine)
        if engine.dialect.name == "postgresql":
            self._dsn = _sqlalchemy_url_to_asyncpg_dsn(engine.url)
        else:
            self._db_path = (
                str(engine.url.database) if engine.url.database else ":memory:"
            )

        # Run migrations after DBOS is launched (if configured)
        if self.config.get("run_migrations_on_launch", True):
            self.run_migrations()

    def destroy(self, destroy_dbos: bool = True) -> None:
        """Clean up DBOS runtime resources.

        Args:
            destroy_dbos: If True (default), also calls DBOS.destroy().
                Set to False when DBOS lifecycle is managed externally
                (e.g., shared across multiple runtimes in tests).
        """
        self._tracked_workflows.clear()
        self._tracked_workflow_ids.clear()
        self._registered.clear()
        self._dbos_launched = False
        self._sql_engine = None
        self._migrations_run = False
        self._dsn = None
        self._db_path = None
        self._schema = None
        if self._pool is not None:
            try:
                self._pool.terminate()
            except Exception:
                logger.debug(
                    "Failed to terminate asyncpg pool during destroy", exc_info=True
                )
            self._pool = None
        if self._workflow_store is not None:
            if isinstance(self._workflow_store, PostgresWorkflowStore):
                pool = self._workflow_store._pool
                if pool is not None:
                    try:
                        pool.terminate()
                    except Exception:
                        logger.debug(
                            "Failed to terminate workflow store pool during destroy",
                            exc_info=True,
                        )
                self._workflow_store._pool = None
                self._workflow_store._listen_conn = None
            self._workflow_store = None
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if destroy_dbos:
            DBOS.destroy()


EnsurePoolFn = Callable[[], Awaitable[asyncpg.Pool]]

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
    - wait_for_next_task coordinates task completion ordering for deterministic replay
    """

    def __init__(
        self,
        run_id: str,
        engine: Engine,
        state_type: type[BaseModel] | None = None,
        schema: str | None = None,
        state_table_name: str = DEFAULT_STATE_TABLE_NAME,
        journal_table_name: str = DEFAULT_JOURNAL_TABLE_NAME,
        ensure_pool: EnsurePoolFn | None = None,
        pool: asyncpg.Pool | None = None,
        db_path: str | None = None,
    ) -> None:
        self._run_id = run_id
        self._engine = engine
        self._state_type = state_type
        self._schema = schema
        self._state_table_name = state_table_name
        self._journal_table_name = journal_table_name
        self._ensure_pool = ensure_pool
        self._resolved_pool: asyncpg.Pool | None = pool
        self._db_path = db_path
        self._closed = False
        self._state_store: StateStore[Any] | None = None
        # Journal for deterministic task ordering - lazily initialized
        self._journal: TaskJournal | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    async def write_to_event_stream(self, event: Event) -> None:
        """Record events to the workflow store, skipping duplicates on replay.

        During replay, non-terminal events are skipped to avoid duplicates
        (they were already persisted in the original run). Terminal events
        are always written because the TaskJournal records step completion
        *before* events are published — if a crash occurs between the journal
        write and the event write, the terminal event was never persisted and
        must be re-emitted so that ``subscribe_events()`` can detect workflow
        completion. Status updates (``_handle_status_update``) are idempotent,
        so writing them again is safe; a duplicate ``append_event`` for a
        terminal event is harmless.

        The inner adapter forwarding always happens so that runtime-level
        concerns (e.g. idle detection, DBOS stream) still function during
        replay.
        """
        await DBOS.write_stream_async(_IO_STREAM_PUBLISHED_EVENTS_NAME, event)

    async def get_now(self) -> float:
        return _durable_time()

    async def send_event(self, tick: WorkflowTick) -> None:
        # Use run_in_executor to escape DBOS step context.
        # DBOS yells at you for writing to the event stream from a step (since is not idempotent)
        # However that's the expected semantics of llama index workflow steps, so it's ok.
        loop = asyncio.get_running_loop()
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

        # Timeout 1x per day at least. This will just cause a wakeup loop of the control loop.
        result = await DBOS.recv_async(
            _IO_STREAM_TICK_TOPIC,
            timeout_seconds=timeout_seconds or _UNBOUNDED_WAIT_TIMEOUT_SECONDS,
        )
        if result is None:
            return WaitResultTimeout()
        if isinstance(result, _DBOSInternalShutdown):
            self._closed = True
            raise asyncio.CancelledError("Adapter closed")
        return WaitResultTick(tick=result)

    async def close(self) -> None:
        """Signal shutdown by sending internal message to wake blocked recv."""
        if self._closed:
            return
        self._closed = True

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: DBOS.send(
                self._run_id, _DBOSInternalShutdown(), topic=_IO_STREAM_TICK_TOPIC
            ),
        )

    async def _resolve_pool(self) -> asyncpg.Pool:
        """Resolve the asyncpg pool, lazily creating it via the runtime callback."""
        if self._resolved_pool is not None:
            return self._resolved_pool
        if self._ensure_pool is None:
            raise RuntimeError(
                "No asyncpg pool configured. Either not launched or using sqlite dialect."
            )
        self._resolved_pool = await self._ensure_pool()
        return self._resolved_pool

    def _get_or_create_state_store(self) -> StateStore[Any]:
        """Get or lazily create the state store.

        For PostgreSQL, the pool must be resolved first via _resolve_pool().
        Call _ensure_resources() before accessing the state store.
        """
        if self._state_store is None:
            if self._resolved_pool is not None:
                self._state_store = PostgresStateStore(
                    pool=self._resolved_pool,
                    run_id=self._run_id,
                    state_type=cast(type[Any], self._state_type),
                    schema=self._schema,
                )
            elif self._db_path is not None:
                self._state_store = SqliteStateStore(
                    db_path=self._db_path,
                    run_id=self._run_id,
                    state_type=cast(type[Any], self._state_type),
                )
            else:
                raise RuntimeError(
                    "No pool or db_path configured for state store. "
                    "Ensure the runtime pool is initialized before accessing state."
                )
        return self._state_store

    def get_state_store(self) -> StateStore[Any] | None:
        return self._get_or_create_state_store()

    def is_replaying(self) -> bool:
        if (
            self._journal is None
            and self._resolved_pool is None
            and self._db_path is None
        ):
            return False
        journal = self._get_or_create_journal()
        return journal.is_replaying()

    def _get_or_create_journal(self) -> TaskJournal:
        """Get or lazily create the task journal."""
        if self._journal is None:
            if self._resolved_pool is not None:
                crud = PostgresJournalCrud(
                    pool=self._resolved_pool,
                    table_name=self._journal_table_name,
                    schema=self._schema,
                )
            elif self._db_path is not None:
                crud = SqliteJournalCrud(
                    db_path=self._db_path,
                    table_name=self._journal_table_name,
                )
            else:
                raise RuntimeError("No pool or db_path configured for journal.")
            self._journal = TaskJournal(self._run_id, crud)
        return self._journal

    async def wait_for_next_task(
        self,
        task_set: list[NamedTask],
        timeout: float | None = None,
    ) -> asyncio.Task[Any] | None:
        """Wait for and return the next task that should complete.

        During replay, waits for the specific task that completed in the original run.
        During fresh execution, waits for any task and records the completion order.

        **Journal ordering caveat:** The journal records task completion *before*
        the control loop publishes events produced by the completed step. If a
        crash occurs between the journal write and the event write, replay skips
        the step (already journaled) but the event was never persisted.

        - In the standalone DBOS path, ``write_to_event_stream`` publishes to
          DBOS streams which are separately journaled, so the gap is less of an
          issue.
        - In the server path, ``_ServerInternalRunAdapter`` makes event writes
          idempotent by comparing against already-persisted events, working
          around the ordering gap.
        - Self-publishing events (``InputRequiredEvent`` / ``HumanInputRequired``
          subtypes) are affected: if a crash occurs after the journal records the
          step but before the event is published, the event is lost on replay.
        - Proper fix: defer journal recording until after all commands from a
          tick are processed.

        Args:
            task_set: List of NamedTasks with stable string keys for identification
            timeout: Timeout in seconds, None for no timeout

        Returns:
            The completed task, or None on timeout.
        """
        tasks = NamedTask.all_tasks(task_set)
        if not tasks:
            return None

        # Ensure pool is resolved before journal creation (needed for postgres)
        if self._ensure_pool is not None and self._resolved_pool is None:
            await self._resolve_pool()

        journal = self._get_or_create_journal()
        await journal.load()

        expected_key = journal.next_expected_key()
        if expected_key is not None:
            # Replay mode: wait for specific task
            target_task = NamedTask.find_by_key(task_set, expected_key)

            if target_task is None:
                logger.warning(
                    f"Non-deterministic execution detected during replay! "
                    f"Expected task {expected_key} not in set yet. "
                    f"Falling back to awaiting all tasks."
                )
            else:
                try:
                    await asyncio.wait_for(asyncio.shield(target_task), timeout=timeout)
                except (asyncio.TimeoutError, TimeoutError):
                    return None
                journal.advance()
                return target_task

        # Fresh execution: wait for first, record it
        done, _ = await asyncio.wait(
            tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            return None

        completed = done.pop()
        key = NamedTask.get_key(task_set, completed)
        await journal.record(key)

        return completed


class ExternalDBOSAdapter(ExternalRunAdapter):
    """
    External DBOS adapter for workflow interaction.

    - send_event puts ticks into the shared mailbox queue
    - stream_published_events reads from DBOS streams
    - close is a no-op
    """

    def __init__(
        self,
        run_id: str,
        polling_interval_sec: float = 1.0,
        startup_task: asyncio.Task[None] | None = None,
    ) -> None:
        self._run_id = run_id
        self._polling_interval_sec = polling_interval_sec
        self._startup_task = startup_task  # None means workflow already started

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
        """Wait for the workflow startup task to complete if one was provided."""
        if self._startup_task is not None:
            await self._startup_task
            self._startup_task = None  # Clear after awaiting
