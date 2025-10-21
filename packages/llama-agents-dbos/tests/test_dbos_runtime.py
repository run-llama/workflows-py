# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""DBOS-specific runtime tests for adapter behavior.

These tests focus on the internal mechanics of the DBOS adapter,
particularly around run_id matching and state store availability.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Any, Generator, cast
from unittest.mock import patch

import pytest
from dbos import DBOS, DBOSConfig
from llama_agents.dbos import DBOSRuntime
from llama_agents.dbos.journal.task_journal import TaskJournal
from llama_agents.dbos.runtime import InternalDBOSAdapter
from llama_agents.dbos.state_store import (
    SqlSerializedState,
    SqlStateStore,
    parse_serialized_state,
)
from pydantic import Field
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from workflows.context import Context
from workflows.context.state_store import InMemorySerializedState
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.runtime.types.named_task import NamedTask
from workflows.testing import WorkflowTestRunner
from workflows.workflow import Workflow


@pytest.fixture(scope="module")
def dbos_config(tmp_path_factory: pytest.TempPathFactory) -> DBOSConfig:
    """Create DBOS config with a fresh SQLite database."""
    db_file = tmp_path_factory.mktemp("dbos") / "dbos_debug_test.sqlite3"
    system_db_url = f"sqlite+pysqlite:///{db_file}?check_same_thread=false"
    return {
        "name": "workflows-dbos-debug",
        "system_database_url": system_db_url,
        "run_admin_server": False,
    }  # type: ignore[return-value]


@pytest.fixture(scope="module")
def dbos_runtime(dbos_config: DBOSConfig) -> Generator[DBOSRuntime, None, None]:
    """Module-scoped DBOS runtime with fast polling for tests."""
    DBOS(config=dbos_config)
    runtime = DBOSRuntime(polling_interval_sec=0.01)
    try:
        yield runtime
    finally:
        runtime.destroy()


class DebugEvent(Event):
    captured_run_id: str = Field(default="")
    captured_dbos_workflow_id: str = Field(default="")
    state_store_available: bool = Field(default=False)


class RunIdCaptureWorkflow(Workflow):
    """Workflow that captures run_id info for debugging."""

    @step
    async def capture_ids(self, ev: StartEvent) -> StopEvent:
        dbos_workflow_id = DBOS.workflow_id or "None"
        return StopEvent(result={"dbos_workflow_id": dbos_workflow_id})


class StateStoreAccessWorkflow(Workflow):
    """Workflow that attempts to access state store."""

    @step
    async def access_store(self, ctx: Context, ev: StartEvent) -> StopEvent:
        dbos_workflow_id = DBOS.workflow_id or "None"

        try:
            await ctx.store.set("test_key", "test_value")
            value = await ctx.store.get("test_key")
            store_works = value == "test_value"
        except Exception as e:
            return StopEvent(
                result={
                    "dbos_workflow_id": dbos_workflow_id,
                    "store_works": False,
                    "error": str(e),
                }
            )

        return StopEvent(
            result={
                "dbos_workflow_id": dbos_workflow_id,
                "store_works": store_works,
            }
        )


class StateStoreCounterWorkflow(Workflow):
    """Workflow that increments a counter in state store."""

    @step
    async def increment(self, ctx: Context, ev: StartEvent) -> StopEvent:
        cur = await ctx.store.get("counter", default=0)
        await ctx.store.set("counter", cur + 1)
        return StopEvent(result=cur + 1)


@pytest.mark.asyncio
async def test_dbos_workflow_id_available(dbos_runtime: DBOSRuntime) -> None:
    """Verify DBOS.workflow_id is set inside workflow execution."""
    wf = RunIdCaptureWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    r = await WorkflowTestRunner(wf).run()
    result = r.result

    assert result["dbos_workflow_id"] != "None", (
        "DBOS.workflow_id should be set inside workflow"
    )


@pytest.mark.asyncio
async def test_state_store_access_in_step(dbos_runtime: DBOSRuntime) -> None:
    """Test whether state store is accessible inside a workflow step."""
    wf = StateStoreAccessWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    r = await WorkflowTestRunner(wf).run()
    result = r.result

    assert result["store_works"], (
        f"State store should be accessible. Got error: {result.get('error', 'unknown')}"
    )


@pytest.mark.asyncio
async def test_internal_adapter_run_id_matches(dbos_runtime: DBOSRuntime) -> None:
    """Verify internal adapter run_id matches DBOS.workflow_id."""
    captured_ids: dict[str, Any] = {}

    class IdTracingWorkflow(Workflow):
        @step
        async def trace_ids(self, ev: StartEvent) -> StopEvent:
            captured_ids["dbos_workflow_id"] = DBOS.workflow_id

            internal_adapter = dbos_runtime.get_internal_adapter(self)
            captured_ids["adapter_run_id"] = internal_adapter.run_id

            store = internal_adapter.get_state_store()
            captured_ids["state_store_found"] = store is not None

            return StopEvent(result="done")

    wf = IdTracingWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    await WorkflowTestRunner(wf).run()

    assert captured_ids["adapter_run_id"] == captured_ids["dbos_workflow_id"], (
        f"Adapter run_id '{captured_ids['adapter_run_id']}' should match "
        f"DBOS.workflow_id '{captured_ids['dbos_workflow_id']}'"
    )
    assert captured_ids["state_store_found"], "State store should be available"


@pytest.mark.asyncio
async def test_external_run_id_vs_internal(dbos_runtime: DBOSRuntime) -> None:
    """Compare external adapter run_id with what's seen internally."""
    internal_run_id: str | None = None

    class CompareWorkflow(Workflow):
        @step
        async def capture(self, ev: StartEvent) -> StopEvent:
            nonlocal internal_run_id
            internal_run_id = DBOS.workflow_id
            return StopEvent(result="done")

    wf = CompareWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    handler = wf.run()
    external_run_id = handler.run_id

    await handler

    assert external_run_id == internal_run_id, (
        f"External run_id '{external_run_id}' should match "
        f"internal DBOS.workflow_id '{internal_run_id}'"
    )


@pytest.mark.asyncio
async def test_state_store_lazy_creation(dbos_runtime: DBOSRuntime) -> None:
    """Test that state store is lazily created by the internal adapter."""
    store_info: dict[str, Any] = {}

    class LazyStoreWorkflow(Workflow):
        @step
        async def check_store(self, ctx: Context, ev: StartEvent) -> StopEvent:
            internal_adapter = dbos_runtime.get_internal_adapter(self)

            # First call should create the store
            store1 = internal_adapter.get_state_store()
            store_info["first_store_id"] = id(store1)
            store_info["first_store_exists"] = store1 is not None

            # Second call should return the same store
            store2 = internal_adapter.get_state_store()
            store_info["second_store_id"] = id(store2)
            store_info["same_store"] = store1 is store2

            # Store should work
            await ctx.store.set("lazy_key", "lazy_value")
            value = await ctx.store.get("lazy_key")
            store_info["store_works"] = value == "lazy_value"

            return StopEvent(result="done")

    wf = LazyStoreWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    await WorkflowTestRunner(wf).run()

    assert store_info["first_store_exists"], "Store should be created on first access"
    assert store_info["same_store"], "Same store instance should be returned"
    assert store_info["store_works"], "Store should be functional"


@pytest.mark.asyncio
async def test_run_workflow_does_not_create_store(dbos_runtime: DBOSRuntime) -> None:
    """Verify run_workflow doesn't eagerly create a state store."""
    call_log: list[dict[str, Any]] = []
    original_run_workflow = dbos_runtime.run_workflow

    def patched_run_workflow(*args: Any, **kwargs: Any) -> Any:
        call_log.append({"run_id": kwargs.get("run_id")})
        return original_run_workflow(*args, **kwargs)

    class SimpleWf(Workflow):
        @step
        async def do_it(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

    wf = SimpleWf(runtime=dbos_runtime)
    dbos_runtime.launch()

    with patch.object(dbos_runtime, "run_workflow", patched_run_workflow):
        handler = wf.run()
        await handler

    assert len(call_log) == 1, "run_workflow should be called exactly once"


@pytest.mark.asyncio
async def test_replay_wait_for_next_task_timeout_returns_none(
    sqlite_engine: Engine,
) -> None:
    """Replay wait timeout should return None and not raise."""
    run_id = "replay-timeout-run"

    journal = TaskJournal(run_id, sqlite_engine)
    await journal.load()
    await journal.record("step_a:0")

    adapter = InternalDBOSAdapter(run_id=run_id, engine=sqlite_engine)
    task = asyncio.create_task(asyncio.sleep(5.0))

    try:
        result = await adapter.wait_for_next_task(
            [NamedTask.worker("step_a", 0, task)],
            timeout=0.01,
        )
        assert result is None
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


# ============================================================================
# SqlSerializedState and parse_serialized_state Tests
# ============================================================================


def test_parse_serialized_state_sql_store_type() -> None:
    """Test that store_type='sql' parses as SqlSerializedState."""
    serialized = {
        "store_type": "sql",
        "run_id": "run-12345",
        "schema": "public",
    }

    result = parse_serialized_state(serialized)

    assert isinstance(result, SqlSerializedState)
    assert result.store_type == "sql"
    assert result.run_id == "run-12345"
    assert result.db_schema == "public"


def test_parse_serialized_state_sql_with_null_schema() -> None:
    """Test that SqlSerializedState accepts null schema."""
    serialized = {
        "store_type": "sql",
        "run_id": "run-67890",
        "schema": None,
    }

    result = parse_serialized_state(serialized)

    assert isinstance(result, SqlSerializedState)
    assert result.db_schema is None


def test_parse_serialized_state_in_memory_format() -> None:
    """Test that in_memory format is still handled."""
    serialized = {
        "store_type": "in_memory",
        "state_type": "DictState",
        "state_module": "workflows.context.state_store",
        "state_data": {"_data": {"counter": 42}},
    }

    result = parse_serialized_state(serialized)

    assert isinstance(result, InMemorySerializedState)
    assert result.store_type == "in_memory"


def test_parse_serialized_state_unknown_store_type_raises() -> None:
    """Test that unknown store_type raises ValueError."""
    serialized = {
        "store_type": "redis",  # Unknown store type
        "state_type": "SomeState",
        "state_module": "some.module",
    }

    with pytest.raises(ValueError, match="Unknown store_type"):
        parse_serialized_state(serialized)


@pytest.mark.asyncio
async def test_edit_state_rolls_back_and_closes_on_error(sqlite_engine: Engine) -> None:
    """Errors inside edit_state should rollback/close and leave store usable."""
    store = SqlStateStore(run_id="state-run", engine=sqlite_engine)

    with pytest.raises(RuntimeError, match="boom"):
        async with store.edit_state() as state:
            state["transient"] = "value"
            raise RuntimeError("boom")

    assert await store.get("transient", default=None) is None

    await store.set("after", "ok")
    assert await store.get("after") == "ok"

    with sqlite_engine.connect() as conn:
        raw = conn.connection
        assert not bool(getattr(raw, "in_transaction", False))


@pytest.mark.asyncio
async def test_edit_state_failure_releases_checked_out_connection(
    tmp_path: Path,
) -> None:
    """Failed edit_state should not leak checked-out pooled connections."""
    db_file = tmp_path / "state.sqlite3"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
    )
    store = SqlStateStore(run_id="pool-run", engine=engine)
    pool = cast(QueuePool, engine.pool)

    assert pool.checkedout() == 0

    with pytest.raises(ValueError, match="force failure"):
        async with store.edit_state() as state:
            state["x"] = 1
            raise ValueError("force failure")

    assert pool.checkedout() == 0

    await store.set("y", 2)
    assert await store.get("y") == 2
    assert pool.checkedout() == 0
