"""DBOS-specific runtime tests for debugging adapter behavior.

These tests focus on the internal mechanics of the DBOS adapter,
particularly around run_id matching and state store availability.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Generator

import pytest
from conftest import make_test_dbos_config, make_test_dbos_postgres_config
from dbos import DBOS, DBOSConfig
from llama_agents.runtime.dbos import (
    DBOSRuntime,
    _dbos_state_stores,
)
from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.testing import WorkflowTestRunner
from workflows.workflow import Workflow

# -- Fixtures --


@pytest.fixture(scope="module")
def dbos_config(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[DBOSConfig, None, None]:
    """Create DBOS config with a fresh database.

    Uses PostgreSQL if DBOS_TEST_POSTGRES=1 or test is marked with @pytest.mark.postgres,
    otherwise uses SQLite (default).
    """
    use_postgres = (
        os.environ.get("DBOS_TEST_POSTGRES", "").lower() in ("1", "true", "yes")
        or request.node.get_closest_marker("postgres") is not None
    )

    if use_postgres:
        from py_pglite import PGliteManager

        manager = PGliteManager()
        manager.start()
        try:
            yield make_test_dbos_postgres_config("workflows-py-dbos-debug", manager)
        finally:
            manager.stop()
    else:
        db_file = tmp_path_factory.mktemp("dbos") / "dbos_debug_test.sqlite3"
        yield make_test_dbos_config("workflows-py-dbos-debug", db_file)


@pytest.fixture(scope="module")
def dbos_runtime(dbos_config: DBOSConfig) -> Generator[DBOSRuntime, None, None]:
    """Module-scoped DBOS runtime with fast polling for tests."""
    DBOS(config=dbos_config)
    runtime = DBOSRuntime(polling_interval_sec=0.01)
    try:
        yield runtime
    finally:
        runtime.destroy()


# -- Test Events --


class DebugEvent(Event):
    captured_run_id: str = Field(default="")
    captured_dbos_workflow_id: str = Field(default="")
    state_store_available: bool = Field(default=False)
    state_store_keys: list[str] = Field(default_factory=list)


# -- Test Workflows --


class RunIdCaptureWorkflow(Workflow):
    """Workflow that captures run_id info for debugging."""

    @step
    async def capture_ids(self, ev: StartEvent) -> StopEvent:
        # Capture what DBOS thinks the workflow_id is
        dbos_workflow_id = DBOS.workflow_id or "None"

        # Check what's in the state store dictionary
        state_store_keys = list(_dbos_state_stores.keys())

        return StopEvent(
            result={
                "dbos_workflow_id": dbos_workflow_id,
                "state_store_keys": state_store_keys,
            }
        )


class StateStoreAccessWorkflow(Workflow):
    """Workflow that attempts to access state store."""

    @step
    async def access_store(self, ctx: Context, ev: StartEvent) -> StopEvent:
        dbos_workflow_id = DBOS.workflow_id or "None"
        state_store_keys = list(_dbos_state_stores.keys())

        # Try to access the state store
        try:
            await ctx.store.set("test_key", "test_value")
            value = await ctx.store.get("test_key")
            store_works = value == "test_value"
        except Exception as e:
            store_works = False
            error_msg = str(e)
            return StopEvent(
                result={
                    "dbos_workflow_id": dbos_workflow_id,
                    "state_store_keys": state_store_keys,
                    "store_works": store_works,
                    "error": error_msg,
                }
            )

        return StopEvent(
            result={
                "dbos_workflow_id": dbos_workflow_id,
                "state_store_keys": state_store_keys,
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


# -- Tests --


@pytest.mark.asyncio
async def test_dbos_workflow_id_available(dbos_runtime: DBOSRuntime) -> None:
    """Verify DBOS.workflow_id is set inside workflow execution."""
    wf = RunIdCaptureWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    r = await WorkflowTestRunner(wf).run()
    result = r.result

    # DBOS should have assigned a workflow_id
    assert result["dbos_workflow_id"] != "None", (
        "DBOS.workflow_id should be set inside workflow"
    )
    print(f"\nDBOS workflow_id inside workflow: {result['dbos_workflow_id']}")
    print(f"State store keys at execution time: {result['state_store_keys']}")


@pytest.mark.asyncio
async def test_run_id_matches_state_store_key(dbos_runtime: DBOSRuntime) -> None:
    """Verify the run_id used for state store matches DBOS.workflow_id."""
    wf = RunIdCaptureWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    # Clear the state stores dict to get a clean view
    _dbos_state_stores.clear()

    handler = wf.run()

    # Give a moment for run_workflow to execute
    await asyncio.sleep(0.1)

    # Capture state store keys right after workflow starts
    keys_after_start = list(_dbos_state_stores.keys())
    print(f"\nState store keys after workflow start: {keys_after_start}")

    # Get the result
    result = await handler
    result_data = result.result if hasattr(result, "result") else result

    dbos_wf_id = result_data["dbos_workflow_id"]
    keys_during_execution = result_data["state_store_keys"]

    print(f"DBOS.workflow_id during execution: {dbos_wf_id}")
    print(f"State store keys during execution: {keys_during_execution}")

    # The key stored should match what DBOS reports as workflow_id
    if keys_after_start:
        stored_key = keys_after_start[0]
        print(f"Stored key: {stored_key}")
        print(f"Keys match: {stored_key == dbos_wf_id}")


@pytest.mark.asyncio
async def test_state_store_access_in_step(dbos_runtime: DBOSRuntime) -> None:
    """Test whether state store is accessible inside a workflow step."""
    wf = StateStoreAccessWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    r = await WorkflowTestRunner(wf).run()
    result = r.result

    print(f"\nDBOS workflow_id: {result['dbos_workflow_id']}")
    print(f"State store keys: {result['state_store_keys']}")
    print(f"Store works: {result['store_works']}")
    if "error" in result:
        print(f"Error: {result['error']}")

    # This is the assertion that currently fails
    assert result["store_works"], (
        f"State store should be accessible. Got error: {result.get('error', 'unknown')}"
    )


@pytest.mark.asyncio
async def test_internal_adapter_run_id_source(dbos_runtime: DBOSRuntime) -> None:
    """Debug test to trace where run_ids come from."""

    captured_ids: dict[str, Any] = {}

    class IdTracingWorkflow(Workflow):
        @step
        async def trace_ids(self, ev: StartEvent) -> StopEvent:
            # What DBOS thinks
            captured_ids["dbos_workflow_id"] = DBOS.workflow_id

            # What the runtime returns
            internal_adapter = dbos_runtime.get_internal_adapter()
            captured_ids["adapter_run_id"] = internal_adapter.run_id

            # What's in the state store dict
            captured_ids["state_store_keys"] = list(_dbos_state_stores.keys())

            # Try to get state store
            store = internal_adapter.get_state_store()
            captured_ids["state_store_found"] = store is not None

            return StopEvent(result="done")

    wf = IdTracingWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    await WorkflowTestRunner(wf).run()

    print("\nCaptured IDs:")
    for k, v in captured_ids.items():
        print(f"  {k}: {v}")

    # The adapter's run_id should match a key in state_store_keys
    assert captured_ids["adapter_run_id"] in captured_ids["state_store_keys"], (
        f"Adapter run_id '{captured_ids['adapter_run_id']}' not found in "
        f"state_store_keys {captured_ids['state_store_keys']}"
    )


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

    # Get the handler to access the external adapter's run_id
    handler = wf.run()
    external_run_id = handler.run_id

    await handler

    print(f"\nExternal run_id (from handler): {external_run_id}")
    print(f"Internal run_id (DBOS.workflow_id): {internal_run_id}")
    print(f"Match: {external_run_id == internal_run_id}")

    assert external_run_id == internal_run_id, (
        f"External run_id '{external_run_id}' should match "
        f"internal DBOS.workflow_id '{internal_run_id}'"
    )


@pytest.mark.asyncio
async def test_state_store_lifecycle(dbos_runtime: DBOSRuntime) -> None:
    """Test state store availability through the workflow lifecycle."""

    lifecycle_events: list[dict[str, Any]] = []

    class LifecycleWorkflow(Workflow):
        @step
        async def check_store(self, ev: StartEvent) -> StopEvent:
            run_id = DBOS.workflow_id
            keys = list(_dbos_state_stores.keys())
            in_dict = run_id in keys if run_id else False

            lifecycle_events.append(
                {
                    "phase": "inside_step",
                    "dbos_workflow_id": run_id,
                    "state_store_keys": keys,
                    "run_id_in_dict": in_dict,
                }
            )

            return StopEvent(result="done")

    wf = LifecycleWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    # Clear for clean test
    _dbos_state_stores.clear()

    # Check before
    lifecycle_events.append(
        {
            "phase": "before_run",
            "state_store_keys": list(_dbos_state_stores.keys()),
        }
    )

    handler = wf.run()
    external_run_id = handler.run_id

    # Small delay for async startup
    await asyncio.sleep(0.05)

    lifecycle_events.append(
        {
            "phase": "after_run_called",
            "external_run_id": external_run_id,
            "state_store_keys": list(_dbos_state_stores.keys()),
        }
    )

    await handler

    lifecycle_events.append(
        {
            "phase": "after_complete",
            "state_store_keys": list(_dbos_state_stores.keys()),
        }
    )

    print("\nLifecycle events:")
    for event in lifecycle_events:
        print(f"  {event}")

    # Find the inside_step event
    inside_step = next(e for e in lifecycle_events if e["phase"] == "inside_step")

    # The external run_id should be in the state store when we store it
    after_run = next(e for e in lifecycle_events if e["phase"] == "after_run_called")
    assert external_run_id in after_run["state_store_keys"], (
        f"External run_id '{external_run_id}' should be stored. "
        f"Keys: {after_run['state_store_keys']}"
    )

    # And the internal DBOS.workflow_id should match
    assert inside_step["run_id_in_dict"], (
        f"DBOS.workflow_id '{inside_step['dbos_workflow_id']}' should be in "
        f"state_store_keys {inside_step['state_store_keys']}"
    )


@pytest.mark.asyncio
async def test_run_workflow_called(dbos_runtime: DBOSRuntime) -> None:
    """Verify run_workflow is actually being called and storing the state."""
    from unittest.mock import patch

    call_log: list[dict[str, Any]] = []
    original_run_workflow = dbos_runtime.run_workflow

    def patched_run_workflow(*args: Any, **kwargs: Any) -> Any:
        call_log.append(
            {
                "args": args,
                "kwargs_keys": list(kwargs.keys()),
                "run_id": kwargs.get("run_id"),
            }
        )
        print(f"\n!!! run_workflow called with run_id: {kwargs.get('run_id')}")
        result = original_run_workflow(*args, **kwargs)
        print(
            f"!!! State store keys after run_workflow: {list(_dbos_state_stores.keys())}"
        )
        return result

    class SimpleWf(Workflow):
        @step
        async def do_it(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

    wf = SimpleWf(runtime=dbos_runtime)
    dbos_runtime.launch()

    _dbos_state_stores.clear()

    with patch.object(dbos_runtime, "run_workflow", patched_run_workflow):
        handler = wf.run()
        print(f"\nHandler run_id: {handler.run_id}")
        print(
            f"State store keys immediately after run(): {list(_dbos_state_stores.keys())}"
        )
        await handler

    print(f"\nCall log: {call_log}")
    assert len(call_log) == 1, "run_workflow should be called exactly once"


@pytest.mark.asyncio
async def test_dict_identity_inside_vs_outside(dbos_runtime: DBOSRuntime) -> None:
    """Check if the dict seen inside vs outside workflow is the same object."""
    from workflows.context.state_store import DictState, InMemoryStateStore

    # Store something before workflow runs
    sentinel_store = InMemoryStateStore(DictState())
    _dbos_state_stores["sentinel"] = sentinel_store

    inside_info: dict[str, Any] = {}

    class DictIdentityWorkflow(Workflow):
        @step
        async def check_dict(self, ev: StartEvent) -> StopEvent:
            # Import fresh inside the step
            from llama_agents.runtime.dbos import _dbos_state_stores as inside_dict

            inside_info["dict_id"] = id(inside_dict)
            inside_info["dict_keys"] = list(inside_dict.keys())
            inside_info["sentinel_present"] = "sentinel" in inside_dict

            return StopEvent(result="done")

    wf = DictIdentityWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    outside_dict_id = id(_dbos_state_stores)
    outside_keys_before = list(_dbos_state_stores.keys())

    print(f"\nOutside dict id: {outside_dict_id}")
    print(f"Outside keys before run: {outside_keys_before}")

    handler = wf.run()
    await handler

    print(f"Inside dict id: {inside_info.get('dict_id')}")
    print(f"Inside dict keys: {inside_info.get('dict_keys')}")
    print(f"Sentinel present inside: {inside_info.get('sentinel_present')}")
    print(f"Same dict object: {outside_dict_id == inside_info.get('dict_id')}")

    # Cleanup
    del sentinel_store
    _dbos_state_stores.pop("sentinel", None)

    assert outside_dict_id == inside_info["dict_id"], "Should be same dict object"
    assert inside_info["sentinel_present"], "Sentinel should be visible inside workflow"


@pytest.mark.asyncio
async def test_state_store_gc_timing(dbos_runtime: DBOSRuntime) -> None:
    """Test if state store is GC'd between run_workflow and step execution."""
    import gc

    gc_events: list[str] = []
    inside_info: dict[str, Any] = {}

    class GCTimingWorkflow(Workflow):
        @step
        async def check_store(self, ev: StartEvent) -> StopEvent:
            run_id = DBOS.workflow_id
            keys = list(_dbos_state_stores.keys())
            in_dict = run_id in keys if run_id else False

            inside_info["dbos_workflow_id"] = run_id
            inside_info["keys"] = keys
            inside_info["in_dict"] = in_dict

            return StopEvent(result="done")

    wf = GCTimingWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    _dbos_state_stores.clear()

    # Monkey-patch run_workflow to trace GC
    original_run_workflow = DBOSRuntime.run_workflow

    def traced_run_workflow(self: DBOSRuntime, *args: Any, **kwargs: Any) -> Any:
        result = original_run_workflow(self, *args, **kwargs)
        gc_events.append(f"after_run_workflow: keys={list(_dbos_state_stores.keys())}")

        # Force GC to simulate worst-case
        gc.collect()
        gc_events.append(f"after_gc: keys={list(_dbos_state_stores.keys())}")

        return result

    DBOSRuntime.run_workflow = traced_run_workflow  # type: ignore

    try:
        handler = wf.run()
        gc_events.append(f"after_wf_run: keys={list(_dbos_state_stores.keys())}")

        # Force another GC
        gc.collect()
        gc_events.append(f"after_gc_2: keys={list(_dbos_state_stores.keys())}")

        await handler
    finally:
        DBOSRuntime.run_workflow = original_run_workflow  # type: ignore

    print("\nGC timing events:")
    for event in gc_events:
        print(f"  {event}")
    print(f"Inside workflow: {inside_info}")

    # The key should persist until the workflow completes
    assert inside_info["in_dict"], (
        f"Run ID '{inside_info['dbos_workflow_id']}' should be in dict. "
        f"Keys at step execution: {inside_info['keys']}"
    )


@pytest.mark.asyncio
@pytest.mark.xfail()
async def test_dbos_workflow_shuts_down_quickly(dbos_runtime: DBOSRuntime) -> None:
    """Test that workflows shut down quickly after completion (no 5s delay)."""
    import time

    class QuickShutdownWorkflow(Workflow):
        @step
        async def quick_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

    wf = QuickShutdownWorkflow(runtime=dbos_runtime)
    dbos_runtime.launch()

    start = time.monotonic()
    r = await WorkflowTestRunner(wf).run()
    elapsed = time.monotonic() - start

    assert r.result == "done"
    # Should complete much faster than the old 5-second polling interval
    assert elapsed < 2.0, f"Workflow took {elapsed:.2f}s, expected < 2s"
