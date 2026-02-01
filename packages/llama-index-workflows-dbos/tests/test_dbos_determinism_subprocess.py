"""Test DBOS determinism with subprocess isolation and real interruption.

This test spawns subprocesses to properly isolate DBOS state and simulate
real Ctrl+C interruptions during workflow execution.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Package paths for subprocess imports
DBOS_PACKAGE_SRC_PATH = str(Path(__file__).parent.parent / "src")
WORKFLOWS_PACKAGE_SRC_PATH = str(
    Path(__file__).parent.parent.parent / "llama-index-workflows" / "src"
)

# Common imports used by all subprocess scripts
COMMON_IMPORTS = """
import asyncio
from dbos import DBOS, DBOSConfig
from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, InputRequiredEvent, StartEvent, StopEvent
from llama_agents.runtime.dbos import DBOSRuntime
from workflows.workflow import Workflow
"""


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "dbos_test.sqlite3"


def run_workflow_script(
    script: str, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    """Run a Python script in a subprocess."""
    full_script = (
        f"import sys; sys.path.insert(0, {repr(DBOS_PACKAGE_SRC_PATH)}); "
        f"sys.path.insert(0, {repr(WORKFLOWS_PACKAGE_SRC_PATH)})\n"
    ) + script
    return subprocess.run(
        [sys.executable, "-c", full_script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def make_dbos_setup(db_url: str, app_name: str = "test-workflow") -> str:
    """Generate DBOS configuration and runtime setup code."""
    return textwrap.dedent(f'''
        config: DBOSConfig = {{
            "name": "{app_name}",
            "system_database_url": "{db_url}",
            "run_admin_server": False,
            "internal_polling_interval_sec": 0.01,
        }}
        DBOS(config=config)
        runtime = DBOSRuntime(polling_interval_sec=0.01)
    ''')


def make_script(
    workflow_code: str, main_code: str, db_url: str, app_name: str = "test-workflow"
) -> str:
    """Assemble a complete subprocess script from components."""
    return (
        COMMON_IMPORTS
        + "\n"
        + workflow_code
        + "\n\nasync def main():\n"
        + textwrap.indent(make_dbos_setup(db_url, app_name), "    ")
        + textwrap.indent(main_code, "    ")
        + "\n\nasyncio.run(main())\n"
    )


def assert_no_determinism_errors(result: subprocess.CompletedProcess[str]) -> None:
    """Check subprocess result for DBOS determinism errors."""
    combined = result.stdout + result.stderr
    if "DBOSUnexpectedStepError" in combined or "Error 11" in combined:
        pytest.fail(
            f"DBOS determinism error on resume!\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


# =============================================================================
# Test 1: Basic interrupt/resume with input events
# =============================================================================

HITL_WORKFLOW_CODE = textwrap.dedent("""
    class AskInputEvent(InputRequiredEvent):
        prefix: str = Field(default="Enter: ")

    class UserInput(Event):
        response: str = Field(default="")

    class TestWorkflow(Workflow):
        @step
        async def ask(self, ctx: Context, ev: StartEvent) -> AskInputEvent:
            await ctx.store.set("asked", True)
            print("STEP:ask:complete", flush=True)
            return AskInputEvent()

        @step
        async def process(self, ctx: Context, ev: UserInput) -> StopEvent:
            await ctx.store.set("processed", ev.response)
            print("STEP:process:complete", flush=True)
            return StopEvent(result={"response": ev.response})
""")


def test_determinism_on_resume_after_interrupt(test_db_path: Path) -> None:
    """Test that resuming an interrupted workflow doesn't hit determinism errors."""
    run_id = "test-determinism-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    start_main = textwrap.dedent(f'''
        wf = TestWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            async for event in handler.stream_events():
                print(f"EVENT:{{type(event).__name__}}", flush=True)
                if isinstance(event, AskInputEvent):
                    print("INTERRUPTING", flush=True)
                    import os
                    os._exit(0)
        finally:
            runtime.destroy()
    ''')

    resume_main = textwrap.dedent(f'''
        wf = TestWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            async for event in handler.stream_events():
                print(f"EVENT:{{type(event).__name__}}", flush=True)
                if isinstance(event, AskInputEvent):
                    if handler.ctx:
                        handler.ctx.send_event(UserInput(response="test_input"))
            result = await handler
            print(f"RESULT:{{result}}", flush=True)
            print("SUCCESS", flush=True)
        except Exception as e:
            print(f"ERROR:{{type(e).__name__}}:{{e}}", flush=True)
            raise
        finally:
            runtime.destroy()
    ''')

    print("\n=== Starting workflow (will interrupt) ===")
    result1 = run_workflow_script(make_script(HITL_WORKFLOW_CODE, start_main, db_url))
    print(f"stdout: {result1.stdout}")
    print(f"stderr: {result1.stderr}")

    assert "STEP:ask:complete" in result1.stdout, "First step should complete"
    assert "INTERRUPTING" in result1.stdout, "Should have interrupted"

    print("\n=== Resuming workflow ===")
    result2 = run_workflow_script(make_script(HITL_WORKFLOW_CODE, resume_main, db_url))
    print(f"stdout: {result2.stdout}")
    print(f"stderr: {result2.stderr}")

    assert_no_determinism_errors(result2)
    assert "SUCCESS" in result2.stdout or result2.returncode == 0, (
        f"Resume should succeed. stdout: {result2.stdout}, stderr: {result2.stderr}"
    )


# =============================================================================
# Test 2: Chained steps determinism
# =============================================================================

CHAINED_WORKFLOW_CODE = textwrap.dedent("""
    class StepOneEvent(Event):
        value: str = Field(default="one")

    class StepTwoEvent(Event):
        value: str = Field(default="two")

    class ChainedWorkflow(Workflow):
        @step
        async def step_one(self, ctx: Context, ev: StartEvent) -> StepOneEvent:
            await ctx.store.set("step_one", True)
            print("STEP:one:complete", flush=True)
            return StepOneEvent()

        @step
        async def step_two(self, ctx: Context, ev: StepOneEvent) -> StepTwoEvent:
            await ctx.store.set("step_two", True)
            print("STEP:two:complete", flush=True)
            return StepTwoEvent()

        @step
        async def step_three(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
            await ctx.store.set("step_three", True)
            print("STEP:three:complete", flush=True)
            return StopEvent(result="done")
""")

# Variant with interrupt in step_two
CHAINED_WORKFLOW_CODE_WITH_INTERRUPT = textwrap.dedent("""
    class StepOneEvent(Event):
        value: str = Field(default="one")

    class StepTwoEvent(Event):
        value: str = Field(default="two")

    class ChainedWorkflow(Workflow):
        @step
        async def step_one(self, ctx: Context, ev: StartEvent) -> StepOneEvent:
            await ctx.store.set("step_one", True)
            print("STEP:one:complete", flush=True)
            return StepOneEvent()

        @step
        async def step_two(self, ctx: Context, ev: StepOneEvent) -> StepTwoEvent:
            await ctx.store.set("step_two", True)
            print("STEP:two:complete", flush=True)
            import os
            os._exit(0)
            return StepTwoEvent()

        @step
        async def step_three(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
            await ctx.store.set("step_three", True)
            print("STEP:three:complete", flush=True)
            return StopEvent(result="done")
""")


def test_chained_steps_determinism_on_resume(test_db_path: Path) -> None:
    """Test determinism with chained steps that trigger each other."""
    run_id = "test-chained-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    start_main = textwrap.dedent(f'''
        wf = ChainedWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            result = await handler
            print(f"RESULT:{{result}}", flush=True)
        finally:
            runtime.destroy()
    ''')

    resume_main = textwrap.dedent(f'''
        wf = ChainedWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            result = await handler
            print(f"RESULT:{{result}}", flush=True)
            print("SUCCESS", flush=True)
        except Exception as e:
            print(f"ERROR:{{type(e).__name__}}:{{e}}", flush=True)
            raise
        finally:
            runtime.destroy()
    ''')

    print("\n=== Starting chained workflow (will interrupt at step 2) ===")
    result1 = run_workflow_script(
        make_script(
            CHAINED_WORKFLOW_CODE_WITH_INTERRUPT, start_main, db_url, "test-chained"
        )
    )
    print(f"stdout: {result1.stdout}")
    print(f"stderr: {result1.stderr}")

    assert "STEP:one:complete" in result1.stdout, "Step one should complete"

    print("\n=== Resuming chained workflow ===")
    result2 = run_workflow_script(
        make_script(CHAINED_WORKFLOW_CODE, resume_main, db_url, "test-chained")
    )
    print(f"stdout: {result2.stdout}")
    print(f"stderr: {result2.stderr}")

    assert_no_determinism_errors(result2)


# =============================================================================
# Test 3: Three-step HITL pattern
# =============================================================================

THREE_STEP_HITL_WORKFLOW_CODE = textwrap.dedent("""
    class NameInputEvent(InputRequiredEvent):
        prefix: str = Field(default="Name: ")

    class NameResponseEvent(Event):
        response: str = Field(default="")

    class QuestInputEvent(InputRequiredEvent):
        prefix: str = Field(default="Quest: ")

    class QuestResponseEvent(Event):
        response: str = Field(default="")

    class HITLWorkflow(Workflow):
        @step
        async def ask_name(self, ctx: Context, ev: StartEvent) -> NameInputEvent:
            await ctx.store.set("asked_name", True)
            print("STEP:ask_name:complete", flush=True)
            return NameInputEvent()

        @step
        async def ask_quest(self, ctx: Context, ev: NameResponseEvent) -> QuestInputEvent:
            await ctx.store.set("name", ev.response)
            print(f"STEP:ask_quest:got_name={ev.response}", flush=True)
            print("STEP:ask_quest:complete", flush=True)
            return QuestInputEvent()

        @step
        async def complete(self, ctx: Context, ev: QuestResponseEvent) -> StopEvent:
            name = await ctx.store.get("name", default="unknown")
            print(f"STEP:complete:got_quest={ev.response}", flush=True)
            return StopEvent(result={"name": name, "quest": ev.response})
""")


def test_hitl_three_step_determinism(test_db_path: Path) -> None:
    """Test the exact HITL pattern with three steps and input events."""
    run_id = "test-hitl-three-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    start_main = textwrap.dedent(f'''
        wf = HITLWorkflow(runtime=runtime)
        runtime.launch()
        ctx = Context(wf)
        handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
        async for event in handler.stream_events():
            print(f"EVENT:{{type(event).__name__}}", flush=True)
            if isinstance(event, NameInputEvent):
                if handler.ctx:
                    handler.ctx.send_event(NameResponseEvent(response="Alice"))
            elif isinstance(event, QuestInputEvent):
                print("INTERRUPTING_AT_QUEST_PROMPT", flush=True)
                import os
                os._exit(0)
    ''')

    resume_main = textwrap.dedent(f'''
        wf = HITLWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            async for event in handler.stream_events():
                print(f"EVENT:{{type(event).__name__}}", flush=True)
                if isinstance(event, NameInputEvent):
                    if handler.ctx:
                        handler.ctx.send_event(NameResponseEvent(response="Alice"))
                elif isinstance(event, QuestInputEvent):
                    if handler.ctx:
                        handler.ctx.send_event(QuestResponseEvent(response="seek the grail"))
            result = await handler
            print(f"RESULT:{{result}}", flush=True)
            print("SUCCESS", flush=True)
        except Exception as e:
            print(f"ERROR:{{type(e).__name__}}:{{e}}", flush=True)
            raise
        finally:
            runtime.destroy()
    ''')

    print("\n=== Starting HITL workflow (will interrupt at quest prompt) ===")
    result1 = run_workflow_script(
        make_script(THREE_STEP_HITL_WORKFLOW_CODE, start_main, db_url, "test-hitl")
    )
    print(f"stdout: {result1.stdout}")
    print(f"stderr: {result1.stderr}")

    assert "STEP:ask_name:complete" in result1.stdout, "ask_name should complete"
    assert "STEP:ask_quest" in result1.stdout, "ask_quest should start"
    assert "INTERRUPTING_AT_QUEST_PROMPT" in result1.stdout, "Should interrupt at quest"

    print("\n=== Resuming HITL workflow ===")
    result2 = run_workflow_script(
        make_script(THREE_STEP_HITL_WORKFLOW_CODE, resume_main, db_url, "test-hitl")
    )
    print(f"stdout: {result2.stdout}")
    print(f"stderr: {result2.stderr}")

    assert_no_determinism_errors(result2)
    assert "SUCCESS" in result2.stdout or result2.returncode == 0, (
        f"Resume should succeed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )


# =============================================================================
# Test 4: Parallel steps - two steps triggered by StartEvent
# This tests that when both branch_a and branch_b are triggered by StartEvent,
# the journal correctly records which completes first and replays in that order.
# =============================================================================

PARALLEL_STEPS_WORKFLOW_CODE = textwrap.dedent("""
    import asyncio
    import random

    class ResultAEvent(Event):
        value: str = Field(default="")

    class ResultBEvent(Event):
        value: str = Field(default="")

    class ParallelWorkflow(Workflow):
        @step
        async def branch_a(self, ctx: Context, ev: StartEvent) -> ResultAEvent:
            # Variable processing time - may complete before or after branch_b
            await asyncio.sleep(random.uniform(0.01, 0.05))
            print("STEP:branch_a:complete", flush=True)
            return ResultAEvent(value="a_result")

        @step
        async def branch_b(self, ctx: Context, ev: StartEvent) -> ResultBEvent:
            # Variable processing time - may complete before or after branch_a
            await asyncio.sleep(random.uniform(0.01, 0.05))
            print("STEP:branch_b:complete", flush=True)
            return ResultBEvent(value="b_result")

        @step
        async def finish_a(self, ctx: Context, ev: ResultAEvent) -> StopEvent:
            print(f"STEP:finish_a:complete", flush=True)
            return StopEvent(result={"winner": "a", "value": ev.value})

        @step
        async def finish_b(self, ctx: Context, ev: ResultBEvent) -> StopEvent:
            print(f"STEP:finish_b:complete", flush=True)
            return StopEvent(result={"winner": "b", "value": ev.value})
""")


def test_parallel_steps_determinism(test_db_path: Path) -> None:
    """Test determinism with parallel steps completing in non-deterministic order."""
    run_id = "test-parallel-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # Run to completion first time
    run_main = textwrap.dedent(f'''
        wf = ParallelWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            result = await handler
            print(f"RESULT:{{result}}", flush=True)
            print("SUCCESS", flush=True)
        except Exception as e:
            print(f"ERROR:{{type(e).__name__}}:{{e}}", flush=True)
            raise
        finally:
            runtime.destroy()
    ''')

    print("\n=== Running parallel workflow to completion ===")
    result1 = run_workflow_script(
        make_script(PARALLEL_STEPS_WORKFLOW_CODE, run_main, db_url, "test-parallel")
    )
    print(f"stdout: {result1.stdout}")
    print(f"stderr: {result1.stderr}")

    assert "SUCCESS" in result1.stdout, (
        f"Should complete successfully.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert_no_determinism_errors(result1)


# =============================================================================
# Test 5: Concurrent workers on same step (num_workers=2)
# This tests that when a single step has multiple workers processing events
# concurrently, the journal correctly records completion order.
# =============================================================================

CONCURRENT_WORKERS_WORKFLOW_CODE = textwrap.dedent("""
    import asyncio
    import random

    class WorkItem(Event):
        item_id: int = Field(default=0)

    class WorkDone(Event):
        item_id: int = Field(default=0)

    class ConcurrentWorkersWorkflow(Workflow):
        @step
        async def dispatch(self, ctx: Context, ev: StartEvent) -> WorkItem:
            # Dispatch work items that will be processed by concurrent workers
            ctx.send_event(WorkItem(item_id=1))
            ctx.send_event(WorkItem(item_id=2))
            print("STEP:dispatch:complete", flush=True)
            return WorkItem(item_id=0)

        @step(num_workers=2)
        async def worker(self, ctx: Context, ev: WorkItem) -> WorkDone:
            # Variable processing time for each item
            await asyncio.sleep(random.uniform(0.01, 0.05))
            print(f"STEP:worker:{ev.item_id}:complete", flush=True)
            return WorkDone(item_id=ev.item_id)

        @step
        async def finish(self, ctx: Context, ev: WorkDone) -> StopEvent:
            # First WorkDone to arrive ends the workflow
            print(f"STEP:finish:{ev.item_id}:complete", flush=True)
            return StopEvent(result={"first_done": ev.item_id})
""")


def test_concurrent_workers_determinism(test_db_path: Path) -> None:
    """Test determinism with multiple workers on same step (num_workers > 1)."""
    run_id = "test-concurrent-workers-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    run_main = textwrap.dedent(f'''
        wf = ConcurrentWorkersWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            result = await handler
            print(f"RESULT:{{result}}", flush=True)
            print("SUCCESS", flush=True)
        except Exception as e:
            print(f"ERROR:{{type(e).__name__}}:{{e}}", flush=True)
            raise
        finally:
            runtime.destroy()
    ''')

    print("\n=== Running concurrent workers workflow ===")
    result1 = run_workflow_script(
        make_script(
            CONCURRENT_WORKERS_WORKFLOW_CODE, run_main, db_url, "test-concurrent"
        )
    )
    print(f"stdout: {result1.stdout}")
    print(f"stderr: {result1.stderr}")

    assert "SUCCESS" in result1.stdout, (
        f"Should complete successfully.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert_no_determinism_errors(result1)


# =============================================================================
# Test 6: Sequential steps with HITL - tests chained execution with human input
# (Simpler than parallel - avoids dual-event race condition)
# =============================================================================

SEQUENTIAL_HITL_WORKFLOW_CODE = textwrap.dedent("""
    import asyncio
    import random

    class ProcessedEvent(Event):
        value: str = Field(default="")

    class WaitForInputEvent(InputRequiredEvent):
        prompt: str = Field(default="")

    class UserContinueEvent(Event):
        continue_value: str = Field(default="")

    class SequentialHITLWorkflow(Workflow):
        @step
        async def process(self, ctx: Context, ev: StartEvent) -> ProcessedEvent:
            await asyncio.sleep(random.uniform(0.01, 0.05))
            print("STEP:process:complete", flush=True)
            return ProcessedEvent(value="processed")

        @step
        async def ask_user(self, ctx: Context, ev: ProcessedEvent) -> WaitForInputEvent:
            print("STEP:ask_user:triggering_wait", flush=True)
            return WaitForInputEvent(prompt=f"Got {ev.value}")

        @step
        async def finalize(self, ctx: Context, ev: UserContinueEvent) -> StopEvent:
            print(f"STEP:finalize:complete:{ev.continue_value}", flush=True)
            return StopEvent(result={"continue": ev.continue_value})
""")


def test_sequential_hitl_interrupt_resume(test_db_path: Path) -> None:
    """Test sequential steps with HITL interrupt and resume."""
    run_id = "test-seq-hitl-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    start_main = textwrap.dedent(f'''
        wf = SequentialHITLWorkflow(runtime=runtime)
        runtime.launch()
        ctx = Context(wf)
        handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
        async for event in handler.stream_events():
            print(f"EVENT:{{type(event).__name__}}", flush=True)
            if isinstance(event, WaitForInputEvent):
                print("INTERRUPTING_AT_HITL", flush=True)
                import os
                os._exit(0)
    ''')

    resume_main = textwrap.dedent(f'''
        wf = SequentialHITLWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            async for event in handler.stream_events():
                print(f"EVENT:{{type(event).__name__}}", flush=True)
                if isinstance(event, WaitForInputEvent):
                    if handler.ctx:
                        handler.ctx.send_event(UserContinueEvent(continue_value="user_input"))
            result = await handler
            print(f"RESULT:{{result}}", flush=True)
            print("SUCCESS", flush=True)
        except Exception as e:
            print(f"ERROR:{{type(e).__name__}}:{{e}}", flush=True)
            raise
        finally:
            runtime.destroy()
    ''')

    print("\n=== Starting sequential HITL workflow (will interrupt) ===")
    result1 = run_workflow_script(
        make_script(SEQUENTIAL_HITL_WORKFLOW_CODE, start_main, db_url, "test-seq-hitl")
    )
    print(f"stdout: {result1.stdout}")
    print(f"stderr: {result1.stderr}")

    assert "STEP:process:complete" in result1.stdout
    assert "INTERRUPTING_AT_HITL" in result1.stdout

    print("\n=== Resuming sequential HITL workflow ===")
    result2 = run_workflow_script(
        make_script(SEQUENTIAL_HITL_WORKFLOW_CODE, resume_main, db_url, "test-seq-hitl")
    )
    print(f"stdout: {result2.stdout}")
    print(f"stderr: {result2.stderr}")

    assert_no_determinism_errors(result2)
    assert "SUCCESS" in result2.stdout or result2.returncode == 0, (
        f"Resume should succeed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )


# =============================================================================
# Stress tests - run scenarios multiple times to catch flaky timing issues
# =============================================================================


@pytest.mark.parametrize("iteration", range(5))
def test_parallel_steps_stress(test_db_path: Path, iteration: int) -> None:
    """Stress test parallel steps - run 5 times to catch timing issues."""
    run_id = f"test-parallel-stress-{iteration}"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    run_main = textwrap.dedent(f'''
        wf = ParallelWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            result = await handler
            print(f"RESULT:{{result}}", flush=True)
            print("SUCCESS", flush=True)
        except Exception as e:
            print(f"ERROR:{{type(e).__name__}}:{{e}}", flush=True)
            raise
        finally:
            runtime.destroy()
    ''')

    result = run_workflow_script(
        make_script(
            PARALLEL_STEPS_WORKFLOW_CODE, run_main, db_url, f"stress-{iteration}"
        )
    )

    assert "SUCCESS" in result.stdout, (
        f"Iteration {iteration} failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert_no_determinism_errors(result)


@pytest.mark.parametrize("iteration", range(5))
def test_concurrent_workers_stress(test_db_path: Path, iteration: int) -> None:
    """Stress test concurrent workers - run 5 times to catch timing issues."""
    run_id = f"test-concurrent-stress-{iteration}"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    run_main = textwrap.dedent(f'''
        wf = ConcurrentWorkersWorkflow(runtime=runtime)
        runtime.launch()
        try:
            ctx = Context(wf)
            handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
            result = await handler
            print(f"RESULT:{{result}}", flush=True)
            print("SUCCESS", flush=True)
        except Exception as e:
            print(f"ERROR:{{type(e).__name__}}:{{e}}", flush=True)
            raise
        finally:
            runtime.destroy()
    ''')

    result = run_workflow_script(
        make_script(
            CONCURRENT_WORKERS_WORKFLOW_CODE,
            run_main,
            db_url,
            f"concurrent-stress-{iteration}",
        )
    )

    assert "SUCCESS" in result.stdout, (
        f"Iteration {iteration} failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert_no_determinism_errors(result)
