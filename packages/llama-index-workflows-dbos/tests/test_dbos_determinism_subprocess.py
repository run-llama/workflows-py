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

# Get the package src paths for subprocess imports
# We need both the DBOS package and the main workflows package
#
# NOTE: The inline DBOSConfig values in the subprocess scripts below should match
# the defaults in conftest.py:TEST_DBOS_DEFAULTS. Since subprocesses can't import
# from conftest, we duplicate the values here.
DBOS_PACKAGE_SRC_PATH = str(Path(__file__).parent.parent / "src")
WORKFLOWS_PACKAGE_SRC_PATH = str(
    Path(__file__).parent.parent.parent / "llama-index-workflows" / "src"
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "dbos_test.sqlite3"


def run_workflow_script(
    script: str, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    """Run a Python script in a subprocess."""
    # Prepend sys.path modifications to the script
    full_script = (
        f"import sys; sys.path.insert(0, {repr(DBOS_PACKAGE_SRC_PATH)}); "
        f"sys.path.insert(0, {repr(WORKFLOWS_PACKAGE_SRC_PATH)})\n"
    ) + script
    result = subprocess.run(
        [sys.executable, "-c", full_script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


def test_determinism_on_resume_after_interrupt(test_db_path: Path) -> None:
    """Test that resuming an interrupted workflow doesn't hit determinism errors.

    This test:
    1. Starts a workflow that waits for external input
    2. Interrupts it (simulated by just exiting after partial execution)
    3. Resumes with the same run_id
    4. Checks for DBOS determinism errors
    """
    run_id = "test-determinism-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # Script that starts a workflow and exits after first step completes
    # (simulating interruption while waiting for input)
    start_script = textwrap.dedent(f'''
        import asyncio
        import sys
        from pathlib import Path

        # Add the package to path
        # sys.path handled by run_workflow_script

        from dbos import DBOS, DBOSConfig, SetWorkflowID
        from pydantic import Field
        from workflows.context import Context
        from workflows.decorators import step
        from workflows.events import Event, InputRequiredEvent, StartEvent, StopEvent
        from llama_agents.runtime.dbos import DBOSRuntime
        from workflows.workflow import Workflow


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
                return StopEvent(result={{"response": ev.response}})


        async def main():
            config: DBOSConfig = {{
                "name": "test-workflow",
                "system_database_url": "{db_url}",
                "run_admin_server": False,
                "internal_polling_interval_sec": 0.01,
            }}
            DBOS(config=config)
            runtime = DBOSRuntime(polling_interval_sec=0.01)
            wf = TestWorkflow(runtime=runtime)
            runtime.launch()

            try:
                ctx = Context(wf)
                handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")

                # Wait for the first event (AskInputEvent)
                event_count = 0
                async for event in handler.stream_events():
                    print(f"EVENT:{{type(event).__name__}}", flush=True)
                    if isinstance(event, AskInputEvent):
                        event_count += 1
                        # Simulate hard interruption - force exit
                        print("INTERRUPTING", flush=True)
                        import os
                        os._exit(0)

                print("EXITING_EARLY", flush=True)
            finally:
                runtime.destroy()

        asyncio.run(main())
    ''')

    # Script that resumes the workflow
    resume_script = textwrap.dedent(f'''
        import asyncio
        import sys
        from pathlib import Path

        # sys.path handled by run_workflow_script

        from dbos import DBOS, DBOSConfig, SetWorkflowID
        from pydantic import Field
        from workflows.context import Context
        from workflows.decorators import step
        from workflows.events import Event, InputRequiredEvent, StartEvent, StopEvent
        from llama_agents.runtime.dbos import DBOSRuntime
        from workflows.workflow import Workflow


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
                return StopEvent(result={{"response": ev.response}})


        async def main():
            config: DBOSConfig = {{
                "name": "test-workflow",
                "system_database_url": "{db_url}",
                "run_admin_server": False,
                "internal_polling_interval_sec": 0.01,
            }}
            DBOS(config=config)
            runtime = DBOSRuntime(polling_interval_sec=0.01)
            wf = TestWorkflow(runtime=runtime)
            runtime.launch()

            try:
                ctx = Context(wf)
                # Resume with same run_id
                handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")

                async for event in handler.stream_events():
                    print(f"EVENT:{{type(event).__name__}}", flush=True)
                    if isinstance(event, AskInputEvent):
                        # Send input to continue
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

        asyncio.run(main())
    ''')

    # Run the start script
    print("\n=== Starting workflow (will interrupt) ===")
    result1 = run_workflow_script(start_script)
    print(f"stdout: {result1.stdout}")
    print(f"stderr: {result1.stderr}")

    assert "STEP:ask:complete" in result1.stdout, "First step should complete"
    assert "INTERRUPTING" in result1.stdout, "Should have interrupted"

    # Run the resume script
    print("\n=== Resuming workflow ===")
    result2 = run_workflow_script(resume_script)
    print(f"stdout: {result2.stdout}")
    print(f"stderr: {result2.stderr}")

    # Check for determinism errors
    if (
        "DBOSUnexpectedStepError" in result2.stderr
        or "DBOSUnexpectedStepError" in result2.stdout
    ):
        pytest.fail(
            f"DBOS determinism error on resume!\n"
            f"stdout: {result2.stdout}\n"
            f"stderr: {result2.stderr}"
        )

    # Should complete successfully
    assert "SUCCESS" in result2.stdout or result2.returncode == 0, (
        f"Resume should succeed. stdout: {result2.stdout}, stderr: {result2.stderr}"
    )


def test_chained_steps_determinism_on_resume(test_db_path: Path) -> None:
    """Test determinism with chained steps that trigger each other.

    This is closer to the HITL bug where:
    1. Step A completes and emits event
    2. Event triggers Step B
    3. Control loop writes to stream while Step B starts
    4. Race condition in DBOS function_id ordering
    """
    run_id = "test-chained-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # Script with chained steps - interrupts after step 2
    start_script = textwrap.dedent(f'''
        import asyncio
        import sys
        from pathlib import Path

        # sys.path handled by run_workflow_script

        from dbos import DBOS, DBOSConfig
        from pydantic import Field
        from workflows.context import Context
        from workflows.decorators import step
        from workflows.events import Event, StartEvent, StopEvent
        from llama_agents.runtime.dbos import DBOSRuntime
        from workflows.workflow import Workflow


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
                # Exit here to simulate interruption
                import os
                os._exit(0)
                return StepTwoEvent()

            @step
            async def step_three(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
                await ctx.store.set("step_three", True)
                print("STEP:three:complete", flush=True)
                return StopEvent(result="done")


        async def main():
            config: DBOSConfig = {{
                "name": "test-chained",
                "system_database_url": "{db_url}",
                "run_admin_server": False,
                "internal_polling_interval_sec": 0.01,
            }}
            DBOS(config=config)
            runtime = DBOSRuntime(polling_interval_sec=0.01)
            wf = ChainedWorkflow(runtime=runtime)
            runtime.launch()

            try:
                ctx = Context(wf)
                handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")
                result = await handler
                print(f"RESULT:{{result}}", flush=True)
            finally:
                runtime.destroy()

        asyncio.run(main())
    ''')

    # Resume script
    resume_script = textwrap.dedent(f'''
        import asyncio
        import sys
        from pathlib import Path

        # sys.path handled by run_workflow_script

        from dbos import DBOS, DBOSConfig
        from pydantic import Field
        from workflows.context import Context
        from workflows.decorators import step
        from workflows.events import Event, StartEvent, StopEvent
        from llama_agents.runtime.dbos import DBOSRuntime
        from workflows.workflow import Workflow


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


        async def main():
            config: DBOSConfig = {{
                "name": "test-chained",
                "system_database_url": "{db_url}",
                "run_admin_server": False,
                "internal_polling_interval_sec": 0.01,
            }}
            DBOS(config=config)
            runtime = DBOSRuntime(polling_interval_sec=0.01)
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

        asyncio.run(main())
    ''')

    # Run start script (will exit mid-step)
    print("\n=== Starting chained workflow (will interrupt at step 2) ===")
    result1 = run_workflow_script(start_script)
    print(f"stdout: {result1.stdout}")
    print(f"stderr: {result1.stderr}")

    assert "STEP:one:complete" in result1.stdout, "Step one should complete"
    # Step two may or may not print before os._exit

    # Run resume script
    print("\n=== Resuming chained workflow ===")
    result2 = run_workflow_script(resume_script)
    print(f"stdout: {result2.stdout}")
    print(f"stderr: {result2.stderr}")

    # Check for determinism errors
    if "DBOSUnexpectedStepError" in result2.stderr:
        pytest.fail(
            f"DBOS determinism error on resume!\n"
            f"This confirms the race condition between step execution and stream writes.\n"
            f"stderr: {result2.stderr}"
        )

    if "Error 11" in result2.stderr:
        pytest.fail(
            f"DBOS Error 11 (unexpected step) on resume!\nstderr: {result2.stderr}"
        )


def test_hitl_three_step_determinism(test_db_path: Path) -> None:
    """Test the exact HITL pattern with three steps and input events.

    This mimics the exact bug scenario:
    1. ask_name emits NameInputEvent
    2. User sends NameResponseEvent
    3. ask_quest receives it, emits QuestInputEvent
    4. User sends QuestResponseEvent
    5. complete receives it, finishes

    The bug occurs because when ask_quest finishes and emits QuestInputEvent,
    the control loop writes to stream AND starts the next step concurrently.
    """
    run_id = "test-hitl-three-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # Script that runs HITL workflow and interrupts after getting name
    start_script = textwrap.dedent(f'''
import asyncio
from dbos import DBOS, DBOSConfig
from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, InputRequiredEvent, StartEvent, StopEvent
from llama_agents.runtime.dbos import DBOSRuntime
from workflows.workflow import Workflow


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
        print(f"STEP:ask_quest:got_name={{ev.response}}", flush=True)
        print("STEP:ask_quest:complete", flush=True)
        return QuestInputEvent()

    @step
    async def complete(self, ctx: Context, ev: QuestResponseEvent) -> StopEvent:
        name = await ctx.store.get("name", default="unknown")
        print(f"STEP:complete:got_quest={{ev.response}}", flush=True)
        return StopEvent(result={{"name": name, "quest": ev.response}})


async def main():
    config: DBOSConfig = {{
        "name": "test-hitl",
        "system_database_url": "{db_url}",
        "run_admin_server": False,
        "internal_polling_interval_sec": 0.01,
    }}
    DBOS(config=config)
    runtime = DBOSRuntime(polling_interval_sec=0.01)
    wf = HITLWorkflow(runtime=runtime)
    runtime.launch()

    ctx = Context(wf)
    handler = ctx._workflow_run(wf, StartEvent(), run_id="{run_id}")

    async for event in handler.stream_events():
        print(f"EVENT:{{type(event).__name__}}", flush=True)
        if isinstance(event, NameInputEvent):
            # Send name response
            if handler.ctx:
                handler.ctx.send_event(NameResponseEvent(response="Alice"))
        elif isinstance(event, QuestInputEvent):
            # Interrupt here - after name was processed, quest prompt emitted
            print("INTERRUPTING_AT_QUEST_PROMPT", flush=True)
            import os
            os._exit(0)

asyncio.run(main())
    ''')

    # Resume script
    resume_script = textwrap.dedent(f'''
import asyncio
from dbos import DBOS, DBOSConfig
from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, InputRequiredEvent, StartEvent, StopEvent
from llama_agents.runtime.dbos import DBOSRuntime
from workflows.workflow import Workflow


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
        print(f"STEP:ask_quest:got_name={{ev.response}}", flush=True)
        print("STEP:ask_quest:complete", flush=True)
        return QuestInputEvent()

    @step
    async def complete(self, ctx: Context, ev: QuestResponseEvent) -> StopEvent:
        name = await ctx.store.get("name", default="unknown")
        print(f"STEP:complete:got_quest={{ev.response}}", flush=True)
        return StopEvent(result={{"name": name, "quest": ev.response}})


async def main():
    config: DBOSConfig = {{
        "name": "test-hitl",
        "system_database_url": "{db_url}",
        "run_admin_server": False,
        "internal_polling_interval_sec": 0.01,
    }}
    DBOS(config=config)
    runtime = DBOSRuntime(polling_interval_sec=0.01)
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

asyncio.run(main())
    ''')

    # Run start script
    print("\n=== Starting HITL workflow (will interrupt at quest prompt) ===")
    result1 = run_workflow_script(start_script)
    print(f"stdout: {result1.stdout}")
    print(f"stderr: {result1.stderr}")

    assert "STEP:ask_name:complete" in result1.stdout, "ask_name should complete"
    assert "STEP:ask_quest" in result1.stdout, "ask_quest should start"
    assert "INTERRUPTING_AT_QUEST_PROMPT" in result1.stdout, "Should interrupt at quest"

    # Run resume script
    print("\n=== Resuming HITL workflow ===")
    result2 = run_workflow_script(resume_script)
    print(f"stdout: {result2.stdout}")
    print(f"stderr: {result2.stderr}")

    # Check for determinism errors
    if "DBOSUnexpectedStepError" in result2.stderr or "Error 11" in result2.stderr:
        pytest.fail(
            f"DBOS determinism error on resume!\n"
            f"This confirms the HITL race condition.\n"
            f"stdout: {result2.stdout}\n"
            f"stderr: {result2.stderr}"
        )

    assert "SUCCESS" in result2.stdout or result2.returncode == 0, (
        f"Resume should succeed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )
