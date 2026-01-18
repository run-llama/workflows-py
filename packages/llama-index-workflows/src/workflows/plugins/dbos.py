# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
"""
DBOS Plugin for durable workflow execution.

This module requires the `dbos` package to be installed.
When imported without dbos, an ImportError will be raised.
"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator

# DBOS is an optional dependency
# Import will fail at runtime if dbos is not installed
from dbos import DBOS, SetWorkflowID  # type: ignore[import-not-found]

from workflows.events import Event, StopEvent
from workflows.runtime.control_loop import control_loop
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ControlLoopFunction,
    Plugin,
    RegisteredWorkflow,
    WorkflowRuntime,
)
from workflows.runtime.types.step_function import (
    StepWorkerFunction,
    as_step_worker_function,
)
from workflows.runtime.types.ticks import WorkflowTick
from workflows.runtime.workflow_tracker import WorkflowTracker
from workflows.workflow import Workflow


@DBOS.step()  # type: ignore[misc]
async def _durable_time() -> float:
    return time.time()


class DBOSRuntime(Plugin):
    """
    DBOS-backed workflow runtime for durable execution.

    Workflows are registered at launch() time with stable names,
    enabling distributed workers and recovery.
    """

    def __init__(self) -> None:
        self._tracker = WorkflowTracker()
        self._dbos_launched = False

    def track_workflow(self, workflow: Workflow) -> None:
        """Track a workflow for registration at launch time."""
        self._tracker.add(workflow)

    def get_registered(self, workflow: Workflow) -> RegisteredWorkflow | None:
        """Get the registered workflow if available."""
        return self._tracker.get_registered(workflow)

    def register(
        self,
        workflow: Workflow,
        workflow_function: ControlLoopFunction,
        steps: dict[str, StepWorkerFunction[Any]],
    ) -> RegisteredWorkflow | None:
        """
        Wrap workflow with DBOS decorators.

        Called at launch() time for each tracked workflow.
        Uses stable names based on workflow class and assigned name.
        """
        # Compute stable name for DBOS registration
        name = self._tracker.get_name(workflow)
        if name is None:
            # Generate name from class if not explicitly set
            name = workflow.__class__.__name__

        # Create DBOS-wrapped control loop with stable name
        @DBOS.workflow(name=f"{name}.control_loop")  # type: ignore[misc]
        async def _dbos_control_loop(
            start_event: Event | None,
            init_state: BrokerState | None,
            run_id: str,
        ) -> StopEvent:
            with SetWorkflowID(run_id):  # pyright: ignore[reportCallIssue]
                return await workflow_function(start_event, init_state, run_id)

        # Wrap steps with stable names
        wrapped_steps: dict[str, StepWorkerFunction[Any]] = {
            step_name: DBOS.step(name=f"{name}.{step_name}")(step)  # type: ignore[misc]
            for step_name, step in steps.items()
        }

        return RegisteredWorkflow(
            workflow_function=_dbos_control_loop, steps=wrapped_steps
        )

    def new_runtime(self, run_id: str) -> WorkflowRuntime:
        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS plugin not launched. Call plugin.launch() before running workflows."
            )
        return DBOSWorkflowRuntime(run_id)

    def launch(self) -> None:
        """
        Launch DBOS and register all tracked workflows.

        Must be called before running any workflows.
        """
        if self._dbos_launched:
            return  # Already launched

        # Register each pending workflow with DBOS
        for workflow in self._tracker.get_pending():
            # Get steps from the workflow instance (includes class + added steps)
            step_funcs = workflow._get_steps()
            step_workers: dict[str, StepWorkerFunction[Any]] = {
                name: as_step_worker_function(getattr(func, "__func__", func))
                for name, func in step_funcs.items()
            }

            # Register with DBOS (this applies decorators)
            registered = self.register(workflow, control_loop, step_workers)
            if registered is not None:
                self._tracker.set_registered(workflow, registered)

        # Mark as launched (no more workflows can be added)
        self._tracker.mark_launched()

        # Launch DBOS runtime
        DBOS.launch()  # type: ignore[misc]
        self._dbos_launched = True

    def destroy(self) -> None:
        """Clean up DBOS resources."""
        self._tracker.clear()
        self._dbos_launched = False
        # Note: DBOS doesn't have a clean shutdown API currently


class DBOSWorkflowRuntime:
    """
    Workflow runtime backed by asyncio mailboxes, with durable timing via DBOS when available.

    - send_event/wait_receive implement the tick mailbox used by the control loop
    - write_to_event_stream/stream_published_events expose published events to callers
    - get_now returns a stable value on first call within a run (durable if DBOS is installed)
    - sleep uses DBOS durable sleep when available, otherwise asyncio.sleep
    - on_tick/replay provide a lightweight snapshot for debug/replay via the broker
    """

    def __init__(
        self,
        run_id: str,
    ) -> None:
        self.run_id = run_id

    # Mailbox used by control loop and broker
    async def wait_receive(self) -> WorkflowTick:
        # Receive next tick via DBOS durable notification
        tick = await DBOS.recv_async()  # type: ignore[misc]
        return tick  # type: ignore[return-value]

    async def send_event(self, tick: WorkflowTick) -> None:
        await DBOS.send_async(self.run_id, tick)  # type: ignore[misc]

    # Event stream used by handlers/observers
    async def write_to_event_stream(self, event: Event) -> None:
        await DBOS.write_stream_async("published_events", event)  # type: ignore[misc]

    async def stream_published_events(self) -> AsyncGenerator[Event, None]:
        async for event in DBOS.read_stream_async(  # type: ignore[misc]
            self.run_id, "published_events"
        ):
            yield event

    # Timing utilities
    async def get_now(self) -> float:
        return await _durable_time()

    async def sleep(self, seconds: float) -> None:
        await DBOS.sleep_async(seconds)  # type: ignore[misc]

    async def close(self) -> None:
        pass
