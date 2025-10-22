# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import time
from typing import AsyncGenerator

from dbos import DBOS, SetWorkflowID  # required extra, import must succeed

from workflows.events import Event, StopEvent
from workflows.runtime.types.plugin import (
    ControlLoopFunction,
    Plugin,
    WorkflowRuntime,
    RegisteredWorkflow,
)
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.step_function import StepWorkerFunction
from workflows.runtime.types.ticks import WorkflowTick

from workflows.workflow import Workflow


@DBOS.step()
async def _durable_time() -> float:
    return time.time()


class DBOSRuntime:
    def register(
        self,
        workflow: Workflow,
        workflow_function: ControlLoopFunction,
        steps: dict[str, StepWorkerFunction],
    ) -> RegisteredWorkflow | None:
        """
        Wrap the workflow control loop in a DBOS workflow so ticks are received via DBOS.recv
        and sent via DBOS.send, enabling durable orchestration.
        """

        @DBOS.workflow()
        async def _dbos_control_loop(
            start_event: Event | None,
            init_state: BrokerState | None,
            run_id: str,
        ) -> StopEvent:
            with SetWorkflowID(run_id):
                return await workflow_function(start_event, init_state, run_id)

        wrapped_steps = {name: DBOS.step()(step) for name, step in steps.items()}

        return RegisteredWorkflow(
            workflow_function=_dbos_control_loop, steps=wrapped_steps
        )

    def new_runtime(self, run_id: str) -> WorkflowRuntime:
        runtime: WorkflowRuntime = DBOSWorkflowRuntime(run_id)
        return runtime


dbos_runtime: Plugin = DBOSRuntime()


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
        tick = await DBOS.recv_async()
        return tick  # type: ignore[return-value]

    async def send_event(self, tick: WorkflowTick) -> None:
        await DBOS.send_async(self.run_id, tick)

    # Event stream used by handlers/observers
    async def write_to_event_stream(self, event: Event) -> None:
        await DBOS.write_stream_async("published_events", event)

    async def stream_published_events(self) -> AsyncGenerator[Event, None]:
        async for event in DBOS.read_stream_async(self.run_id, "published_events"):
            yield event

    # Timing utilities
    async def get_now(self) -> float:
        return await _durable_time()

    async def sleep(self, seconds: float) -> None:
        await DBOS.sleep_async(seconds)

    async def close(self) -> None:
        pass
