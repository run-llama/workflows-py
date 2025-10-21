# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator, Callable

from workflows.events import Event, StopEvent
from workflows.runtime.types.plugin import Plugin, SnapshottableRuntime, WorkflowRuntime
from workflows.runtime.types.step_function import StepWorkerFunction
from workflows.runtime.types.ticks import WorkflowTick
from workflows.decorators import P, R
from workflows.workflow import Workflow


class BasicRuntime:
    def register(
        self,
        workflow: Workflow,
        workflow_function: Callable[P, R],
        steps: dict[str, StepWorkerFunction[R]],
    ) -> None:
        return

    def new_runtime(self, run_id: str) -> WorkflowRuntime:
        snapshottable: SnapshottableRuntime = AsyncioWorkflowRuntime(run_id)
        return snapshottable


basic_runtime: Plugin = BasicRuntime()


class AsyncioWorkflowRuntime:
    """
    A plugin interface to switch out a broker runtime (external library or service that manages durable/distributed step execution)
    """

    def __init__(
        self,
        run_id: str,
    ) -> None:
        self.run_id = run_id
        self.receive_queue: asyncio.Queue[WorkflowTick] = asyncio.Queue()
        self.publish_queue: asyncio.Queue[Event] = asyncio.Queue()
        self.ticks: list[WorkflowTick] = []

    def on_tick(self, tick: WorkflowTick) -> None:
        self.ticks.append(tick)

    def replay(self) -> list[WorkflowTick]:
        return self.ticks

    async def wait_receive(self) -> WorkflowTick:
        return await self.receive_queue.get()

    async def write_to_event_stream(self, event: Event) -> None:
        self.publish_queue.put_nowait(event)

    async def stream_published_events(self) -> AsyncGenerator[Event, None]:
        while True:
            item = await self.publish_queue.get()
            yield item
            if isinstance(item, StopEvent):
                break

    async def send_event(self, tick: WorkflowTick) -> None:
        """called externally to control the workflow from the outside world"""
        self.receive_queue.put_nowait(tick)

    async def register_step_worker(
        self, step_name: str, step_worker: StepWorkerFunction[R]
    ) -> StepWorkerFunction[R]:
        """API that the broker calls to convert a regular function to a registered step function as defined in the plugin's runtime. e.g. to register a task/activity that is cached on replay"""
        return step_worker

    async def register_workflow_function(
        self, workflow_function: Callable[P, R]
    ) -> Callable[P, R]:
        """API that the broker calls to convert a regular function to a registered workflow function as defined in the plugin's runtime."""
        return workflow_function

    async def get_now(self) -> float:
        """API that the broker calls to get the current time in seconds since epoch"""
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        """API that the broker calls to sleep for a given number of seconds"""
        await asyncio.sleep(seconds)

    async def close(self) -> None:
        """API that the broker calls to close the plugin after a workflow run is complete"""
        pass
