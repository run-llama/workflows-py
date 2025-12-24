# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

"""Test fixtures and utilities for runtime tests."""

import asyncio
import time
from typing import Any, AsyncGenerator, Optional

import pytest
import time_machine
from workflows.events import Event, StopEvent
from workflows.runtime.types.plugin import ControlLoopFunction, Plugin, WorkflowRuntime
from workflows.runtime.types.step_function import StepWorkerFunction
from workflows.runtime.types.ticks import WorkflowTick
from workflows.workflow import Workflow


class MockPlugin(Plugin):
    def register(
        self,
        workflow: Workflow,
        workflow_function: ControlLoopFunction,
        steps: dict[str, StepWorkerFunction],
    ) -> None:
        return

    def new_runtime(self, run_id: str) -> WorkflowRuntime:
        return MockRuntimePlugin(run_id)


class MockRuntimePlugin(WorkflowRuntime):
    """Mock WorkflowRuntime for testing control loops."""

    def __init__(
        self, run_id: str, traveller: Optional[time_machine.Coordinates] = None
    ) -> None:
        self.run_id = run_id
        # Queue for events sent from external sources (e.g., via send_event)
        self._external_queue: asyncio.Queue[WorkflowTick] = asyncio.Queue()
        # Queue for events published to the event stream (e.g., for UI/callbacks)
        self._event_stream: asyncio.Queue[Event] = asyncio.Queue()
        # Time-machine traveller for deterministic time control
        self._traveller = traveller
        # Current time in seconds, can be advanced manually for testing
        self._current_time: float = time.time()

    async def close(self) -> None:
        """
        Close the plugin.
        """
        pass

    async def wait_receive(self) -> WorkflowTick:
        return await self._external_queue.get()

    async def write_to_event_stream(self, event: Event) -> None:
        await self._event_stream.put(event)

    async def stream_published_events(self) -> AsyncGenerator[Event, None]:
        while True:
            item = await self._event_stream.get()
            yield item
            if isinstance(item, StopEvent):
                break

    async def send_event(self, tick: WorkflowTick) -> None:
        await self._external_queue.put(tick)

    async def register_step_worker(self, step_name: str, step_worker: Any) -> Any:
        return step_worker

    async def register_workflow_function(self, workflow_function: Any) -> Any:
        return workflow_function

    async def get_now(self) -> float:
        if self._traveller is not None:
            return time.time()
        return self._current_time

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def advance_time(self, seconds: float) -> None:
        if self._traveller is not None:
            self._traveller.shift(seconds)
        else:
            self._current_time += seconds

    async def get_stream_event(self, timeout: float = 1.0) -> Event:
        return await asyncio.wait_for(self._event_stream.get(), timeout=timeout)

    def has_stream_events(self) -> bool:
        return not self._event_stream.empty()


@pytest.fixture
async def test_plugin() -> MockRuntimePlugin:
    return MockRuntimePlugin(run_id="test")


@pytest.fixture
async def test_plugin_with_time_machine() -> AsyncGenerator[
    tuple[MockRuntimePlugin, time_machine.Coordinates], None
]:
    """Plugin with time-machine at epoch 1000.0, tick=True."""
    with time_machine.travel(1000.0, tick=True) as traveller:
        yield MockRuntimePlugin(run_id="test", traveller=traveller), traveller
