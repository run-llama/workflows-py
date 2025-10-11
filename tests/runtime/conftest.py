# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

"""Test fixtures and utilities for runtime tests."""

import asyncio
import time
from typing import Any, AsyncGenerator

import pytest

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
    """
    Mock implementation of BrokerRuntimePlugin for unit testing control loops.

    This plugin uses asyncio queues to simulate external event sending and receiving,
    allowing tests to control the flow of events in a deterministic way.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        # Queue for events sent from external sources (e.g., via send_event)
        self._external_queue: asyncio.Queue[WorkflowTick] = asyncio.Queue()
        # Queue for events published to the event stream (e.g., for UI/callbacks)
        self._event_stream: asyncio.Queue[Event] = asyncio.Queue()
        # Current time in seconds, can be advanced manually for testing
        self._current_time: float = time.time()

    async def close(self) -> None:
        """
        Close the plugin.
        """
        pass

    async def wait_receive(self) -> WorkflowTick:
        """
        Waits until an external event is sent via send_event.

        Returns:
            The next external event in the queue
        """
        return await self._external_queue.get()

    async def write_to_event_stream(self, event: Event) -> None:
        """
        Publishes an event to the event stream for external consumers.

        Args:
            event: The event to publish
        """
        await self._event_stream.put(event)

    async def stream_published_events(self) -> AsyncGenerator[Event, None]:
        """
        Streams published events from the workflow.
        """
        while True:
            item = await self._event_stream.get()
            yield item
            if isinstance(item, StopEvent):
                break

    async def send_event(self, tick: WorkflowTick) -> None:
        """
        Sends an event from the external world into the workflow.

        Args:
            tick: The tick/event to send to the workflow
        """
        await self._external_queue.put(tick)

    async def register_step_worker(self, step_name: str, step_worker: Any) -> Any:
        """
        No-op registration for testing - returns the step worker unchanged.

        Args:
            step_name: Name of the step
            step_worker: The step worker function

        Returns:
            The unmodified step worker
        """
        return step_worker

    async def register_workflow_function(self, workflow_function: Any) -> Any:
        """
        No-op registration for testing - returns the workflow function unchanged.

        Args:
            workflow_function: The workflow function

        Returns:
            The unmodified workflow function
        """
        return workflow_function

    async def get_now(self) -> float:
        """
        Returns the current time for the test.

        Returns:
            Current time in seconds since epoch
        """
        return self._current_time

    async def sleep(self, seconds: float) -> None:
        """
        Simulates sleeping for a given duration.

        Args:
            seconds: Number of seconds to sleep
        """
        await asyncio.sleep(seconds)

    def advance_time(self, seconds: float) -> None:
        """
        Advances the mock time for testing purposes.

        Args:
            seconds: Number of seconds to advance
        """
        self._current_time += seconds

    async def get_stream_event(self, timeout: float = 1.0) -> Event:
        """
        Helper to get the next event from the event stream.

        Args:
            timeout: Maximum time to wait for an event

        Returns:
            The next event from the stream

        Raises:
            asyncio.TimeoutError: If no event is received within timeout
        """
        return await asyncio.wait_for(self._event_stream.get(), timeout=timeout)

    def has_stream_events(self) -> bool:
        """
        Check if there are any events in the stream queue.

        Returns:
            True if there are events waiting
        """
        return not self._event_stream.empty()


@pytest.fixture
async def test_plugin() -> MockRuntimePlugin:
    """
    Provides a test runtime plugin for control loop tests.

    Returns:
        A fresh MockRuntimePlugin instance
    """
    return MockRuntimePlugin(run_id="test")
