# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    AsyncGenerator,
    Callable,
    Coroutine,
    Generic,
    Protocol,
    TYPE_CHECKING,
    cast,
)


from workflows.decorators import P, R
from workflows.events import Event, StopEvent

from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.step_function import StepWorkerFunction
from workflows.runtime.types.ticks import WorkflowTick

if TYPE_CHECKING:
    from workflows.workflow import Workflow
    from workflows.context.context import Context


@dataclass
class RegisteredWorkflow(Generic[P, R]):
    workflow_function: Callable[P, R]
    steps: dict[str, StepWorkerFunction]


class Plugin(Protocol):
    def register(
        self,
        workflow: Workflow,
        workflow_function: ControlLoopFunction,
        steps: dict[str, StepWorkerFunction],
    ) -> None | RegisteredWorkflow:
        """
        Called on a workflow before the first time it is run, in order to register it within the plugin's runtime.

        Provides an opportunity to modify the workflow function and steps, e.g. to a version that is wrapped with the plugin's runtime.
        """
        ...

    def new_runtime(self, run_id: str) -> WorkflowRuntime:
        """
        Called on each workflow run, in order to create a new runtime instance for driving the workflow via the plugin's runtime.
        """
        ...


class WorkflowRuntime(Protocol):
    """
    A plugin interface to switch out a broker runtime (external library or service that manages durable/distributed step execution)
    """

    async def wait_receive(self) -> WorkflowTick:
        """Waits until a send_event event is received (e.g. from the outside world)"""
        ...

    def stream_published_events(self) -> AsyncGenerator[Event, None]:
        """Waits until a stream event published by the workflow. Called from outside of the workflow execution (e.g. read from the event stream)"""
        ...

    async def write_to_event_stream(self, event: Event) -> None:
        """API that the workflow or its steps call to publish streaming events for external listeners. Only called from inside of the workflow execution or within steps"""
        ...

    async def send_event(self, tick: WorkflowTick) -> None:
        """called externally to control the workflow from the outside world"""
        ...

    async def get_now(self) -> float:
        """API that the broker calls to get the current time in seconds since epoch. If workflow is durable via replay, it should return a cached value from the first call."""
        ...

    async def sleep(self, seconds: float) -> None:
        """API that the broker calls to sleep for a given number of seconds"""
        ...

    async def close(self) -> None:
        """API that the broker calls to close the plugin after a workflow run is complete"""
        ...


class SnapshottableRuntime(WorkflowRuntime, Protocol):
    """Snapshot API. Not required to implement on a WorkflowRuntime if the plugin has some other durability guarantees."""

    def on_tick(self, tick: WorkflowTick) -> None:
        """record the tick for replay"""
        ...

    def replay(self) -> list[WorkflowTick]:
        """return the recorded ticks for replay"""
        ...


def as_snapshottable(runtime: WorkflowRuntime) -> SnapshottableRuntime | None:
    """Check if a runtime is snapshottable."""
    if (
        getattr(runtime, "on_tick", None) is not None
        and getattr(runtime, "replay", None) is not None
    ):
        return cast(SnapshottableRuntime, runtime)
    return None


class ControlLoopFunction(Protocol):
    def __call__(
        self,
        start_event: Event | None,
        init_state: BrokerState | None,
        # TODO - get these 3 out of here! Needs to be inferred from scope somehow for proper distributed, static execution
        plugin: WorkflowRuntime,
        context: Context,
        step_workers: dict[str, StepWorkerFunction],
    ) -> Coroutine[None, None, StopEvent]: ...
