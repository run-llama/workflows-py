# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
A runtime interface to switch out a broker runtime (external library or service that manages durable/distributed step execution).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Generator,
    Protocol,
    cast,
)

from workflows.events import Event, StopEvent
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.step_function import StepWorkerFunction
from workflows.runtime.types.ticks import WorkflowTick
from workflows.workflow import Workflow

# Context variable for implicit runtime scoping
_current_runtime: ContextVar[Runtime | None] = ContextVar(
    "current_runtime", default=None
)


@dataclass
class RegisteredWorkflow:
    workflow_function: ControlLoopFunction
    steps: dict[str, StepWorkerFunction[Any]]


class Runtime(ABC):
    """
    Abstract base class for workflow execution runtimes.

    Runtimes control how workflows are registered, launched, and executed.
    The default BasicRuntime uses asyncio; DBOSRuntime adds durability.

    Lifecycle:
    1. Create runtime instance
    2. Create workflow instances (auto-register with runtime via registering())
    3. Call launch() to start workers/register with backend
    4. Run workflows
    5. Call destroy() to clean up

    Use registering() context manager for implicit workflow registration.
    """

    _token: Token[Runtime | None]

    @abstractmethod
    def register(
        self,
        workflow: Workflow,
        workflow_function: ControlLoopFunction,
        steps: dict[str, StepWorkerFunction[Any]],
    ) -> None | RegisteredWorkflow:
        """
        Register a workflow with the runtime.

        Called at launch() time for each tracked workflow. Runtimes can
        wrap the workflow_function and steps (e.g., with DBOS decorators).

        Returns RegisteredWorkflow with wrapped functions, or None to use originals.
        """
        ...

    @abstractmethod
    def new_runtime(self, run_id: str) -> WorkflowRuntime:
        """
        Create a new runtime instance for a workflow run.

        Called on each workflow.run() to create the runtime that manages
        event delivery, timing, and I/O for that run.
        """
        ...

    def launch(self) -> None:
        """
        Launch the runtime and register all tracked workflows.

        For DBOS, this wraps workflows with decorators and calls DBOS.launch().
        For BasicRuntime, this is a no-op.

        Must be called before running workflows.
        """
        pass

    def destroy(self) -> None:
        """
        Clean up runtime resources.

        Called when done with the runtime. Stops workers, closes connections.
        """
        pass

    def track_workflow(self, workflow: Workflow) -> None:
        """
        Track a workflow instance for registration at launch time.

        Called by Workflow.__init__ to register with the runtime.
        Override in runtimes that need to track workflows (e.g., DBOSRuntime).
        Default implementation is a no-op.
        """
        pass

    def get_registered(self, workflow: Workflow) -> RegisteredWorkflow | None:
        """
        Get the registered workflow if available.

        Returns the pre-registered workflow from launch(), or None if not tracked.
        """
        return None

    @contextmanager
    def registering(self) -> Generator[Runtime, None, None]:
        """
        Context manager for implicit workflow registration.

        Workflows created inside this block will automatically register
        with this runtime. Does NOT call launch() on exit.
        """
        token = _current_runtime.set(self)
        try:
            yield self
        finally:
            _current_runtime.reset(token)


class WorkflowRuntime(Protocol):
    """
    A LlamaIndex workflow's internal state is managed via an event-sourced reducer that triggers step executions. It communicates
    with the outside world via messages. Messages may be sent to it to update its event log, and it in turn publishes messages that are made
    available via the event stream.
    """

    async def send_event(self, tick: WorkflowTick) -> None:
        """Called from outside of the workflow to modify the workflow execution. WorkflowTick events are appended to a mailbox and processed sequentially"""
        ...

    async def wait_receive(self) -> WorkflowTick:
        """called from inside of the workflow control loop function to add a tick event from `send_event` to the mailbox. Function waits until a tick event is received."""
        ...

    async def write_to_event_stream(self, event: Event) -> None:
        """Called from inside of a workflow function to write / emit events to listeners outside of the workflow"""
        ...

    def stream_published_events(self) -> AsyncGenerator[Event, None]:
        """Called from outside of a workflow, reads event stream published by the workflow"""
        ...

    async def get_now(self) -> float:
        """Called from within the workflow control loop function to get the current time in seconds since epoch. If workflow is durable via replay, it should return a cached value from the first call. (e.g. this should be implemented similar to a regular durable step)"""
        ...

    async def sleep(self, seconds: float) -> None:
        """Called from within the workflow control loop function to sleep for a given number of seconds. This should integrate with the host plugin for cases where an inactive workflow may be paused, and awoken later via memoized replay. Note that other tasks in the control loop may still be running simultaneously."""
        ...

    async def close(self) -> None:
        """API that the broker calls to close the plugin after a workflow run is fully complete"""
        ...


class SnapshottableRuntime(WorkflowRuntime, Protocol):
    """
    Snapshot API. Optional mix in to a WorkflowRuntime. When implemented, plugin should offer a replay function to return the recorded
    ticks so that callers can debug or replay the workflow. `on_tick` is called whenever a tick event is received externally OR as a result
    from an internal command (e.g. a step function completing, a timeout occurring, etc.)

    """

    def on_tick(self, tick: WorkflowTick) -> None:
        """Called whenever a tick event is received"""
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
    """
    Protocol for a function that starts and runs the internal control loop for a workflow run.
    Plugin decorators to the control loop function must maintain this signature.
    """

    def __call__(
        self,
        start_event: Event | None,
        init_state: BrokerState | None,
        run_id: str,
    ) -> Coroutine[None, None, StopEvent]: ...
