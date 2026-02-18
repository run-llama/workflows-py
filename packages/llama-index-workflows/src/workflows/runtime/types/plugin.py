# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
A runtime interface to switch out a broker runtime (external library or service that manages durable/distributed step execution).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Coroutine,
    Generator,
    Literal,
    Protocol,
    Union,
)

from workflows.context.state_store import StateStore
from workflows.runtime.types.named_task import NamedTask

if TYPE_CHECKING:
    from workflows.context.context import Context
    from workflows.context.serializers import BaseSerializer
    from workflows.runtime.types.internal_state import BrokerState
    from workflows.runtime.types.step_function import StepWorkerFunction
    from workflows.workflow import Workflow

from workflows.events import Event, StartEvent, StopEvent
from workflows.runtime.types.ticks import TickCancelRun, WorkflowTick

# Context variable for implicit runtime scoping
_current_runtime: ContextVar[Runtime | None] = ContextVar(
    "current_runtime", default=None
)


@dataclass
class WaitResultTick:
    """Result containing a received tick."""

    tick: WorkflowTick
    type: Literal["tick"] = "tick"


@dataclass
class WaitResultTimeout:
    """Result indicating timeout expiration."""

    type: Literal["timeout"] = "timeout"


WaitResult = Union[WaitResultTick, WaitResultTimeout]


@dataclass
class RegisteredWorkflow:
    workflow: Workflow
    workflow_run_fn: WorkflowRunFunction
    steps: dict[str, StepWorkerFunction[Any]]


class InternalRunAdapter(ABC):
    """
    Adapter interface for use INSIDE a workflow's control loop.

    This adapter is used by the workflow execution engine (broker) to receive
    ticks from external sources, publish events to listeners, manage timing,
    and perform durable sleeps.

    The InternalRunAdapter is created by Runtime.new_internal_adapter() for each
    workflow run and is passed to the control loop function. It provides the
    internal-facing side of workflow communication:
    - Receiving ticks from the external mailbox (wait_receive)
    - Publishing events that external code can stream (write_to_event_stream)
    - Getting current time with durability support (get_now)
    - Sleeping with durability support (sleep)

    The run_id is always available and required at construction time.
    """

    @property
    @abstractmethod
    def run_id(self) -> str:
        """
        The unique identifier for this workflow run.

        Always available - required at adapter construction time.
        """
        ...

    @abstractmethod
    async def write_to_event_stream(self, event: Event) -> None:
        """
        Publish an event to external listeners.

        Called from inside the workflow to emit events that can be observed
        by external code via the ExternalRunAdapter's stream_published_events().
        """
        ...

    @abstractmethod
    async def get_now(self) -> float:
        """
        Get the current time in seconds since epoch.

        Called from within the workflow control loop. For durable workflows,
        this should return a memoized/replayed value to ensure deterministic
        replay behavior.
        """
        ...

    @abstractmethod
    async def send_event(self, tick: WorkflowTick) -> None:
        """
        Send a tick into the workflow's own mailbox from within the control loop.

        Called from inside the workflow (e.g., from step functions via ctx.send_event)
        to inject events back into the workflow's execution. The tick will be
        received by wait_receive() on the next iteration.
        """
        ...

    @abstractmethod
    async def wait_receive(
        self,
        timeout_seconds: float | None = None,
    ) -> WaitResult:
        """
        Wait for next tick OR timeout expiration.

        This is the primary method for the control loop to wait for events.
        It combines receiving ticks and timeout handling into a single
        deterministic operation.

        Args:
            timeout_seconds: Max time to wait. None means wait indefinitely.

        Returns:
            WaitResultTick if a tick was received
            WaitResultTimeout if timeout expired before receiving tick

        This is a DURABLE operation for durable runtimes:
        - On replay, already-elapsed time is accounted for
        - If timeout already expired in previous run, returns immediately
        """
        ...

    async def close(self) -> None:
        """
        Signal shutdown to wake any blocked wait operations.

        Called during cleanup to allow the adapter to exit gracefully.
        Default is no-op. Custom adapters may send a shutdown signal to wake blocked recv.
        """
        pass

    def get_state_store(self) -> StateStore[Any] | None:
        """
        Get the state store for this workflow run.

        Returns the state store from the runtime, or None if not initialized.
        Default implementation returns None.
        """
        return None

    async def finalize_step(self) -> None:
        """
        Called after a step function completes to perform any adapter-specific cleanup.

        This is called after all background tasks spawned during the step have completed.
        Adapters can override to perform additional finalization (e.g., flush buffers,
        sync state). Default is no-op.
        """
        pass

    async def wait_for_next_task(
        self,
        task_set: list[NamedTask],
        timeout: float | None = None,
    ) -> asyncio.Task[Any] | None:
        """Wait for and return the next task that should complete.

        Args:
            task_set: List of NamedTasks with stable string keys for identification.
                      The order indicates priority - first items should be returned first
                      when multiple tasks complete simultaneously.
            timeout: Timeout in seconds, None for no timeout

        Returns:
            The completed task, or None on timeout.

        IMPORTANT: Must return at most ONE task per call.

        Default implementation uses asyncio.wait(FIRST_COMPLETED) and returns
        the highest-priority completed task (workers before pull).
        Custom adapters may override to coordinate based on journal for
        deterministic replay, using the stable keys from NamedTask to identify tasks.
        """
        tasks = NamedTask.all_tasks(task_set)
        if not tasks:
            return None
        done, _ = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            return None
        # Return the highest-priority completed task (first in task_set order)
        for named_task in task_set:
            if named_task.task in done:
                return named_task.task
        # Fallback (shouldn't happen)
        return done.pop()


class ExternalRunAdapter(ABC):
    """
    Adapter interface for use OUTSIDE a workflow's control loop.

    This adapter is used by external code (e.g., HTTP handlers, client code)
    to interact with a running workflow - sending events into the workflow
    and streaming events published by the workflow.

    The ExternalRunAdapter is created by Runtime.new_external_adapter() and
    provides the external-facing side of workflow communication:
    - Sending ticks into the workflow mailbox (send_event)
    - Streaming events published by the workflow (stream_published_events)
    - Cleaning up resources when done (close)

    The run_id is always available and matches the internal adapter's run_id.
    """

    @property
    @abstractmethod
    def run_id(self) -> str:
        """
        The unique identifier for this workflow run.

        Always available - matches the InternalRunAdapter's run_id.
        """
        ...

    @abstractmethod
    async def send_event(self, tick: WorkflowTick) -> None:
        """
        Send a tick into the workflow mailbox.

        Called from outside the workflow to inject events into the workflow's
        execution. The tick will be received by the internal adapter's
        wait_receive() method.
        """
        ...

    @abstractmethod
    def stream_published_events(self) -> AsyncGenerator[Event, None]:
        """
        Stream events published by the workflow.

        Called from outside the workflow to observe events emitted by the
        workflow via the internal adapter's write_to_event_stream().
        Returns an async generator that yields events as they are published.
        """
        ...

    async def close(self) -> None:
        """
        Clean up adapter resources.

        Called when done interacting with the workflow to release any
        resources held by this adapter (e.g., close streams, release locks).
        """

        pass

    @abstractmethod
    async def get_result(self) -> StopEvent:
        """
        Get the result of the workflow run, if completed. Will raise if the workflow failed or was cancelled
        """
        ...

    async def cancel(self) -> None:
        """
        Cancel the workflow run if it is still running.
        """
        await self.send_event(TickCancelRun())

    def get_state_store(self) -> StateStore[Any] | None:
        """
        Get the state store for this workflow run.

        Returns the state store if this adapter owns it, or None if state
        is managed externally. Default implementation returns None.
        """
        return None


@dataclass
class RunContext:
    """Context for an active workflow run, available via get_current_run()."""

    workflow: Workflow
    run_adapter: InternalRunAdapter
    context: Context
    steps: dict[str, StepWorkerFunction[Any]]


_current_run: ContextVar[RunContext | None] = ContextVar("current_run", default=None)


@contextmanager
def run_context(ctx: RunContext) -> Generator[RunContext, None, None]:
    """Set the current run context for the duration of a workflow run."""
    token = _current_run.set(ctx)
    try:
        yield ctx
    finally:
        _current_run.reset(token)


def get_current_run() -> RunContext:
    """Get the current run context. Raises if not in a workflow run."""
    ctx = _current_run.get()
    if ctx is None:
        raise RuntimeError("Not in a workflow run context")
    return ctx


class Runtime(ABC):
    """
    Abstract base class for workflow execution runtimes.

    Runtimes control how workflows are registered, launched, and executed.
    The default BasicRuntime uses asyncio; Other's plug into their own durability and distributed execution models.

    Lifecycle:
    1. Create runtime instance
    2. Create workflow instances (auto-register with runtime via registering())
    3. Call launch() to start workers/register with backend
    4. Run workflows
    5. Call destroy() to clean up

    Use registering() context manager for implicit workflow registration.
    """

    _token: Token[Runtime | None]

    def get_or_register(self, workflow: "Workflow") -> RegisteredWorkflow:
        """Get the registered workflow if available, otherwise register it."""
        registered = self.get_registered(workflow)
        if registered is None:
            registered = self.register(workflow)
        return registered

    @abstractmethod
    def register(self, workflow: "Workflow") -> RegisteredWorkflow:
        """
        Register a workflow with the runtime.

        Called at launch() time for each tracked workflow. Runtimes can
        wrap the control_loop and steps to fit in their registration/decoration model.

        Returns RegisteredWorkflow with wrapped functions
        """
        ...

    @abstractmethod
    def run_workflow(
        self,
        run_id: str,
        workflow: Workflow,
        init_state: BrokerState,
        start_event: StartEvent | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> ExternalRunAdapter:
        """
        Launch a workflow run.

        The runtime creates and owns the state store based on serialized_state.
        Returns the external adapter for the workflow run.

        Args:
            run_id: Unique identifier for this workflow run.
            registered: The registered workflow to run.
            init_state: Initial broker state (queues, workers, etc).
            start_event: Optional start event to begin the workflow.
            serialized_state: Serialized state store data to restore from.
            serializer: Serializer to use for deserializing state.
        """
        ...

    @abstractmethod
    def get_internal_adapter(self, workflow: "Workflow") -> InternalRunAdapter:
        """
        Get the internal adapter for a workflow run.

        Called on each workflow.run() to instantiate an interface for the workflow run internals to communicate with the runtime.

        Args:
            workflow: The workflow instance being run. Used by runtimes to access workflow metadata (e.g., state type).
        """
        ...

    @abstractmethod
    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        """
        Get the external adapter for a workflow run.

        Called after launching a workflow run, or when getting a handle for an existing workflow run.
        Used to send events into the workflow and stream published events.

        The run_id must match the internal adapter's run_id for the same run.
        The external adapter is used by client code interacting with the workflow.
        """
        ...

    def launch(self) -> None:
        """
        Launch the runtime and register all tracked workflows.

        For many runtime's, this must be called before running workflows.
        """
        pass

    def destroy(self) -> None:
        """
        Clean up runtime resources.

        Called when done with the runtime. Stops workers, closes connections.
        """
        pass

    def track_workflow(self, workflow: "Workflow") -> None:
        """
        Track a workflow instance for registration at launch time.

        Called by Workflow.__init__ to register with the runtime.
        Override in runtimes that need to track workflows.
        Default implementation is a no-op.
        """
        pass

    def get_registered(self, workflow: "Workflow") -> RegisteredWorkflow | None:
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


class SnapshottableAdapter(ABC):
    """
    Mixin interface that adds snapshot/replay capabilities to adapters.

    This is a standalone mixin (not inheriting from InternalRunAdapter or
    ExternalRunAdapter) that can be combined with adapter implementations
    to add tick recording for debugging or replay purposes.

    Adapters that implement this interface can record ticks as they occur
    and replay them later. `on_tick` is called whenever a tick event is
    received externally OR as a result from an internal command (e.g., a
    step function completing, a timeout occurring, etc.)

    Use `as_snapshottable_adapter()` to check if an adapter supports snapshotting.
    """

    @property
    @abstractmethod
    def init_state(self) -> "BrokerState":
        """
        Get the initial state of the adapter.
        """
        ...

    @abstractmethod
    def on_tick(self, tick: WorkflowTick) -> None:
        """
        Called whenever a tick event is received.

        This method is invoked for both external ticks (sent via send_event)
        and internal ticks (generated by step completions, timeouts, etc.).
        Implementations should record the tick for later replay.
        """
        ...

    @abstractmethod
    def replay(self) -> list[WorkflowTick]:
        """
        Return the recorded ticks for replay.

        Returns all ticks that were recorded via on_tick(), in the order
        they were received. Used for debugging and workflow replay.
        """
        ...


def as_snapshottable_adapter(
    adapter: ExternalRunAdapter | InternalRunAdapter,
) -> SnapshottableAdapter | None:
    """
    Check if an internal adapter supports snapshotting.

    Returns the adapter cast to SnapshottableAdapter if it implements
    the snapshotting interface, or None otherwise.
    """
    if isinstance(adapter, SnapshottableAdapter):
        return adapter
    return None


class V2RuntimeCompatibilityShim(ABC):
    """
    This interface will be deleted in V3. Temporary shim to support deprecated v2 functionality
    """

    @abstractmethod
    def get_result_or_none(self) -> StopEvent | None:
        """
        Get the result of the workflow run, if completed. Will raise if the workflow failed or was cancelled, otherwise return None if still running
        """
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the workflow run is still running.
        """
        ...

    @abstractmethod
    def abort(self) -> None:
        """
        Forcefully abort the workflow execution (ungraceful hard cancel).

        This immediately terminates execution by cancelling the underlying task.
        Unlike cancel() which sends a graceful cancellation signal:
        - In-flight step work is cancelled immediately
        - No WorkflowCancelledEvent is emitted
        - The workflow does not finalize gracefully

        This is deprecated v2 behavior - prefer cancel_run() for graceful cancellation.
        """
        ...


def as_v2_runtime_compatibility_shim(
    adapter: ExternalRunAdapter,
) -> V2RuntimeCompatibilityShim | None:
    """
    Check if an adapter supports the V2 runtime compatibility shim.
    """
    if isinstance(adapter, V2RuntimeCompatibilityShim):
        return adapter
    return None


class ControlLoopFunction(Protocol):
    """
    Protocol for a function that starts and runs the internal control loop for a workflow run.
    Runtime decorators to the control loop function must maintain this signature.
    """

    def __call__(
        self,
        start_event: Event | None,
        init_state: BrokerState | None,
        run_id: str,
    ) -> Coroutine[None, None, StopEvent]: ...


class WorkflowRunFunction(Protocol):
    """
    Protocol for a function that runs a workflow. Wraps a control loop function with glue to the runtime.
    """

    def __call__(
        self,
        init_state: BrokerState,
        start_event: StartEvent | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Coroutine[None, None, StopEvent]: ...
