# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.
"""
DBOS Runtime for durable workflow execution.

This module requires the `dbos` package to be installed.
When imported without dbos, an ImportError will be raised.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from typing import TYPE_CHECKING, Any, AsyncGenerator

from llama_index_instrumentation.dispatcher import active_instrument_tags

if TYPE_CHECKING:
    from workflows.context.serializers import BaseSerializer
    from workflows.context.state_store import InMemoryStateStore

# DBOS is an optional dependency
# Import will fail at runtime if dbos is not installed
from dbos import DBOS, SetWorkflowID

from workflows.events import Event, StartEvent, StopEvent
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    RegisteredWorkflow,
    Runtime,
)
from workflows.runtime.types.step_function import (
    StepWorkerFunction,
    as_step_worker_functions,
    create_workflow_run_function,
)
from workflows.runtime.types.ticks import WorkflowTick
from workflows.runtime.workflow_tracker import WorkflowTracker
from workflows.workflow import Workflow

logger = logging.getLogger(__name__)


@DBOS.step()
async def _durable_time() -> float:
    return time.time()


class DBOSRuntime(Runtime):
    """
    DBOS-backed workflow runtime for durable execution.

    Workflows are registered at launch() time with stable names,
    enabling distributed workers and recovery.
    """

    def __init__(self) -> None:
        self._tracker = WorkflowTracker()
        self._dbos_launched = False
        self._tasks: list[asyncio.Task[None]] = []

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.append(task)
        task.add_done_callback(self._tasks.remove)

    def track_workflow(self, workflow: Workflow) -> None:
        """Track a workflow for registration at launch time."""
        self._tracker.add(workflow)

    def get_registered(self, workflow: Workflow) -> RegisteredWorkflow | None:
        """Get the registered workflow if available."""
        return self._tracker.get_registered(workflow)

    def register(self, workflow: Workflow) -> RegisteredWorkflow:
        """
        Wrap workflow with DBOS decorators.

        Called at launch() time for each tracked workflow.
        Uses workflow.workflow_name for stable DBOS registration names.
        """
        # Use workflow's name directly
        name = workflow.workflow_name

        # Create DBOS-wrapped control loop with stable name
        @DBOS.workflow(name=f"{name}.control_loop")
        async def _dbos_control_loop(
            init_state: BrokerState,
            start_event: StartEvent | None = None,
            tags: dict[str, Any] = {},
        ) -> StopEvent:
            workflow_run_fn = create_workflow_run_function(workflow)
            return await workflow_run_fn(init_state, start_event, tags)

        # Wrap steps with stable names
        wrapped_steps: dict[str, StepWorkerFunction[Any]] = {
            step_name: DBOS.step(name=f"{name}.{step_name}")(step)
            for step_name, step in as_step_worker_functions(workflow).items()
        }

        return RegisteredWorkflow(
            workflow=workflow, workflow_run_fn=_dbos_control_loop, steps=wrapped_steps
        )

    def run_workflow(
        self,
        run_id: str,
        workflow: Workflow,
        init_state: BrokerState,
        start_event: StartEvent | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> ExternalRunAdapter:
        """Set up a workflow run. Currently only creates state store.

        Note: Execution is still managed by the broker for now. This will
        change as we refactor to have the runtime fully own execution.
        """
        from workflows.context.serializers import JsonSerializer
        from workflows.context.state_store import InMemoryStateStore, infer_state_type

        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before running workflows."
            )

        registered = self.get_registered(workflow)
        if registered is None:
            raise RuntimeError(
                "DBOSRuntime workflows must be registered before running. Did you forget to call runtime.launch()?"
            )

        # TODO: Actually have a distributed interface for the state store instead of this sad sorry pretending
        # Create state store from serialized state or infer type from workflow
        active_serializer = serializer or JsonSerializer()
        if serialized_state:
            state_store = InMemoryStateStore.from_dict(
                serialized_state, active_serializer
            )
        else:
            # Infer state type from workflow step configs
            state_type = infer_state_type(registered.workflow)
            state_store = InMemoryStateStore(state_type())
        _dbos_state_stores[run_id] = state_store

        async def _run_workflow() -> None:
            # Capture strong reference to state_store for the task's lifetime,
            # preventing GC while the workflow runs (it's in a WeakValueDictionary).
            _ = state_store
            with SetWorkflowID(run_id):
                try:
                    await DBOS.start_workflow_async(
                        registered.workflow_run_fn,
                        init_state,
                        start_event,
                        active_instrument_tags.get(),
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to submit work to DBOS for {run_id} with start event: {start_event} and init state: {init_state}. Error: {e}",
                        exc_info=True,
                    )
                    raise e

        # fire and forget
        self._track_task(asyncio.create_task(_run_workflow()))

        # Return adapter - execution is handled by broker for now
        return ExternalDBOSAdapter(run_id)

    def get_internal_adapter(self) -> InternalRunAdapter:
        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before running workflows."
            )
        run_id = DBOS.workflow_id
        if run_id is None:
            raise RuntimeError(
                "No current run id. Must be called within a workflow run."
            )
        return InternalDBOSAdapter(run_id)

    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        if not self._dbos_launched:
            raise RuntimeError(
                "DBOS runtime not launched. Call runtime.launch() before running workflows."
            )
        return ExternalDBOSAdapter(run_id)

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

            # Register with DBOS (this applies decorators)
            registered = self.register(workflow)
            if registered is not None:
                self._tracker.set_registered(workflow, registered)

        # Mark as launched (no more workflows can be added)
        self._tracker.mark_launched()

        # Launch DBOS runtime
        DBOS.launch()
        self._dbos_launched = True

    def destroy(self) -> None:
        """Clean up DBOS resources."""
        self._tracker.clear()
        self._dbos_launched = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
        DBOS.destroy()


# Weak reference to state stores by run_id - allows GC when workflow completes
_dbos_state_stores: weakref.WeakValueDictionary[str, "InMemoryStateStore[Any]"] = (
    weakref.WeakValueDictionary()
)


_IO_STREAM_PUBLISHED_EVENTS_NAME = "published_events"
_IO_STREAM_TICK_TOPIC = "ticks"


class InternalDBOSAdapter(InternalRunAdapter):
    """
    Internal DBOS adapter for the workflow control loop.

    - wait_receive gets ticks from the shared mailbox queue
    - write_to_event_stream publishes events via DBOS streams
    - get_now returns a durable timestamp
    - sleep uses DBOS durable sleep
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    async def wait_receive(self) -> WorkflowTick:
        return await DBOS.recv_async(_IO_STREAM_TICK_TOPIC)

    async def write_to_event_stream(self, event: Event) -> None:
        await DBOS.write_stream_async(_IO_STREAM_PUBLISHED_EVENTS_NAME, event)

    async def get_now(self) -> float:
        return await _durable_time()

    async def sleep(self, seconds: float) -> None:
        await DBOS.sleep_async(seconds)

    async def send_event(self, tick: WorkflowTick) -> None:
        await DBOS.send_async(_IO_STREAM_TICK_TOPIC, tick)

    def get_state_store(self) -> "InMemoryStateStore[Any] | None":
        return _dbos_state_stores.get(self._run_id)


class ExternalDBOSAdapter(ExternalRunAdapter):
    """
    External DBOS adapter for workflow interaction.

    - send_event puts ticks into the shared mailbox queue
    - stream_published_events reads from DBOS streams
    - close is a no-op
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        """Get the workflow run ID."""
        return self._run_id

    async def send_event(self, tick: WorkflowTick) -> None:
        await DBOS.send_async(_IO_STREAM_TICK_TOPIC, tick)

    async def stream_published_events(self) -> AsyncGenerator[Event, None]:
        await self._ensure_workflow_started()

        async for event in DBOS.read_stream_async(self.run_id, "published_events"):
            yield event

    async def get_result(self) -> StopEvent:
        await self._ensure_workflow_started()
        handle = await DBOS.retrieve_workflow_async(self.run_id)
        # TODO - do better than basic interval polling. Not a great interface for it in DBOS yet.
        return await handle.get_result()

    async def close(self) -> None:
        pass

    async def _ensure_workflow_started(self) -> None:
        # Wait for the workflow to exist in DBOS before trying to read from its stream.
        workflow_id = self.run_id
        max_wait = 10.0
        poll_interval = 0.01
        max_poll_interval = 1.0
        elapsed = 0.0
        while elapsed < max_wait:
            status = await DBOS.get_workflow_status_async(workflow_id)
            if status is not None:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            poll_interval = min(poll_interval * 2.0, max_poll_interval)
        else:
            raise RuntimeError(
                f"Workflow {workflow_id} did not start within {max_wait} seconds"
            )
