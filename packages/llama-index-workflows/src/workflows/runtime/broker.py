# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Type,
    TypeVar,
    cast,
)

from llama_index_instrumentation.dispatcher import (
    active_instrument_tags,
    instrument_tags,
)
from workflows.errors import WorkflowRuntimeError
from workflows.events import (
    Event,
    StartEvent,
)
from workflows.handler import WorkflowHandler
from workflows.runtime.control_loop import control_loop, rebuild_state_from_ticks
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import Plugin, WorkflowRuntime, as_snapshottable
from workflows.runtime.types.results import (
    AddCollectedEvent,
    AddWaiter,
    DeleteCollectedEvent,
    DeleteWaiter,
    StepWorkerContext,
    StepWorkerStateContextVar,
    WaitingForEvent,
)
from workflows.runtime.types.step_function import (
    StepWorkerFunction,
    as_step_worker_function,
)
from workflows.runtime.types.ticks import TickAddEvent, TickCancelRun, WorkflowTick
from workflows.runtime.workflow_registry import workflow_registry
from workflows.utils import _nanoid as nanoid

from ..context.state_store import MODEL_T

if TYPE_CHECKING:
    from workflows import Workflow
    from workflows.context.context import Context


T = TypeVar("T", bound=Event)
EventBuffer = dict[str, list[Event]]

logger = logging.getLogger()


# Only warn once about unserializable keys
class UnserializableKeyWarning(Warning):
    pass


class WorkflowBroker(Generic[MODEL_T]):
    """
    The workflow broker manages starting up and connecting a workflow handler, a runtime, and triggering the
    execution of the workflow. From there it manages communication between the workflow and the outside world.
    """

    _context: Context[MODEL_T]
    _runtime: WorkflowRuntime
    _plugin: Plugin
    _is_running: bool
    _handler: WorkflowHandler | None
    _workflow: Workflow
    # transient tasks to run async ops in background, exposing sync interfaces
    _workers: list[asyncio.Task]
    _init_state: BrokerState | None

    def __init__(
        self,
        workflow: Workflow,
        context: Context[MODEL_T],
        runtime: WorkflowRuntime,
        plugin: Plugin,
    ) -> None:
        self._context = context
        self._runtime = runtime
        self._plugin = plugin
        self._is_running = False
        self._handler = None
        self._workflow = workflow
        self._workers = []
        self._init_state = None

    def _execute_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._workers.append(task)

        def _remove_task(_: asyncio.Task[Any]) -> None:
            try:
                self._workers.remove(task)
            except ValueError:
                # Handle Task was already cleared during shutdown or cleanup.
                pass

        task.add_done_callback(_remove_task)
        return task

    # context API only
    def start(
        self,
        workflow: Workflow,
        previous: BrokerState | None = None,
        start_event: StartEvent | None = None,
        before_start: Callable[[], Awaitable[None]] | None = None,
        after_complete: Callable[[], Awaitable[None]] | None = None,
    ) -> WorkflowHandler:
        """Start the workflow run. Can only be called once."""
        if self._handler is not None:
            raise WorkflowRuntimeError(
                "this WorkflowBroker already run or running. Cannot start again."
            )
        self._init_state = previous

        async def _run_workflow(run_id: str, tags: dict[str, Any]) -> None:
            with instrument_tags({"run_id": run_id, **tags}):
                # defer execution to make sure the task can be captured and passed
                # to the handler as async exception, protecting against exceptions from before_start
                self._is_running = True
                await asyncio.sleep(0)
                if before_start is not None:
                    await before_start()
                try:
                    init_state = previous or BrokerState.from_workflow(workflow)

                    try:
                        exception_raised = None

                        step_workers: dict[str, StepWorkerFunction] = {}
                        for name, step_func in workflow._get_steps().items():
                            # Avoid capturing a bound method (which retains the instance).
                            # If it's a bound method, extract the unbound function from the class.
                            unbound = getattr(step_func, "__func__", step_func)
                            step_workers[name] = as_step_worker_function(unbound)

                        registered = workflow_registry.get_registered_workflow(
                            workflow, self._plugin, control_loop, step_workers
                        )

                        # Register run context prior to invoking control loop
                        workflow_registry.register_run(
                            run_id=run_id,
                            workflow=workflow,
                            plugin=self._runtime,
                            context=self._context,  # type: ignore
                            steps=registered.steps,
                        )

                        try:
                            workflow_result = await registered.workflow_function(
                                start_event,
                                init_state,
                                run_id,
                            )
                        finally:
                            # ensure run context is cleaned up even on failure
                            workflow_registry.delete_run(run_id)
                        result._set_stop_event(workflow_result)
                    except Exception as e:
                        exception_raised = e

                    if exception_raised:
                        # cancel the stream
                        if not result.done():
                            result.set_exception(exception_raised)
                finally:
                    if after_complete is not None:
                        await after_complete()
                    self._is_running = False

        # Start the machinery in a new Context or use the provided one
        run_id = nanoid()

        # If a previous context is provided, pass its serialized form

        run_task = self._execute_task(
            _run_workflow(run_id, tags=active_instrument_tags.get())
        )
        result = WorkflowHandler(
            ctx=self._context,  # type: ignore
            run_id=run_id,
            run_task=run_task,
        )
        self._handler = result
        return result

    # outer handler API to cancel the workflow run
    def cancel_run(self) -> None:
        self._execute_task(self._runtime.send_event(TickCancelRun()))

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def _state(self) -> BrokerState:
        ticks = self._tick_log
        state = self._init_state or BrokerState.from_workflow(self._workflow)
        new_state = rebuild_state_from_ticks(state, ticks)
        return new_state

    @property
    def _tick_log(self) -> list[WorkflowTick]:
        snapshottable = as_snapshottable(self._runtime)
        if snapshottable is None:
            raise WorkflowRuntimeError("Plugin is not snapshottable")
        return snapshottable.replay()

    # mostly a debug API. May be removed in the future.
    async def running_steps(self) -> list[str]:
        return [
            step
            for step in self._state.workers.keys()
            if self._state.workers[step].in_progress
        ]

    # step api only
    def collect_events(
        self, ev: Event, expected: list[Type[Event]], buffer_id: str | None = None
    ) -> list[Event] | None:
        step_ctx = self._get_step_ctx(fn="collect_events")

        # If no events are expected, return an empty list immediately
        if not expected:
            return []

        buffer_id = buffer_id or "default"

        collected_events = step_ctx.state.collected_events.get(buffer_id, [])

        remaining_event_types = Counter(expected) - Counter(
            [type(e) for e in collected_events]
        )

        if remaining_event_types != Counter([type(ev)]):
            if type(ev) in remaining_event_types:
                step_ctx.returns.return_values.append(
                    AddCollectedEvent(event_id=buffer_id, event=ev)
                )
            return None

        total = []
        by_type = defaultdict(list)
        for e in collected_events + [ev]:
            by_type[type(e)].append(e)
        # order by expected type
        for e_type in expected:
            total.append(by_type[e_type].pop(0))
        # if we got here, it means the collection is fulfilled. Clear the collected events when the step is complete
        step_ctx.returns.return_values.append(DeleteCollectedEvent(event_id=buffer_id))
        return total

    # may be called from both step API and outer handler API
    def send_event(self, message: Event, step: str | None = None) -> None:
        if step is not None:
            if step not in self._workflow._get_steps():
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            # Validate that the step accepts this event type
            step_func = self._workflow._get_steps()[step]
            step_config = step_func._step_config
            if type(message) not in step_config.accepted_events:
                raise WorkflowRuntimeError(
                    f"Step {step} does not accept event of type {type(message)}"
                )

        self._execute_task(
            self._runtime.send_event(TickAddEvent(event=message, step_name=step))
        )

    def _get_step_ctx(self, fn: str) -> StepWorkerContext:
        try:
            return StepWorkerStateContextVar.get()
        except LookupError:
            raise WorkflowRuntimeError(
                f"{fn} may only be called from within a step function"
            )

    # step api only
    async def wait_for_event(
        self,
        event_type: Type[T],
        waiter_event: Event | None = None,
        waiter_id: str | None = None,
        requirements: dict[str, Any] | None = None,
        timeout: float | None = 2000,
    ) -> T:
        step_ctx = self._get_step_ctx(fn="wait_for_event")

        collected_waiters = step_ctx.state.collected_waiters
        requirements = requirements or {}

        # Generate a unique key for the waiter
        event_str = self._get_full_path(event_type)
        requirements_str = str(requirements)
        waiter_id = waiter_id or f"waiter_{event_str}_{requirements_str}"

        waiter = next((w for w in collected_waiters if w.waiter_id == waiter_id), None)
        if waiter is None or waiter.resolved_event is None:
            raise WaitingForEvent(
                AddWaiter(
                    waiter_id=waiter_id,
                    requirements=requirements,
                    timeout=timeout,
                    event_type=event_type,
                    waiter_event=waiter_event,
                )
            )
        else:
            step_ctx.returns.return_values.append(DeleteWaiter(waiter_id=waiter_id))
            return cast(T, waiter.resolved_event)

    def _get_full_path(self, ev_type: Type[Event]) -> str:
        return f"{ev_type.__module__}.{ev_type.__name__}"

    def stream_published_events(self) -> AsyncGenerator[Event, None]:
        """The internal queue used for streaming events to callers."""
        return self._runtime.stream_published_events()

    # step API only
    def write_event_to_stream(self, ev: Event | None) -> None:
        if ev is not None:
            self._execute_task(self._runtime.write_to_event_stream(ev))

    async def shutdown(self) -> None:
        """Cancels the running workflow loop

        Cancels all outstanding workers, waits for them to finish, and marks the
        broker as not running. Queues and state remain available so callers can
        inspect or drain leftover events.
        """
        await self._runtime.send_event(TickCancelRun())
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        await self._runtime.close()
