# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    Generic,
    Type,
    TypeVar,
    cast,
)

from llama_index_instrumentation.dispatcher import Dispatcher

from workflows.decorators import StepConfig, StepFunction, step
from workflows.errors import (
    WorkflowCancelledByUser,
    WorkflowDone,
    WorkflowRuntimeError,
    WorkflowTimeoutError,
)
from workflows.events import (
    Event,
    EventsQueueChanged,
    InputRequiredEvent,
    StepState,
    StepStateChanged,
    StopEvent,
)
from workflows.resource import ResourceManager
from workflows.types import RunResultT

from ..context.state_store import MODEL_T, DictState
from workflows.runtime.state import WorkflowBrokerState

if TYPE_CHECKING:  # pragma: no cover
    from workflows import Workflow
    from workflows.handler import WorkflowHandler
    from workflows.context.context import Context

T = TypeVar("T", bound=Event)
EventBuffer = dict[str, list[Event]]

logger = logging.getLogger()


# Only warn once about unserializable keys
class UnserializableKeyWarning(Warning):
    pass


class WorkflowBroker(Generic[MODEL_T]):
    """
    A workflow broker executes a single workflow run's internal execution. It's the core engine that powers workflow events.

    It contains a small handful of methods for sending and receiving events.
    """

    # the context that we are running
    _context: Context[MODEL_T]

    # These keys are set by pre-built workflows and
    # are known to be unserializable in some cases.
    known_unserializable_keys = ("memory",)

    # Current runtime / transient state
    # Background worker tasks (per-step and cancel worker)
    _tasks: set[asyncio.Task]
    # Flag to cooperatively cancel a running workflow
    _cancel_flag: asyncio.Event
    # Per-step wake flags for workers
    _step_flags: dict[str, asyncio.Event]
    # Transient coordination buffer for step events
    _step_events_holding: list[Event] | None
    # Lock for step events
    _step_lock: asyncio.Lock
    # Final result set by StopEvent. Internally public - Workflow / handler access this to set it's own result value
    _retval: RunResultT

    # workflow specific configs
    # (step_name, step_config) pairs
    _step_configs: dict[str, StepConfig | None]

    # Global lock for concurrent mutations
    _lock: asyncio.Lock

    # Instrumentation dispatcher for step spans
    _dispatcher: Dispatcher

    # Coordination primitive for step lifecycle changes
    _step_condition: asyncio.Condition
    # Coordination primitive for event stream writes
    _step_event_written: asyncio.Condition
    # Counts of active workers per step
    _currently_running_steps: DefaultDict[str, int]

    # serializable state stuff
    _state: WorkflowBrokerState

    # run state
    _handler: WorkflowHandler | None

    def __init__(
        self,
        workflow: Workflow,
        context: Context[MODEL_T],
        state: WorkflowBrokerState,
        run_id: str,
    ) -> None:
        self._context = context
        self._handler = None

        # Store the step configs of this workflow, to be used in send_event
        self._step_configs = {}

        # Make the system step "_done" accept custom stop events
        done_step = self._done
        done_accepted_events = done_step._step_config.accepted_events
        if workflow._stop_event_class not in done_accepted_events:
            done_accepted_events.append(workflow._stop_event_class)

        steps: dict[str, StepFunction] = {
            "_done": done_step,
            **workflow._get_steps(),
        }
        for step_name, step_func in steps.items():
            self._step_configs[step_name] = step_func._step_config

        # Transient runtime fields (always reinitialized)
        self._tasks = set()
        self._cancel_flag = asyncio.Event()
        self._step_flags = {}
        self._step_events_holding = None
        self._step_lock = asyncio.Lock()
        self._retval = None

        self._lock = asyncio.Lock()

        self._dispatcher = workflow._dispatcher

        self._step_condition = asyncio.Condition(lock=self._step_lock)
        self._step_event_written = asyncio.Condition(lock=self._step_lock)
        # Keep track of the steps currently running (transient)
        self._currently_running_steps = defaultdict(int)
        # Default initial values for persistent fields

        self._state = state

        # initialize running state from workflow
        for name, step_func in steps.items():
            if name not in self._state.queues:
                self._state.queues[name] = asyncio.Queue()

            if name not in self._step_flags:
                self._step_flags[name] = asyncio.Event()

            step_config: StepConfig = step_func._step_config

            for _ in range(step_config.num_workers):
                self._add_step_worker(
                    name=name,
                    step=step_func,
                    config=step_config,
                    verbose=workflow._verbose,
                    run_id=run_id,
                    worker_id=str(uuid.uuid4()),
                    resource_manager=workflow._resource_manager,
                )

        # add dedicated cancel task
        self._add_cancel_worker()

    @step(num_workers=1)
    async def _done(self, ev: StopEvent) -> None:
        """Tears down the whole workflow and stop execution."""
        result = ev.result if type(ev) is StopEvent else ev
        self.finalize_run(ev, result)
        self.write_event_to_stream(ev)
        # Signal we want to stop the workflow
        raise WorkflowDone

    @property
    def is_running(self) -> bool:
        return self._state.is_running

    def start(
        self,
        workflow: Workflow,
        start_event: Event | None = None,
        before_start: Callable[[], Awaitable[None]] | None = None,
        after_complete: Callable[[], Awaitable[None]] | None = None,
    ) -> WorkflowHandler:
        from workflows.handler import WorkflowHandler
        from workflows.context.context import Context

        """Start the workflow run. Can only be called once."""
        if self._handler is not None:
            raise WorkflowRuntimeError(
                "this WorkflowBroker already run or running. Cannot start again."
            )
        if self._state.is_running and start_event is not None:
            raise WorkflowRuntimeError(
                "Workflow already running. Cannot start with a start event."
            )
        elif not self._state.is_running and start_event is None:
            raise WorkflowRuntimeError(
                "Workflow not running. Cannot start without a start event."
            )

        async def _run_workflow() -> None:
            # defer execution to make sure the task can be captured and passed
            # to the handler, protecting against exceptions from before_start
            await asyncio.sleep(0)
            if before_start is not None:
                await before_start()
            try:
                if start_event is not None:
                    self._state.is_running = True
                    self.send_event(start_event)

                done, unfinished = await asyncio.wait(
                    self._tasks,
                    timeout=workflow._timeout,
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                we_done = False
                exception_raised = None
                for task in done:
                    e = task.exception()
                    if type(e) is WorkflowDone:
                        we_done = True
                    elif e is not None:
                        exception_raised = e
                        break

                # Cancel any pending tasks
                for t in unfinished:
                    t.cancel()

                # wait for cancelled tasks to cleanup
                # prevents any tasks from being stuck
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*unfinished, return_exceptions=True),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some tasks did not clean up within timeout")

                # the context is no longer running
                self._state.is_running = False

                if exception_raised:
                    print("exception_raised", exception_raised)
                    # cancel the stream
                    self.write_event_to_stream(StopEvent())

                    raise exception_raised

                if not we_done:
                    # cancel the stream
                    self.write_event_to_stream(StopEvent())

                    msg = f"Operation timed out after {workflow._timeout} seconds"
                    raise WorkflowTimeoutError(msg)

                result.set_result(self._retval)
            except Exception as e:
                if not result.done():
                    result.set_exception(e)
            finally:
                await self.shutdown()
                if after_complete is not None:
                    await after_complete()

        # Start the machinery in a new Context or use the provided one
        run_id = str(uuid.uuid4())

        # If a previous context is provided, pass its serialized form

        run_task = asyncio.create_task(_run_workflow())
        result = WorkflowHandler(
            ctx=cast(Context[DictState], self._context),
            run_id=run_id,
            run_task=run_task,
        )
        self._handler = result
        return result

    def finalize_run(self, event: StopEvent, result: RunResultT) -> None:
        if self._handler is None:
            raise WorkflowRuntimeError("Workflow handler is not set")
        self._retval = result

    def cancel_run(self) -> None:
        self._cancel_flag.set()

    async def running_steps(self) -> list[str]:
        """Return the list of currently running step names.

        Returns:
            list[str]: Names of steps that have at least one active worker.
        """
        async with self._lock:
            return list(self._currently_running_steps)

    def collect_events(
        self, ev: Event, expected: list[Type[Event]], buffer_id: str | None = None
    ) -> list[Event] | None:
        buffer_id = buffer_id or self._get_event_buffer_id(expected)

        if buffer_id not in self._state.event_buffers:
            self._state.event_buffers[buffer_id] = defaultdict(list)

        event_type_path = self._get_full_path(type(ev))
        self._state.event_buffers[buffer_id][event_type_path].append(ev)

        retval: list[Event] = []
        for e_type in expected:
            e_type_path = self._get_full_path(e_type)
            e_instance_list = self._state.event_buffers[buffer_id].get(e_type_path, [])
            if e_instance_list:
                retval.append(e_instance_list.pop(0))
            else:
                # We already know we don't have all the events
                break

        if len(retval) == len(expected):
            return retval

        # put back the events if unable to collect all
        for i, ev_to_restore in enumerate(retval):
            e_type = type(retval[i])
            e_type_path = self._get_full_path(e_type)
            self._state.event_buffers[buffer_id][e_type_path].append(ev_to_restore)

        return None

    def send_event(self, message: Event, step: str | None = None) -> None:
        if step is None:
            for name, queue in self._state.queues.items():
                queue.put_nowait(message)
                self.write_event_to_stream(
                    EventsQueueChanged(name=name, size=queue.qsize())
                )
        else:
            if step not in self._step_configs:
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            step_config = self._step_configs[step]
            if step_config and type(message) in step_config.accepted_events:
                self._state.queues[step].put_nowait(message)
                self.write_event_to_stream(
                    EventsQueueChanged(name=step, size=self._state.queues[step].qsize())
                )
            else:
                raise WorkflowRuntimeError(
                    f"Step {step} does not accept event of type {type(message)}"
                )

        self._state.broker_log.append(message)

    async def wait_for_event(
        self,
        event_type: Type[T],
        waiter_event: Event | None = None,
        waiter_id: str | None = None,
        requirements: dict[str, Any] | None = None,
        timeout: float | None = 2000,
    ) -> T:
        requirements = requirements or {}

        # Generate a unique key for the waiter
        event_str = self._get_full_path(event_type)
        requirements_str = str(requirements)
        waiter_id = waiter_id or f"waiter_{event_str}_{requirements_str}"

        if waiter_id not in self._state.queues:
            self._state.queues[waiter_id] = asyncio.Queue()
            self.write_event_to_stream(
                EventsQueueChanged(
                    name=waiter_id, size=self._state.queues[waiter_id].qsize()
                )
            )

        # send the waiter event if it's not already sent
        if waiter_event is not None:
            is_waiting = waiter_id in self._state.waiting_ids
            if not is_waiting:
                self._state.waiting_ids.add(waiter_id)
                self.write_event_to_stream(waiter_event)

        while True:
            event = await asyncio.wait_for(
                self._state.queues[waiter_id].get(), timeout=timeout
            )
            if type(event) is event_type:
                if all(getattr(event, k, None) == v for k, v in requirements.items()):
                    if waiter_id in self._state.waiting_ids:
                        self._state.waiting_ids.remove(waiter_id)
                    return event
                else:
                    continue
            self.write_event_to_stream(
                EventsQueueChanged(
                    name=waiter_id,
                    size=self._state.queues[waiter_id].qsize(),
                )
            )

    @property
    def streaming_queue(self) -> asyncio.Queue:
        """The internal queue used for streaming events to callers."""
        return self._state.streaming_queue

    def write_event_to_stream(self, ev: Event | None) -> None:
        self._state.streaming_queue.put_nowait(ev)

    async def shutdown(self) -> None:
        """Shut down the workflow run and clean up background tasks.

        Cancels all outstanding workers, waits for them to finish, and marks the
        context as not running. Queues and state remain available so callers can
        inspect or drain leftover events.
        """
        self._state.is_running = False
        # Cancel all running tasks
        for task in self._tasks:
            task.cancel()
        # Wait for all tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _mark_in_progress(
        self, name: str, ev: Event, worker_id: str = ""
    ) -> None:
        """
        Add input event to in_progress dict.

        Args:
            name (str): The name of the step that is in progress.
            ev (Event): The input event that kicked off this step.

        """
        async with self._lock:
            self.write_event_to_stream(
                StepStateChanged(
                    step_state=StepState.IN_PROGRESS,
                    name=name,
                    input_event_name=(str(type(ev))),
                    worker_id=worker_id,
                )
            )
            self._state.in_progress[name].append(ev)

    async def _remove_from_in_progress(
        self, name: str, ev: Event, worker_id: str = ""
    ) -> None:
        """
        Remove input event from active steps.

        Args:
            name (str): The name of the step that has been completed.
            ev (Event): The associated input event that kicked off this completed step.

        """
        async with self._lock:
            self.write_event_to_stream(
                StepStateChanged(
                    step_state=StepState.NOT_IN_PROGRESS,
                    name=name,
                    input_event_name=(str(type(ev))),
                    worker_id=worker_id,
                )
            )
            events = [e for e in self._state.in_progress[name] if e != ev]
            self._state.in_progress[name] = events

    def _get_full_path(self, ev_type: Type[Event]) -> str:
        return f"{ev_type.__module__}.{ev_type.__name__}"

    def _get_event_buffer_id(self, events: list[Type[Event]]) -> str:
        # Try getting the current task name
        try:
            current_task = asyncio.current_task()
            if current_task:
                t_name = current_task.get_name()
                # Do not use the default value 'Task'
                if t_name != "Task":
                    return t_name
        except RuntimeError:
            # This is a sync step, fallback to using events list
            pass

        # Fall back to creating a stable identifier from expected events
        return ":".join(sorted(self._get_full_path(e_type) for e_type in events))

    def _add_step_worker(
        self,
        name: str,
        step: Callable,
        config: StepConfig,
        verbose: bool,
        run_id: str,
        worker_id: str,
        resource_manager: ResourceManager,
    ) -> None:
        """Spawn a background worker task to process events for a step.

        Args:
            name (str): Step name.
            step (Callable): Step function (sync or async).
            config (StepConfig): Resolved configuration for the step.
            verbose (bool): If True, print step activity.
            run_id (str): Run identifier for instrumentation.
            worker_id (str): ID of the worker running the step
            resource_manager (ResourceManager): Resource injector for the step.
        """
        self._tasks.add(
            asyncio.create_task(
                self._step_worker(
                    name=name,
                    step=step,
                    config=config,
                    verbose=verbose,
                    run_id=run_id,
                    worker_id=worker_id,
                    resource_manager=resource_manager,
                ),
                name=name,
            )
        )

    async def _step_worker(
        self,
        name: str,
        step: Callable,
        config: StepConfig,
        verbose: bool,
        run_id: str,
        worker_id: str,
        resource_manager: ResourceManager,
    ) -> None:
        while True:
            ev = await self._state.queues[name].get()
            if type(ev) not in config.accepted_events:
                continue

            if verbose and name != "_done":
                print(f"Running step {name}")

            kwargs: dict[str, Any] = {}
            if config.context_parameter:
                kwargs[config.context_parameter] = self._context
            for resource in config.resources:
                kwargs[resource.name] = await resource_manager.get(
                    resource=resource.resource
                )
            kwargs[config.event_name] = ev

            # wrap the step with instrumentation
            instrumented_step = self._dispatcher.span(step)

            # - check if its async or not
            # - if not async, run it in an executor
            self.write_event_to_stream(
                StepStateChanged(
                    name=name,
                    step_state=StepState.PREPARING,
                    worker_id=worker_id,
                    input_event_name=str(type(ev)),
                )
            )
            if asyncio.iscoroutinefunction(step):
                retry_start_at = time.time()
                attempts = 0
                while True:
                    await self._mark_in_progress(name=name, ev=ev, worker_id=worker_id)
                    # add the running step
                    async with self._lock:
                        self._currently_running_steps[name] += 1
                    # publish the step state changed event for observers
                    self.write_event_to_stream(
                        StepStateChanged(
                            name=name,
                            step_state=StepState.RUNNING,
                            worker_id=worker_id,
                            input_event_name=str(type(ev)),
                        )
                    )
                    try:
                        new_ev = await instrumented_step(**kwargs)
                        kwargs.clear()
                        break  # exit the retrying loop

                    except WorkflowDone:
                        await self._remove_from_in_progress(
                            name=name, ev=ev, worker_id=worker_id
                        )
                        raise
                    except Exception as e:
                        if config.retry_policy is None:
                            raise

                        delay = config.retry_policy.next(
                            retry_start_at + time.time(), attempts, e
                        )
                        if delay is None:
                            raise

                        attempts += 1
                        if verbose:
                            print(
                                f"Step {name} produced an error, retry in {delay} seconds"
                            )
                        await asyncio.sleep(delay)
                    finally:
                        # remove the running step
                        async with self._lock:
                            self._currently_running_steps[name] -= 1
                            if self._currently_running_steps[name] == 0:
                                del self._currently_running_steps[name]
                        # publish the step state changed event for observers
                        self.write_event_to_stream(
                            StepStateChanged(
                                name=name,
                                step_state=StepState.NOT_RUNNING,
                                worker_id=worker_id,
                                input_event_name=str(type(ev)),
                            )
                        )

            else:
                try:
                    run_task = functools.partial(instrumented_step, **kwargs)
                    kwargs.clear()
                    new_ev = await asyncio.get_event_loop().run_in_executor(
                        None, run_task
                    )
                except WorkflowDone:
                    await self._remove_from_in_progress(
                        name=name, ev=ev, worker_id=worker_id
                    )
                    raise
                except Exception as e:
                    raise WorkflowRuntimeError(f"Error in step '{name}': {e!s}") from e

            self.write_event_to_stream(
                StepStateChanged(
                    name=name,
                    step_state=StepState.NOT_IN_PROGRESS,
                    worker_id=worker_id,
                    input_event_name=str(type(ev)),
                )
            )
            if verbose and name != "_done":
                if new_ev is not None:
                    print(f"Step {name} produced event {type(new_ev).__name__}")
                else:
                    print(f"Step {name} produced no event")

            # Store the accepted event for the drawing operations
            if new_ev is not None:
                self._state.accepted_events.append((name, type(ev).__name__))

            # Fail if the step returned something that's not an event
            if new_ev is not None and not isinstance(new_ev, Event):
                msg = f"Step function {name} returned {type(new_ev).__name__} instead of an Event instance."
                raise WorkflowRuntimeError(msg)

            await self._remove_from_in_progress(name=name, ev=ev, worker_id=worker_id)
            self.write_event_to_stream(
                StepStateChanged(
                    name=name,
                    step_state=StepState.EXITED,
                    worker_id=worker_id,
                    input_event_name=str(type(ev)),
                    output_event_name=str(type(new_ev)),
                )
            )
            # InputRequiredEvent's are special case and need to be written to the stream
            # this way, the user can access and respond to the event
            if isinstance(new_ev, InputRequiredEvent):
                self.write_event_to_stream(new_ev)
            elif new_ev is not None:
                self.send_event(new_ev)

    def _add_cancel_worker(self) -> None:
        """Install a worker that turns a cancel flag into an exception.

        When the cancel flag is set, a `WorkflowCancelledByUser` will be raised
        internally to abort the run.

        See Also:
            - [WorkflowCancelledByUser][workflows.errors.WorkflowCancelledByUser]
        """
        self._tasks.add(asyncio.create_task(self._cancel_worker()))

    async def _cancel_worker(self) -> None:
        try:
            await self._cancel_flag.wait()
            raise WorkflowCancelledByUser
        except asyncio.CancelledError:
            return
