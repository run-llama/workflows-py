# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import replace
from typing import TYPE_CHECKING

from workflows.decorators import R
from workflows.errors import (
    WorkflowCancelledByUser,
    WorkflowRuntimeError,
    WorkflowTimeoutError,
)
from workflows.events import (
    Event,
    InputRequiredEvent,
    StartEvent,
    StepState,
    StepStateChanged,
    StopEvent,
    UnhandledEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
    WorkflowIdleEvent,
    WorkflowTimedOutEvent,
)
from workflows.runtime.types.commands import (
    CommandCompleteRun,
    CommandFailWorkflow,
    CommandHalt,
    CommandPublishEvent,
    CommandQueueEvent,
    CommandRunWorker,
    WorkflowCommand,
    indicates_exit,
)
from workflows.runtime.types.internal_state import (
    BrokerState,
    EventAttempt,
    InProgressState,
    InternalStepWorkerState,
)
from workflows.runtime.types.plugin import (
    WorkflowRuntime,
    as_snapshottable,
)
from workflows.runtime.types.results import (
    AddCollectedEvent,
    AddWaiter,
    DeleteCollectedEvent,
    DeleteWaiter,
    StepWorkerFailed,
    StepWorkerResult,
    StepWorkerState,
    StepWorkerWaiter,
)
from workflows.runtime.types.step_function import (
    StepWorkerFunction,
)
from workflows.runtime.types.ticks import (
    TickAddEvent,
    TickCancelRun,
    TickPublishEvent,
    TickStepResult,
    TickTimeout,
    WorkflowTick,
)
from workflows.runtime.workflow_registry import workflow_registry
from workflows.workflow import Workflow

if TYPE_CHECKING:
    from workflows.context.context import Context


logger = logging.getLogger()


class _ControlLoopRunner:
    """
    Private class to encapsulate the async control loop runtime state and behavior.
    Keeps the pure transformation functions at module level for testability.
    """

    def __init__(
        self,
        workflow: Workflow,
        plugin: WorkflowRuntime,
        context: Context,
        step_workers: dict[str, StepWorkerFunction],
        init_state: BrokerState,
    ):
        self.workflow = workflow
        self.plugin = plugin
        self.context = context
        self.step_workers = step_workers
        self.state = init_state
        self.workers: list[asyncio.Task] = []
        self.queue: asyncio.Queue[WorkflowTick] = asyncio.Queue()
        for tick in self.state.rehydrate_with_ticks():
            self.queue.put_nowait(tick)
        self.snapshot_plugin = as_snapshottable(plugin)

    async def wait_for_tick(self) -> WorkflowTick:
        """Wait for the next tick from the internal queue."""
        return await self.queue.get()

    def queue_tick(self, tick: WorkflowTick, delay: float | None = None) -> None:
        """Queue a tick event for processing, optionally after a delay."""
        if delay:

            async def _delayed_queue() -> None:
                await self.plugin.sleep(delay)
                self.queue.put_nowait(tick)

            task = asyncio.create_task(_delayed_queue())
            self.workers.append(task)
        else:
            self.queue.put_nowait(tick)

    def run_worker(self, command: CommandRunWorker) -> None:
        """Run a worker for a step function."""

        async def _run_worker() -> None:
            try:
                worker = next(
                    (
                        w
                        for w in self.state.workers[command.step_name].in_progress
                        if w.worker_id == command.id
                    ),
                    None,
                )
                if worker is None:
                    raise WorkflowRuntimeError(
                        f"Worker {command.id} not found in in_progress. This should not happen."
                    )
                snapshot = worker.shared_state
                step_fn: StepWorkerFunction = self.step_workers[command.step_name]

                result = await step_fn(
                    state=snapshot,
                    step_name=command.step_name,
                    event=command.event,
                    context=self.context,
                    workflow=self.workflow,
                )
                self.queue_tick(
                    TickStepResult(
                        step_name=command.step_name,
                        worker_id=command.id,
                        event=command.event,
                        result=result,
                    )
                )
            except Exception as e:
                logger.error("error running step worker function: ", e, exc_info=True)
                self.queue_tick(
                    TickStepResult(
                        step_name=command.step_name,
                        worker_id=command.id,
                        event=command.event,
                        result=[
                            StepWorkerFailed(
                                exception=e, failed_at=await self.plugin.get_now()
                            )
                        ],
                    )
                )
                raise e

        task = asyncio.create_task(_run_worker())
        task.add_done_callback(lambda _: self.workers.remove(task))
        self.workers.append(task)

    async def process_command(self, command: WorkflowCommand) -> None | StopEvent:
        """Process a single command returned from tick reduction."""
        if isinstance(command, CommandQueueEvent):
            self.queue_tick(
                TickAddEvent(
                    event=command.event,
                    step_name=command.step_name,
                    attempts=command.attempts,
                    first_attempt_at=command.first_attempt_at,
                ),
                delay=command.delay,
            )
            return None
        elif isinstance(command, CommandRunWorker):
            self.run_worker(command)
            return None
        elif isinstance(command, CommandHalt):
            await self.cleanup_tasks()
            if command.exception is not None:
                raise command.exception
        elif isinstance(command, CommandCompleteRun):
            return command.result
        elif isinstance(command, CommandPublishEvent):
            await self.plugin.write_to_event_stream(command.event)
            return None
        elif isinstance(command, CommandFailWorkflow):
            await self.cleanup_tasks()
            raise command.exception
        else:
            raise ValueError(f"Unknown command type: {type(command)}")

    async def cleanup_tasks(self) -> None:
        """Cancel and cleanup all running worker tasks."""
        try:
            for worker in self.workers:
                worker.cancel()
        except Exception:
            pass
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True),
                timeout=0.5,
            )
        except Exception:
            pass

    async def run(
        self, start_event: Event | None = None, start_with_timeout: bool = True
    ) -> StopEvent:
        """
        Run the control loop until completion.

        Args:
            start_event: Optional initial event to process
            start_with_timeout: Whether to start the timeout timer

        Returns:
            The final StopEvent from the workflow
        """

        # Start external event listener
        async def _pull() -> None:
            while True:
                tick = await self.plugin.wait_receive()
                self.queue_tick(tick)

        self.workers.append(asyncio.create_task(_pull()))

        # Queue initial event and timeout
        if start_event is not None:
            self.queue_tick(TickAddEvent(event=start_event))

        if start_with_timeout and self.workflow._timeout is not None:
            self.queue_tick(
                TickTimeout(timeout=self.workflow._timeout),
                delay=self.workflow._timeout,
            )

        # Resume any in-progress work
        self.state, commands = rewind_in_progress(
            self.state, await self.plugin.get_now()
        )
        for command in commands:
            try:
                await self.process_command(command)
            except Exception:
                await self.cleanup_tasks()
                raise

        # Main event loop
        try:
            while True:
                tick = await self.wait_for_tick()
                try:
                    self.state, commands = _reduce_tick(
                        tick, self.state, await self.plugin.get_now()
                    )
                except Exception:
                    await self.cleanup_tasks()
                    logger.error(
                        "Unexpected error in internal control loop of workflow. This shouldn't happen. ",
                        exc_info=True,
                    )
                    raise
                if self.snapshot_plugin is not None:
                    self.snapshot_plugin.on_tick(tick)
                for command in commands:
                    try:
                        result = await self.process_command(command)
                    except Exception:
                        await self.cleanup_tasks()
                        raise

                    if result is not None:
                        return result
        finally:
            await self.cleanup_tasks()


async def control_loop(
    start_event: Event | None,
    init_state: BrokerState | None,
    run_id: str,
) -> StopEvent:
    """
    The main async control loop for a workflow run.
    """
    # Prefer run-scoped context if available (set by broker)
    current = workflow_registry.get_run(run_id)
    if current is None:
        raise WorkflowRuntimeError("Run context not found for control loop")
    state = init_state or BrokerState.from_workflow(current.workflow)
    runner = _ControlLoopRunner(
        current.workflow, current.plugin, current.context, current.steps, state
    )
    return await runner.run(start_event=start_event)


def rebuild_state_from_ticks(
    state: BrokerState,
    ticks: list[WorkflowTick],
) -> BrokerState:
    """Rebuild the state from a list of ticks.

    When reconstructing state (e.g., for checkpointing), we must first apply
    rewind_in_progress() to match what happens at runtime when resuming a workflow.
    This clears in_progress, moves events back to the queue, and then re-assigns
    new worker IDs starting from 0.

    Without this, resuming a workflow and then checkpointing again would fail
    because the original in_progress worker IDs don't match the new worker IDs
    assigned after rewind.
    """
    # Apply rewind_in_progress to match what happens at runtime when resuming.
    # This re-assigns worker IDs so they align with the ticks that were recorded
    # after the workflow was resumed.
    state, _ = rewind_in_progress(state, time.time())

    # Replay ticks to rebuild state
    for tick in ticks:
        state, _ = _reduce_tick(
            tick, state, time.time()
        )  # somewhat broken kludge on the timestamps, need to move these to ticks
    return state


def _reduce_tick(
    tick: WorkflowTick, init: BrokerState, now_seconds: float
) -> tuple[BrokerState, list[WorkflowCommand]]:
    if isinstance(tick, TickStepResult):
        return _process_step_result_tick(tick, init, now_seconds)
    elif isinstance(tick, TickAddEvent):
        return _process_add_event_tick(tick, init, now_seconds)
    elif isinstance(tick, TickCancelRun):
        return _process_cancel_run_tick(tick, init)
    elif isinstance(tick, TickPublishEvent):
        return _process_publish_event_tick(tick, init)
    elif isinstance(tick, TickTimeout):
        return _process_timeout_tick(tick, init)
    else:
        raise ValueError(f"Unknown tick type: {type(tick)}")


def rewind_in_progress(
    state: BrokerState,
    now_seconds: float,
) -> tuple[BrokerState, list[WorkflowCommand]]:
    """Rewind the in_progress state, extracting commands to re-initiate the workers"""
    state = state.deepcopy()
    commands: list[WorkflowCommand] = []
    for step_name, step_state in sorted(state.workers.items(), key=lambda x: x[0]):
        for in_progress in step_state.in_progress:
            step_state.queue.insert(
                0,
                EventAttempt(
                    event=in_progress.event,
                    attempts=in_progress.attempts,
                    first_attempt_at=in_progress.first_attempt_at,
                ),
            )
        step_state.in_progress = []
        while (
            len(step_state.queue) > 0
            and len(step_state.in_progress) < step_state.config.num_workers
        ):
            event = step_state.queue.pop(0)
            commands.extend(
                _add_or_enqueue_event(event, step_name, step_state, now_seconds)
            )
    return state, commands


def _check_idle_state(state: BrokerState) -> bool:
    """Returns True if workflow is idle (waiting only on external events).

    A workflow is idle when:
    1. The workflow is running (hasn't completed/failed/cancelled)
    2. All steps have no pending events in their queues
    3. All steps have no workers currently executing
    4. At least one step has an active waiter (from ctx.wait_for_event())
    """
    if not state.is_running:
        return False

    for worker_state in state.workers.values():
        if worker_state.queue or worker_state.in_progress:
            return False

    return any(ws.collected_waiters for ws in state.workers.values())


def _process_step_result_tick(
    tick: TickStepResult[R], init: BrokerState, now_seconds: float
) -> tuple[BrokerState, list[WorkflowCommand]]:
    """
    processes the results from a step function execution
    """
    state = init.deepcopy()
    commands: list[WorkflowCommand] = []
    worker_state = state.workers[tick.step_name]
    # get the current execution details and mark it as no longer in progress
    this_execution = next(
        (w for w in worker_state.in_progress if w.worker_id == tick.worker_id), None
    )
    if this_execution is None:
        # this should not happen unless there's a logic bug in the control loop
        raise ValueError(f"Worker {tick.worker_id} not found in in_progress")
    output_event_name: str | None = None

    did_complete_step = bool(
        [x for x in tick.result if isinstance(x, StepWorkerResult)]
    )
    step_no_longer_in_progress = True

    for result in tick.result:
        if isinstance(result, StepWorkerResult):
            output_event_name = str(type(result.result))
            if isinstance(result.result, StopEvent):
                # huzzah! The workflow has completed
                commands.append(
                    CommandPublishEvent(event=result.result)
                )  # stop event always published to the stream
                state.is_running = False
                commands.append(CommandCompleteRun(result=result.result))
            elif isinstance(result.result, Event):
                # queue any subsequent events
                # human input required are automatically published to the stream
                if isinstance(result.result, InputRequiredEvent):
                    commands.append(CommandPublishEvent(event=result.result))
                commands.append(CommandQueueEvent(event=result.result))
            elif result.result is None:
                # None means skip
                pass
            else:
                logger.warning(
                    f"Unknown result type returned from step function ({tick.step_name}): {type(result.result)}"
                )
        elif isinstance(result, StepWorkerFailed):
            # Schedulea a retry if permitted, otherwise fail the workflow
            retries = worker_state.config.retry_policy
            failures = this_execution.attempts + 1
            elapsed_time = result.failed_at - this_execution.first_attempt_at
            delay = (
                retries.next(elapsed_time, failures, result.exception)
                if retries is not None
                else None
            )
            if delay is not None:
                commands.append(
                    CommandQueueEvent(
                        event=tick.event,
                        delay=delay,
                        step_name=tick.step_name,
                        attempts=this_execution.attempts + 1,
                        first_attempt_at=this_execution.first_attempt_at,
                    )
                )
            else:
                # Publish a WorkflowFailedEvent to inform stream consumers about the failure
                state.is_running = False
                exception = result.exception
                exc_type = type(exception)
                exc_module = exc_type.__module__
                exc_qualname = f"{exc_module}.{exc_type.__qualname__}"
                exc_traceback = "".join(
                    traceback.format_exception(
                        exc_type, exception, exception.__traceback__
                    )
                )
                total_attempts = this_execution.attempts + 1
                elapsed = result.failed_at - this_execution.first_attempt_at
                commands.append(
                    CommandPublishEvent(
                        event=WorkflowFailedEvent(
                            step_name=tick.step_name,
                            exception_type=exc_qualname,
                            exception_message=str(exception),
                            traceback=exc_traceback,
                            attempts=total_attempts,
                            elapsed_seconds=elapsed,
                        )
                    )
                )
                commands.append(
                    CommandFailWorkflow(step_name=tick.step_name, exception=exception)
                )
        elif isinstance(result, AddCollectedEvent):
            # The current state of collected events.
            collected_events = state.workers[
                tick.step_name
            ].collected_events.setdefault(result.event_id, [])
            # the events snapshot that was sent with the step function execution that yielded this result
            sent_events = this_execution.shared_state.collected_events.get(
                result.event_id, []
            )
            if len(collected_events) > len(sent_events):
                # rerun it, and don't append now to ensure serializability
                # updating the run state
                step_no_longer_in_progress = False
                updated_state = replace(
                    this_execution.shared_state,
                    collected_events={
                        x: list(y)
                        for x, y in state.workers[
                            tick.step_name
                        ].collected_events.items()
                    },
                )
                this_execution.shared_state = updated_state
                commands.append(
                    CommandRunWorker(
                        step_name=tick.step_name,
                        event=result.event,
                        id=this_execution.worker_id,
                    )
                )
            else:
                collected_events.append(result.event)
        elif isinstance(result, DeleteCollectedEvent):
            if did_complete_step:  # allow retries to grab the events
                # indicates that a run has successfully collected its events, and they can be deleted from the collected events state
                state.workers[tick.step_name].collected_events.pop(
                    result.event_id, None
                )
        elif isinstance(result, AddWaiter):
            # indicates that a run has added a waiter to the collected waiters state
            existing = next(
                (
                    (i)
                    for i, x in enumerate(worker_state.collected_waiters)
                    if x.waiter_id == result.waiter_id
                ),
                None,
            )
            new_waiter = StepWorkerWaiter(
                waiter_id=result.waiter_id,
                event=this_execution.event,
                waiting_for_event=result.event_type,  # ty: ignore[invalid-argument-type] - ty choking here, with result.event_type resolved as "object"
                requirements=result.requirements,
                has_requirements=bool(len(result.requirements)),
                resolved_event=None,
            )
            if existing is not None:
                worker_state.collected_waiters[existing] = new_waiter
            else:
                worker_state.collected_waiters.append(new_waiter)
                if result.waiter_event:
                    commands.append(CommandPublishEvent(event=result.waiter_event))

        elif isinstance(result, DeleteWaiter):
            if did_complete_step:  # allow retries to grab the waiter events
                # indicates that a run has obtained the waiting event, and it can be deleted from the collected waiters state
                to_remove = result.waiter_id
                waiters = state.workers[tick.step_name].collected_waiters
                item = next(filter(lambda w: w.waiter_id == to_remove, waiters), None)
                if item is not None:
                    waiters.remove(item)
        else:
            raise ValueError(f"Unknown result type: {type(result)}")

    is_completed = len([x for x in commands if indicates_exit(x)]) > 0
    if step_no_longer_in_progress:
        commands.insert(
            0,
            CommandPublishEvent(
                StepStateChanged(
                    step_state=StepState.NOT_RUNNING,
                    name=tick.step_name,
                    input_event_name=str(type(tick.event)),
                    output_event_name=output_event_name,
                    worker_id=str(tick.worker_id),
                )
            ),
        )
        worker_state.in_progress.remove(this_execution)
    # enqueue next events if there are any
    if not is_completed:
        while (
            len(worker_state.queue) > 0
            and len(worker_state.in_progress) < worker_state.config.num_workers
        ):
            event = worker_state.queue.pop(0)
            subcommands = _add_or_enqueue_event(
                event, tick.step_name, worker_state, now_seconds
            )
            commands.extend(subcommands)

    # Check for idle transition at end of processing
    was_idle = _check_idle_state(init)
    now_idle = _check_idle_state(state)

    if now_idle and not was_idle:
        commands.append(CommandPublishEvent(WorkflowIdleEvent()))

    return state, commands


def _add_or_enqueue_event(
    event: EventAttempt,
    step_name: str,
    state: InternalStepWorkerState,
    now_seconds: float,
) -> list[WorkflowCommand]:
    """
    Small helper to assist in adding an event to a step worker state, or enqueuing it if it's not accepted.
    Note! This mutates the state, assuming that its already been deepcopied in an outer scope.
    """
    commands: list[WorkflowCommand] = []
    # Determine if there is available capacity based on in_progress workers
    has_space = len(state.in_progress) < state.config.num_workers
    if has_space:
        # Assign the smallest available worker id
        used = set(x.worker_id for x in state.in_progress)
        id_candidates = [i for i in range(state.config.num_workers) if i not in used]
        id = id_candidates[0]
        state_copy = state._deepcopy()
        shared_state: StepWorkerState = StepWorkerState(
            step_name=step_name,
            collected_events=state_copy.collected_events,
            collected_waiters=state_copy.collected_waiters,
        )
        state.in_progress.append(
            InProgressState(
                event=event.event,
                worker_id=id,
                shared_state=shared_state,
                attempts=event.attempts or 0,
                first_attempt_at=event.first_attempt_at or now_seconds,
            )
        )
        commands.append(CommandRunWorker(step_name=step_name, event=event.event, id=id))
        commands.append(
            CommandPublishEvent(
                StepStateChanged(
                    step_state=StepState.RUNNING,
                    name=step_name,
                    input_event_name=type(event.event).__name__,
                    worker_id=str(id),
                )
            )
        )
    else:
        commands.append(
            CommandPublishEvent(
                StepStateChanged(
                    step_state=StepState.PREPARING,
                    name=step_name,
                    input_event_name=type(event.event).__name__,
                    worker_id="<enqueued>",
                )
            )
        )
        state.queue.append(event)
    return commands


def _process_add_event_tick(
    tick: TickAddEvent, init: BrokerState, now_seconds: float
) -> tuple[BrokerState, list[WorkflowCommand]]:
    state = init.deepcopy()
    # iterate through the steps, and add to steps work queue if it's accepted.
    commands: list[WorkflowCommand] = []
    handled = False
    if isinstance(tick.event, StartEvent):
        state.is_running = True
    for step_name, step_config in state.config.steps.items():
        is_accepted = type(tick.event) in step_config.accepted_events
        if is_accepted and (tick.step_name is None or tick.step_name == step_name):
            handled = True
            subcommands = _add_or_enqueue_event(
                EventAttempt(
                    event=tick.event,
                    attempts=tick.attempts,
                    first_attempt_at=tick.first_attempt_at,
                ),
                step_name,
                state.workers[step_name],
                now_seconds,
            )
            commands.extend(subcommands)

    # separately, check if the event is a waiting event, and if so, update the waiting event state
    # and set the resolved event. Add the original event as a command
    for step_name, step_config in state.config.steps.items():
        wait_conditions = state.workers[step_name].collected_waiters
        for wait_condition in wait_conditions:
            is_match = type(tick.event) is wait_condition.waiting_for_event
            is_match = is_match and all(
                getattr(tick.event, k, None) == v
                for k, v in wait_condition.requirements.items()
            )
            if is_match:
                handled = True
                wait_condition.resolved_event = tick.event
                subcommands = _add_or_enqueue_event(
                    EventAttempt(event=wait_condition.event),
                    step_name,
                    state.workers[step_name],
                    now_seconds,
                )
                commands.extend(subcommands)
    if not handled:
        # InputRequiredEvent subclasses are intentionally designed to be handled
        # externally by human consumers, not by workflow steps. Don't emit
        # UnhandledEvent for these since they're working as intended.
        if not isinstance(tick.event, InputRequiredEvent):
            event_cls = type(tick.event)
            commands.append(
                CommandPublishEvent(
                    UnhandledEvent(
                        event_type=event_cls.__name__,
                        qualified_name=f"{event_cls.__module__}.{event_cls.__name__}",
                        step_name=tick.step_name,
                        idle=_check_idle_state(state),
                    )
                )
            )
    return state, commands


def _process_cancel_run_tick(
    tick: TickCancelRun, init: BrokerState
) -> tuple[BrokerState, list[WorkflowCommand]]:
    state = init.deepcopy()
    # Retain running state for resumption.
    return state, [
        CommandPublishEvent(event=WorkflowCancelledEvent()),
        CommandHalt(exception=WorkflowCancelledByUser()),
    ]


def _process_publish_event_tick(
    tick: TickPublishEvent, init: BrokerState
) -> tuple[BrokerState, list[WorkflowCommand]]:
    # doesn't affect state. Pass through as publish command
    return init, [CommandPublishEvent(event=tick.event)]


def _process_timeout_tick(
    tick: TickTimeout, init: BrokerState
) -> tuple[BrokerState, list[WorkflowCommand]]:
    state = init.deepcopy()
    state.is_running = False
    active_steps = [
        step_name
        for step_name, worker_state in init.workers.items()
        if len(worker_state.in_progress) > 0
    ]
    steps_info = (
        "Currently active steps: " + ", ".join(active_steps)
        if active_steps
        else "No steps active"
    )
    return state, [
        CommandPublishEvent(
            event=WorkflowTimedOutEvent(
                timeout=tick.timeout,
                active_steps=active_steps,
            )
        ),
        CommandHalt(
            exception=WorkflowTimeoutError(
                f"Operation timed out after {tick.timeout} seconds. {steps_info}"
            )
        ),
    ]
