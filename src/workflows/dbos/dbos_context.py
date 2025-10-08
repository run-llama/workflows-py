from __future__ import annotations

import asyncio
import functools
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
)
from typing_extensions import override

from workflows.context.context import Context
from dbos import DBOS, SetWorkflowID, WorkflowHandle

from workflows.context.serializers import JsonSerializer
from workflows.context.state_store import MODEL_T, DictState
from workflows.decorators import StepConfig
from workflows.errors import WorkflowCancelledByUser, WorkflowDone, WorkflowRuntimeError
from workflows.events import (
    Event,
    InputRequiredEvent,
    StepState,
    StepStateChanged,
    EventsQueueChanged,
)
from workflows.resource import ResourceManager

if TYPE_CHECKING:  # pragma: no cover
    from workflows import Workflow


@DBOS.step()
async def _durable_time() -> float:
    return time.time()


class DBOSContext(Context[MODEL_T]):
    """
    DBOS Context class for durable workflows.
    It requires a unique context name to identify the DBOS workflows.
    """

    def __init__(self, workflow: "Workflow", context_name: str) -> None:
        super().__init__(workflow)

        self.context_name = context_name
        self.workflow = workflow

        self._dbos_wf_handle: set[WorkflowHandle] = set()
        self._step_wf_handle_map: dict[str, set[WorkflowHandle]] = {}
        # Register wrapped workflows for sync and async step workers
        step_names = self.workflow._get_steps().keys()
        self.dbos_wrapped_step_workers: dict[str, Callable] = {}

        for step_name in step_names:

            @DBOS.workflow(name=f"{self.context_name}.{step_name}.worker")
            async def wrapped_step_worker(
                name: str,
                config: StepConfig,
                verbose: bool,
                run_id: str,
                worker_id: str,
                resource_manager: ResourceManager,
            ) -> None:
                step: Callable = self.workflow._get_steps()[name]
                print(f"DBOS executing step worker for step: {name}")
                await self._internal_step_worker(
                    name,
                    step,
                    config,
                    verbose,
                    run_id,
                    worker_id,
                    resource_manager,
                )

            self.dbos_wrapped_step_workers[step_name] = wrapped_step_worker

        # Register wrapped workflow for cancel worker
        @DBOS.step()
        async def _wrapped_cancel_worker() -> None:
            try:
                if self._cancel_flag.is_set():
                    raise WorkflowCancelledByUser
            except asyncio.CancelledError:
                return

        @DBOS.workflow(name=f"{self.context_name}._cancel.worker")
        async def wrapped_cancel_worker() -> None:
            print("DBOS executing cancel worker")
            # Non-blocking wait for the cancel flag to be set
            while True:
                await _wrapped_cancel_worker()
                recv_res = await DBOS.recv_async(timeout_seconds=3)
                if recv_res == "done":
                    # Meaning the end of the workflow.
                    print("Cancel worker received termination signal")
                    return

        self.dbos_wrapped_cancel_worker = wrapped_cancel_worker

    @override
    def add_step_worker(
        self,
        name: str,
        step: Callable,
        config: StepConfig,
        verbose: bool,
        run_id: str,
        worker_id: str,
        resource_manager: ResourceManager,
    ) -> None:
        """Spawn a DBOS workflow to process events for a step.

        Args:
            name (str): Step name.
            step (Callable): Step function (sync or async).
            config (StepConfig): Resolved configuration for the step.
            verbose (bool): If True, print step activity.
            run_id (str): Run identifier for instrumentation.
            worker_id (str): ID of the worker running the step
            resource_manager (ResourceManager): Resource injector for the step.
        """
        print(f"Starting worker for step {name} with worker ID {worker_id}")
        with SetWorkflowID(worker_id):
            wf_func = self.dbos_wrapped_step_workers[name]
            wf_handle = DBOS.start_workflow(
                wf_func, name, config, verbose, run_id, worker_id, resource_manager
            )
            self._dbos_wf_handle.add(wf_handle)
            self._step_wf_handle_map.setdefault(name, set()).add(wf_handle)

    async def _internal_step_worker(
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
            # Don't block forever, set recv timeout. So we're able to cancel the DBOS workflow
            ev: Any = None
            recv_res = await DBOS.recv_async(timeout_seconds=3)
            # Try to receive cancellation signal
            if recv_res == "done":
                # Meaning the end of the workflow.
                print(f"Step worker {name} received termination signal")
                return
            else:
                ev = recv_res
                if type(ev) not in config.accepted_events:
                    continue

            if verbose and name != "_done":
                print(f"Running step {name}")

            # run step
            # Initialize state manager if needed
            if self._state_store is None:
                if (
                    hasattr(config, "context_state_type")
                    and config.context_state_type is not None
                ):
                    # Instantiate the state class and initialize the state manager
                    try:
                        # Try to instantiate the state class
                        state_instance = cast(MODEL_T, config.context_state_type())
                        await self._init_state_store(state_instance)
                    except Exception as e:
                        raise WorkflowRuntimeError(
                            f"Failed to initialize state of type {config.context_state_type}. "
                            "Does your state define defaults for all fields? Original error:\n"
                            f"{e}"
                        ) from e
                else:
                    # Initialize state manager with DictState by default
                    dict_state = cast(MODEL_T, DictState())
                    await self._init_state_store(dict_state)

            kwargs: dict[str, Any] = {}
            if config.context_parameter:
                kwargs[config.context_parameter] = self
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
                    context_state=self.store.to_dict_snapshot(JsonSerializer()),
                )
            )
            if asyncio.iscoroutinefunction(step):
                retry_start_at = await _durable_time()
                attempts = 0
                while True:
                    await self.mark_in_progress(name=name, ev=ev, worker_id=worker_id)
                    await self.add_running_step(name)
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
                        await self.remove_from_in_progress(
                            name=name, ev=ev, worker_id=worker_id
                        )
                        raise
                    except Exception as e:
                        if config.retry_policy is None:
                            raise

                        curr_time = await _durable_time()
                        delay = config.retry_policy.next(
                            retry_start_at + curr_time, attempts, e
                        )
                        if delay is None:
                            raise

                        attempts += 1
                        if verbose:
                            print(
                                f"Step {name} produced an error, retry in {delay} seconds"
                            )
                        await DBOS.sleep_async(delay)
                    finally:
                        await self.remove_running_step(name)
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
                    await self.remove_from_in_progress(
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
                    context_state=self.store.to_dict_snapshot(JsonSerializer()),
                )
            )
            if verbose and name != "_done":
                if new_ev is not None:
                    print(f"Step {name} produced event {type(new_ev).__name__}")
                else:
                    print(f"Step {name} produced no event")

            # Store the accepted event for the drawing operations
            if new_ev is not None:
                self._accepted_events.append((name, type(ev).__name__))

            # Fail if the step returned something that's not an event
            if new_ev is not None and not isinstance(new_ev, Event):
                msg = f"Step function {name} returned {type(new_ev).__name__} instead of an Event instance."
                raise WorkflowRuntimeError(msg)

            await self.remove_from_in_progress(name=name, ev=ev, worker_id=worker_id)
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
                await self.send_event_async(new_ev)

    @override
    def add_cancel_worker(self) -> None:
        """Install a worker that turns a cancel flag into an exception.

        When the cancel flag is set, a `WorkflowCancelledByUser` will be raised
        internally to abort the run.

        See Also:
            - [WorkflowCancelledByUser][workflows.errors.WorkflowCancelledByUser]
        """
        print("Starting DBOS cancel worker")
        wf_handle = DBOS.start_workflow(self.dbos_wrapped_cancel_worker)
        self._dbos_wf_handle.add(wf_handle)
        self._step_wf_handle_map.setdefault("_cancel", set()).add(wf_handle)

    @override
    async def shutdown(self) -> None:
        """Shut down the workflow run and clean up background tasks.

        Cancels all outstanding workers, waits for them to finish, and marks the
        context as not running. Queues and state remain available so callers can
        inspect or drain leftover events.
        """
        print("Shutting down DBOSContext")
        self.is_running = False
        self._dbos_wf_handle.clear()

    @override
    def send_event(self, message: Event, step: str | None = None) -> None:
        """Dispatch an event to one or all workflow steps.

        If `step` is omitted, the event is broadcast to all step queues and
        non-matching steps will ignore it. When `step` is provided, the target
        step must accept the event type or a
        [WorkflowRuntimeError][workflows.errors.WorkflowRuntimeError] is raised.

        This method uses DBOS durable notification to send events.
        """
        print("DBOSContext sending event")
        if step is None:
            # Send through DBOS
            for name, wf_handles in self._step_wf_handle_map.items():
                for wf_handle in wf_handles:
                    DBOS.send(wf_handle.get_workflow_id(), message)
                    # TODO (Qian): keep a counter of the number of events in the queue?
                    #  For now, just write size=1
                    self.write_event_to_stream(EventsQueueChanged(name=name, size=1))

        else:
            if step not in self._step_configs:
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            step_config = self._step_configs[step]
            if step_config and type(message) in step_config.accepted_events:
                wf_handles = self._step_wf_handle_map.get(step, set())
                for wf_handle in wf_handles:
                    DBOS.send(wf_handle.get_workflow_id(), message)
                # TODO (Qian): keep a counter of the number of events in the queue?
                self.write_event_to_stream(EventsQueueChanged(name=step, size=1))
            else:
                raise WorkflowRuntimeError(
                    f"Step {step} does not accept event of type {type(message)}"
                )

        self._broker_log.append(message)

    async def send_event_async(
        self, message: Event | str, step: str | None = None
    ) -> None:
        """Dispatch an event to one or all workflow steps.

        If `step` is omitted, the event is broadcast to all step queues and
        non-matching steps will ignore it. When `step` is provided, the target
        step must accept the event type or a
        [WorkflowRuntimeError][workflows.errors.WorkflowRuntimeError] is raised.

        This method uses DBOS durable notification to send events.
        """
        print(f"DBOSContext sending event async: {message}")
        if step is None:
            # Send through DBOS
            for name, wf_handles in self._step_wf_handle_map.items():
                for wf_handle in wf_handles:
                    await DBOS.send_async(wf_handle.get_workflow_id(), message)
                    # TODO (Qian): keep a counter of the number of events in the queue?
                    #  For now, just write size=1
                    self.write_event_to_stream(EventsQueueChanged(name=name, size=1))

        else:
            if step not in self._step_configs:
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            step_config = self._step_configs[step]
            if step_config and type(message) in step_config.accepted_events:
                wf_handles = self._step_wf_handle_map.get(step, set())
                for wf_handle in wf_handles:
                    await DBOS.send_async(wf_handle.get_workflow_id(), message)
                # TODO (Qian): keep a counter of the number of events in the queue?
                self.write_event_to_stream(EventsQueueChanged(name=step, size=1))
            else:
                raise WorkflowRuntimeError(
                    f"Step {step} does not accept event of type {type(message)}"
                )

        if isinstance(message, Event):
            self._broker_log.append(message)
