# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import (
    Any,
    Callable,
    Tuple,
)
from weakref import WeakSet

from llama_index_instrumentation import get_dispatcher
from pydantic import ValidationError

from .context import Context
from .decorators import StepConfig, step
from .errors import (
    WorkflowConfigurationError,
    WorkflowDone,
    WorkflowRuntimeError,
    WorkflowTimeoutError,
    WorkflowValidationError,
)
from .events import (
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from .handler import WorkflowHandler
from .resource import ResourceManager
from .types import RunResultT
from .utils import get_steps_from_class, get_steps_from_instance

dispatcher = get_dispatcher(__name__)
logger = logging.getLogger()


class WorkflowMeta(type):
    def __init__(cls, name: str, bases: Tuple[type, ...], dct: dict[str, Any]) -> None:
        super().__init__(name, bases, dct)
        cls._step_functions: dict[str, Callable] = {}


class Workflow(metaclass=WorkflowMeta):
    """
    Event-driven orchestrator to define and run application flows using typed steps.

    A `Workflow` is composed of `@step`-decorated callables that accept and emit
    typed [Event][workflows.events.Event]s. Steps can be declared as instance
    methods or as free functions registered via the decorator.

    Key features:
    - Validation of step signatures and event graph before running
    - Typed start/stop events
    - Streaming of intermediate events
    - Optional human-in-the-loop events
    - Retry policies per step
    - Resource injection

    Examples:
        Basic usage:

        ```python
        from workflows import Workflow, step
        from workflows.events import StartEvent, StopEvent

        class MyFlow(Workflow):
            @step
            async def start(self, ev: StartEvent) -> StopEvent:
                return StopEvent(result="done")

        result = await MyFlow(timeout=60).run(topic="Pirates")
        ```

        Custom start/stop events and streaming:

        ```python
        handler = MyFlow().run()
        async for ev in handler.stream_events():
            ...
        result = await handler
        ```

    See Also:
        - [step][workflows.decorators.step]
        - [Event][workflows.events.Event]
        - [Context][workflows.context.context.Context]
        - [WorkflowHandler][workflows.handler.WorkflowHandler]
        - [RetryPolicy][workflows.retry_policy.RetryPolicy]
    """

    def __init__(
        self,
        timeout: float | None = 45.0,
        disable_validation: bool = False,
        verbose: bool = False,
        resource_manager: ResourceManager | None = None,
        num_concurrent_runs: int | None = None,
    ) -> None:
        """
        Initialize a workflow instance.

        Args:
            timeout (float | None): Max seconds to wait for completion. `None`
                disables the timeout.
            disable_validation (bool): Skip pre-run validation of the event graph
                (not recommended).
            verbose (bool): If True, print step activity.
            resource_manager (ResourceManager | None): Custom resource manager
                for dependency injection.
            num_concurrent_runs (int | None): Limit on concurrent `run()` calls.
        """
        # Configuration
        self._timeout = timeout
        self._verbose = verbose
        self._disable_validation = disable_validation
        self._num_concurrent_runs = num_concurrent_runs
        self._stop_event_class = self._ensure_stop_event_class()
        self._start_event_class = self._ensure_start_event_class()
        self._sem = (
            asyncio.Semaphore(num_concurrent_runs) if num_concurrent_runs else None
        )
        # Broker machinery
        self._contexts: WeakSet[Context] = WeakSet()
        # Resource management
        self._resource_manager = resource_manager or ResourceManager()
        # Instrumentation
        self._dispatcher = dispatcher

    def _ensure_start_event_class(self) -> type[StartEvent]:
        """
        Returns the StartEvent type used in this workflow.

        It works by inspecting the events received by the step methods.
        """
        start_events_found: set[type[StartEvent]] = set()
        for step_func in self._get_steps().values():
            step_config: StepConfig = getattr(step_func, "__step_config")
            for event_type in step_config.accepted_events:
                if issubclass(event_type, StartEvent):
                    start_events_found.add(event_type)

        num_found = len(start_events_found)
        if num_found == 0:
            msg = "At least one Event of type StartEvent must be received by any step."
            raise WorkflowConfigurationError(msg)
        elif num_found > 1:
            msg = f"Only one type of StartEvent is allowed per workflow, found {num_found}: {start_events_found}."
            raise WorkflowConfigurationError(msg)
        else:
            return start_events_found.pop()

    @property
    def start_event_class(self) -> type[StartEvent]:
        """The `StartEvent` subclass accepted by this workflow.

        Determined by inspecting step input types.
        """
        return self._start_event_class

    def _ensure_stop_event_class(self) -> type[RunResultT]:
        """
        Returns the StopEvent type used in this workflow.

        It works by inspecting the events returned.
        """
        stop_events_found: set[type[StopEvent]] = set()
        for step_func in self._get_steps().values():
            step_config: StepConfig = getattr(step_func, "__step_config")
            for event_type in step_config.return_types:
                if issubclass(event_type, StopEvent):
                    stop_events_found.add(event_type)

        num_found = len(stop_events_found)
        if num_found == 0:
            msg = "At least one Event of type StopEvent must be returned by any step."
            raise WorkflowConfigurationError(msg)
        elif num_found > 1:
            msg = f"Only one type of StopEvent is allowed per workflow, found {num_found}: {stop_events_found}."
            raise WorkflowConfigurationError(msg)
        else:
            return stop_events_found.pop()

    @property
    def stop_event_class(self) -> type[RunResultT]:
        """The `StopEvent` subclass produced by this workflow.

        Determined by inspecting step return annotations.
        """
        return self._stop_event_class

    @classmethod
    def add_step(cls, func: Callable) -> None:
        """
        Adds a free function as step for this workflow instance.

        It raises an exception if a step with the same name was already added to the workflow.
        """
        step_config: StepConfig | None = getattr(func, "__step_config", None)
        if not step_config:
            msg = f"Step function {func.__name__} is missing the `@step` decorator."
            raise WorkflowValidationError(msg)

        if func.__name__ in {**get_steps_from_class(cls), **cls._step_functions}:
            msg = f"A step {func.__name__} is already part of this workflow, please choose another name."
            raise WorkflowValidationError(msg)

        cls._step_functions[func.__name__] = func

    def _get_steps(self) -> dict[str, Callable]:
        """Returns all the steps, whether defined as methods or free functions."""
        return {**get_steps_from_instance(self), **self._step_functions}  # type: ignore[attr-defined]

    def _start(
        self,
        ctx: Context | None = None,
    ) -> Tuple[Context, str]:
        """
        sets up the queues and tasks for each declared step.

        This method also launches each step as an async task.
        """
        run_id = str(uuid.uuid4())
        if ctx is None:
            ctx = Context(self)
            self._contexts.add(ctx)
        else:
            # clean up the context from the previous run
            ctx._tasks = set()
            ctx._retval = None
            ctx._step_events_holding = None
            ctx._cancel_flag.clear()

        for name, step_func in self._get_steps().items():
            if name not in ctx._queues:
                ctx._queues[name] = asyncio.Queue()

            if name not in ctx._step_flags:
                ctx._step_flags[name] = asyncio.Event()

            # At this point, step_func is guaranteed to have the `__step_config` attribute
            step_config: StepConfig = getattr(step_func, "__step_config")

            # Make the system step "_done" accept custom stop events
            if (
                name == "_done"
                and self._stop_event_class not in step_config.accepted_events
            ):
                step_config.accepted_events.append(self._stop_event_class)

            for _ in range(step_config.num_workers):
                ctx.add_step_worker(
                    name=name,
                    step=step_func,
                    config=step_config,
                    verbose=self._verbose,
                    run_id=run_id,
                    worker_id=str(uuid.uuid4()),
                    resource_manager=self._resource_manager,
                )

        # add dedicated cancel task
        ctx.add_cancel_worker()

        return ctx, run_id

    def _get_start_event_instance(
        self, start_event: StartEvent | None, **kwargs: Any
    ) -> StartEvent:
        if start_event is not None:
            # start_event was used wrong
            if not isinstance(start_event, StartEvent):
                msg = "The 'start_event' argument must be an instance of 'StartEvent'."
                raise ValueError(msg)

            # start_event is ok but point out that additional kwargs will be ignored in this case
            if kwargs:
                msg = (
                    "Keyword arguments are not supported when 'run()' is invoked with the 'start_event' parameter."
                    f" These keyword arguments will be ignored: {kwargs}"
                )
                logger.warning(msg)
            return start_event

        # Old style start event creation, with kwargs used to create an instance of self._start_event_class
        try:
            return self._start_event_class(**kwargs)
        except ValidationError as e:
            ev_name = self._start_event_class.__name__
            msg = f"Failed creating a start event of type '{ev_name}' with the keyword arguments: {kwargs}"
            logger.debug(e)
            raise WorkflowRuntimeError(msg)

    @dispatcher.span
    def run(
        self,
        ctx: Context | None = None,
        start_event: StartEvent | None = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        """Run the workflow and return a handler for results and streaming.

        This schedules the workflow execution in the background and returns a
        [WorkflowHandler][workflows.handler.WorkflowHandler] that can be awaited
        for the final result or used to stream intermediate events.

        You may pass either a concrete `start_event` instance or keyword
        arguments that will be used to construct the inferred
        [StartEvent][workflows.events.StartEvent] subclass.

        Args:
            ctx (Context | None): Optional context to resume or share state
                across runs. If omitted, a fresh context is created.
            start_event (StartEvent | None): Optional explicit start event.
            **kwargs (Any): Keyword args to initialize the start event when
                `start_event` is not provided.

        Returns:
            WorkflowHandler: A future-like object to await the final result and
            stream events.

        Raises:
            WorkflowValidationError: If validation fails and validation is
                enabled.
            WorkflowRuntimeError: If the start event cannot be created from kwargs.
            WorkflowTimeoutError: If execution exceeds the configured timeout.

        Examples:
            ```python
            # Create and run with kwargs
            handler = MyFlow().run(topic="Pirates")

            # Stream events
            async for ev in handler.stream_events():
                ...

            # Await final result
            result = await handler
            ```

            If you subclassed the start event, you can also directly pass it in:

            ```python
            result = await my_workflow.run(start_event=MyStartEvent(topic="Pirates"))
            ```
        """

        # Validate the workflow
        self._validate()

        # Start the machinery in a new Context or use the provided one
        ctx, run_id = self._start(ctx=ctx)

        result = WorkflowHandler(ctx=ctx, run_id=run_id)

        async def _run_workflow() -> None:
            if self._sem:
                await self._sem.acquire()
            try:
                if not ctx.is_running:
                    # Send the first event
                    start_event_instance = self._get_start_event_instance(
                        start_event, **kwargs
                    )
                    ctx.send_event(start_event_instance)

                    # the context is now running
                    ctx.is_running = True

                done, unfinished = await asyncio.wait(
                    ctx._tasks,
                    timeout=self._timeout,
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
                ctx.is_running = False

                if exception_raised:
                    # cancel the stream
                    ctx.write_event_to_stream(StopEvent())

                    raise exception_raised

                if not we_done:
                    # cancel the stream
                    ctx.write_event_to_stream(StopEvent())

                    msg = f"Operation timed out after {self._timeout} seconds"
                    raise WorkflowTimeoutError(msg)

                result.set_result(ctx._retval)
            except Exception as e:
                result.set_exception(e)
            finally:
                if self._sem:
                    self._sem.release()
                await ctx.shutdown()

        asyncio.create_task(_run_workflow())
        return result

    @step(num_workers=1)
    async def _done(self, ctx: Context, ev: StopEvent) -> None:
        """Tears down the whole workflow and stop execution."""
        if self._stop_event_class is StopEvent:
            ctx._retval = ev.result
        else:
            ctx._retval = ev

        ctx.write_event_to_stream(ev)

        # Signal we want to stop the workflow. Since we're about to raise, delete
        # the reference to ctx explicitly to avoid it becoming dangling
        del ctx
        raise WorkflowDone

    def _validate(self) -> bool:
        """
        Validate the workflow to ensure it's well-formed.

        Returns True if the workflow uses human-in-the-loop, False otherwise.
        """
        if self._disable_validation:
            return False

        produced_events: set[type] = {self._start_event_class}
        consumed_events: set[type] = set()

        # Collect steps that incorrectly accept StopEvent
        steps_accepting_stop_event: list[str] = []

        for name, step_func in self._get_steps().items():
            step_config: StepConfig | None = getattr(step_func, "__step_config")
            # At this point we know step config is not None, let's make the checker happy
            assert step_config is not None

            # Check that no user-defined step accepts StopEvent (only _done step should)
            if name != "_done":
                for event_type in step_config.accepted_events:
                    if issubclass(event_type, StopEvent):
                        steps_accepting_stop_event.append(name)
                        break

            for event_type in step_config.accepted_events:
                consumed_events.add(event_type)

            for event_type in step_config.return_types:
                if event_type is type(None):
                    # some events may not trigger other events
                    continue

                produced_events.add(event_type)

        # Raise error if any steps incorrectly accept StopEvent
        if steps_accepting_stop_event:
            step_names = "', '".join(steps_accepting_stop_event)
            plural = "" if len(steps_accepting_stop_event) == 1 else "s"
            msg = f"Step{plural} '{step_names}' cannot accept StopEvent. StopEvent signals the end of the workflow. Use a different Event type instead."
            raise WorkflowValidationError(msg)

        # Check if no StopEvent is produced
        stop_ok = False
        for ev in produced_events:
            if issubclass(ev, StopEvent):
                stop_ok = True
                break
        if not stop_ok:
            msg = "No event of type StopEvent is produced."
            raise WorkflowValidationError(msg)

        # Check if all consumed events are produced (except specific built-in events)
        unconsumed_events = consumed_events - produced_events
        unconsumed_events = {
            x
            for x in unconsumed_events
            if not issubclass(x, (InputRequiredEvent, HumanResponseEvent, StopEvent))
        }
        if unconsumed_events:
            names = ", ".join(ev.__name__ for ev in unconsumed_events)
            raise WorkflowValidationError(
                f"The following events are consumed but never produced: {names}"
            )

        # Check if there are any unused produced events (except specific built-in events)
        unused_events = produced_events - consumed_events
        unused_events = {
            x
            for x in unused_events
            if not issubclass(
                x, (InputRequiredEvent, HumanResponseEvent, self._stop_event_class)
            )
        }
        if unused_events:
            names = ", ".join(ev.__name__ for ev in unused_events)
            raise WorkflowValidationError(
                f"The following events are produced but never consumed: {names}"
            )

        # Check if the workflow uses human-in-the-loop
        return (
            InputRequiredEvent in produced_events
            or HumanResponseEvent in consumed_events
        )
