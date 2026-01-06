# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Tuple,
)

from llama_index_instrumentation import get_dispatcher
from pydantic import ValidationError

if TYPE_CHECKING:  # pragma: no cover
    from .context import Context
from .decorators import StepConfig, StepFunction
from .errors import (
    WorkflowConfigurationError,
    WorkflowRuntimeError,
    WorkflowValidationError,
)
from .events import (
    Event,
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
        cls._step_functions: dict[str, StepFunction] = {}


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

    # Populated by the metaclass; declared here for type checkers.
    _step_functions: dict[str, StepFunction]

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
        # Detect StartEvent issues before StopEvent for clearer guidance
        self._start_event_class = self._ensure_start_event_class()
        self._stop_event_class = self._ensure_stop_event_class()
        self._events = self._ensure_events_collected()
        self._sem = (
            asyncio.Semaphore(num_concurrent_runs) if num_concurrent_runs else None
        )
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
            step_config: StepConfig = step_func._step_config
            for event_type in step_config.accepted_events:
                if issubclass(event_type, StartEvent):
                    start_events_found.add(event_type)

        num_found = len(start_events_found)
        if num_found == 0:
            cls_name = self.__class__.__name__
            msg = (
                "At least one Event of type StartEvent must be received by any step. "
                f"(Workflow '{cls_name}' has no @step that accepts StartEvent.)"
            )
            raise WorkflowConfigurationError(msg)
        elif num_found > 1:
            cls_name = self.__class__.__name__
            msg = (
                f"Only one type of StartEvent is allowed per workflow, found {num_found}: {start_events_found} "
                f"in workflow '{cls_name}'."
            )
            raise WorkflowConfigurationError(msg)
        else:
            return start_events_found.pop()

    @property
    def start_event_class(self) -> type[StartEvent]:
        """The `StartEvent` subclass accepted by this workflow.

        Determined by inspecting step input types.
        """
        return self._start_event_class

    @property
    def events(self) -> list[type[Event]]:
        """Returns all known events emitted by this workflow.

        Determined by inspecting step input/output types.
        """
        return self._events

    def _ensure_events_collected(self) -> list[type[Event]]:
        """Returns all known events emitted by this workflow.

        Determined by inspecting step input/output types.
        """
        events_found: set[type[Event]] = set()
        for step_func in self._get_steps().values():
            step_config: StepConfig = step_func._step_config

            # Do not collect events from the done step
            if step_func.__name__ == "_done":
                continue

            for event_type in step_config.return_types:
                if issubclass(event_type, Event):
                    events_found.add(event_type)
            for event_type in step_config.accepted_events:
                if issubclass(event_type, Event):
                    events_found.add(event_type)

        return list(events_found)

    def _ensure_stop_event_class(self) -> type[RunResultT]:
        """
        Returns the StopEvent type used in this workflow.

        It works by inspecting the events returned.
        """
        stop_events_found: set[type[StopEvent]] = set()
        for step_func in self._get_steps().values():
            step_config: StepConfig = step_func._step_config
            for event_type in step_config.return_types:
                if issubclass(event_type, StopEvent):
                    stop_events_found.add(event_type)

        num_found = len(stop_events_found)
        if num_found == 0:
            cls_name = self.__class__.__name__
            msg = (
                "At least one Event of type StopEvent must be returned by any step. "
                f"(Workflow '{cls_name}' has no @step that returns StopEvent.)"
            )
            raise WorkflowConfigurationError(msg)
        elif num_found > 1:
            cls_name = self.__class__.__name__
            msg = (
                f"Only one type of StopEvent is allowed per workflow, found {num_found}: {stop_events_found} "
                f"in workflow '{cls_name}'."
            )
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
    def add_step(cls, func: StepFunction) -> None:
        """
        Adds a free function as step for this workflow instance.

        It raises an exception if a step with the same name was already added to the workflow.
        """
        step_config: StepConfig | None = getattr(func, "_step_config", None)
        if not step_config:
            msg = f"Step function {func.__name__} is missing the `@step` decorator."
            raise WorkflowValidationError(msg)

        if func.__name__ in {**get_steps_from_class(cls), **cls._step_functions}:
            msg = f"A step {func.__name__} is already part of this workflow, please choose another name."
            raise WorkflowValidationError(msg)

        cls._step_functions[func.__name__] = func

    def _get_steps(self) -> dict[str, StepFunction]:
        """Returns all the steps, whether defined as methods or free functions."""
        return {**get_steps_from_instance(self), **self.__class__._step_functions}

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
        from workflows.context import Context

        # Validate the workflow
        self._validate()

        # If a previous context is provided, pass its serialized form
        ctx = ctx if ctx is not None else Context(self)
        start_event_instance: StartEvent | None = (
            None
            if ctx.is_running
            else self._get_start_event_instance(start_event, **kwargs)
        )
        return ctx._workflow_run(
            workflow=self, start_event=start_event_instance, semaphore=self._sem
        )

    def validate(self) -> bool:
        """
        Validate the workflow to ensure it's well-formed.

        Returns True if the workflow uses human-in-the-loop, False otherwise.
        """
        return self._validate()

    def _validate(self) -> bool:
        if self._disable_validation:
            return False

        # Ensure at least one step is configured before inspecting events
        if not self._get_steps():
            cls_name = self.__class__.__name__
            msg = (
                f"Workflow '{cls_name}' has no configured steps. "
                "Did you forget to annotate methods with @step or to register "
                "free-function steps via @step(workflow=...)?"
            )
            raise WorkflowConfigurationError(msg)

        # Recompute StartEvent and StopEvent classes here to support dynamic changes
        # and to surface StartEvent errors before StopEvent during validation.
        self._start_event_class = self._ensure_start_event_class()
        self._stop_event_class = self._ensure_stop_event_class()

        produced_events: set[type] = {self._start_event_class}
        consumed_events: set[type] = set()

        # Collect steps that incorrectly accept StopEvent
        steps_accepting_stop_event: list[str] = []

        for name, step_func in self._get_steps().items():
            step_config: StepConfig = step_func._step_config

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
