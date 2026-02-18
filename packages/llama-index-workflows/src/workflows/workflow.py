# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Tuple,
)

from llama_index_instrumentation import get_dispatcher
from llama_index_instrumentation.dispatcher import active_instrument_tags
from llama_index_instrumentation.events.span import SpanDropEvent
from llama_index_instrumentation.span import active_span_id
from pydantic import ValidationError

if TYPE_CHECKING:  # pragma: no cover
    from .context import Context
    from .runtime.types.plugin import Runtime
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
from .resource import (
    ResourceDescriptor,
    ResourceManager,
    _ResourceConfig,
)
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

    _runtime: Runtime
    _workflow_name: str | None

    def __init__(
        self,
        timeout: float | None = 45.0,
        disable_validation: bool = False,
        verbose: bool = False,
        resource_manager: ResourceManager | None = None,
        num_concurrent_runs: int | None = None,
        runtime: Runtime | None = None,
        workflow_name: str | None = None,
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
            runtime (Runtime | None): Optional runtime to use for this workflow.
                If not provided, uses the current context-scoped runtime or
                falls back to basic_runtime.
            workflow_name (str | None): Optional explicit name for this workflow.
                If not provided, a module-qualified name is computed from
                the class's `__module__` and `__qualname__` attributes.
        """
        # Configuration
        self._timeout = timeout
        self._verbose = verbose
        self._disable_validation = disable_validation
        self._num_concurrent_runs = num_concurrent_runs
        # Store explicit name (None means use computed name)
        self._workflow_name = workflow_name
        # Detect StartEvent issues before StopEvent for clearer guidance
        self._start_event_class = self._ensure_start_event_class()
        self._stop_event_class = self._ensure_stop_event_class()
        self._events = self._ensure_events_collected()
        # Resource management
        self._resource_manager = resource_manager or ResourceManager()
        # Instrumentation
        self._dispatcher = dispatcher

        # Runtime registration: explicit > context-scoped > basic_runtime
        from workflows.plugins._context import get_current_runtime

        if runtime is not None:
            self._runtime = runtime
        else:
            # get_current_runtime() falls back to basic_runtime
            self._runtime = get_current_runtime()

        # Register with runtime for tracking (no-op for BasicRuntime)
        self._runtime.track_workflow(self)

    def _validate_valid_step_message(self, step: str, message: Event) -> None:
        """Validate that a step name exists in the workflow."""
        if step not in self._get_steps():
            raise WorkflowRuntimeError(f"Step {step} does not exist")

        step_func = self._get_steps()[step]
        step_config = step_func._step_config
        if type(message) not in step_config.accepted_events:
            raise WorkflowRuntimeError(
                f"Step {step} does not accept event of type {type(message)}"
            )

    @property
    def runtime(self) -> Runtime:
        """The runtime this workflow is registered with."""
        return self._runtime

    @property
    def workflow_name(self) -> str:
        """
        The workflow name.

        If an explicit name was provided at construction, returns that.
        Otherwise, returns a module-qualified name based on the class's
        __module__ and __qualname__ attributes.

        Examples:
            - Explicit: `Workflow(workflow_name="my-workflow")` → `"my-workflow"`
            - Module-level class: `"mymodule.MyWorkflow"`
            - Nested class: `"mymodule.Outer.Inner"`
            - Function-scoped: `"mymodule.func.<locals>.LocalWorkflow"`
        """
        if self._workflow_name is not None:
            return self._workflow_name
        cls = self.__class__
        return f"{cls.__module__}.{cls.__qualname__}"

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

        # Manually manage span to keep it open until workflow completes
        # llama-index-instrumentation currently does not manage Awaitable's well (i.e. the workflow handler)
        # this pattern is unusual enough to special case it here
        # First, generate span ID
        cls_name = self.__class__.__name__
        span_id = f"{cls_name}.run-{uuid.uuid4()}"

        # Get parent span ID for nesting
        parent_span_id = active_span_id.get()

        # Create bound args for span_enter/exit
        bound_args = inspect.signature(self.run).bind(
            ctx=ctx, start_event=start_event, **kwargs
        )

        # Set active span and notify span handlers
        span_token = active_span_id.set(span_id)

        dispatcher.span_enter(
            id_=span_id,
            bound_args=bound_args,
            instance=self,
            parent_id=parent_span_id,
            tags=active_instrument_tags.get(),
        )

        try:
            # Validate the workflow
            self._validate()

            # Extract run_id before passing remaining kwargs to start event
            run_id = kwargs.pop("run_id", None)

            # If a previous context is provided, pass its serialized form
            ctx = ctx if ctx is not None else Context(self)
            # TODO(v3) - remove dependency on is running for choosing whether to send a StartEvent.
            # Is not an easily synchronously queryable property.
            start_event_instance: StartEvent | None = (
                None
                if ctx.is_running
                else self._get_start_event_instance(start_event, **kwargs)
            )
            handler = ctx._workflow_run(
                workflow=self, start_event=start_event_instance, run_id=run_id
            )

            # Add callback to close span when workflow completes
            def _on_workflow_complete(task: asyncio.Task[Any]) -> None:
                try:
                    # Get result or exception
                    if task.cancelled():
                        result = None
                    else:
                        try:
                            result = task.result()
                        except Exception:
                            result = None

                    # Notify span exit
                    dispatcher.span_exit(
                        id_=span_id,
                        bound_args=bound_args,
                        instance=self,
                        result=result,
                    )
                finally:
                    # Reset span token
                    try:
                        active_span_id.reset(span_token)
                    except ValueError:
                        # Token might be from different context, ignore
                        pass

            # Attach callback to the handler's result task
            handler._result_task.add_done_callback(_on_workflow_complete)

            return handler
        except BaseException as e:
            # If run() fails, drop the span
            dispatcher.event(SpanDropEvent(span_id=span_id, err_str=str(e)))
            dispatcher.span_drop(
                id_=span_id, bound_args=bound_args, instance=self, err=e
            )
            active_span_id.reset(span_token)
            raise

    def _validate_resource_configs(self) -> list[str]:
        """Validate all resource configs (including nested ones) by loading them."""
        errors: list[str] = []
        seen: set[str] = set()

        # Stack-based traversal of all resources and their dependencies
        stack: list[_ResourceValidationContext] = []
        for step_func in self._get_steps().values():
            step_name = step_func.__name__
            for res_def in step_func._step_config.resources:
                res_def.resource.set_type_annotation(res_def.type_annotation)
                stack.append(
                    _ResourceValidationContext(
                        resource=res_def.resource,
                        step_name=step_name,
                        param_name=res_def.name,
                        resource_chain=[res_def.resource.name],
                    )
                )

        while stack:
            ctx = stack.pop()
            if ctx.resource.name in seen:
                continue
            seen.add(ctx.resource.name)

            # Add dependencies to stack
            for _dep_param, dep, type_ann in ctx.resource.get_dependencies():
                dep.set_type_annotation(type_ann)
                stack.append(ctx.with_dependency(dep))

            # Validate if it's a config
            if isinstance(ctx.resource, _ResourceConfig):
                try:
                    ctx.resource.call()
                except Exception as e:
                    errors.append(f"In {ctx.format_location()}: {e}")

        return errors

    async def _validate_resources(self) -> list[str]:
        """Validate all resources by resolving them (catches circular deps)."""
        errors: list[str] = []
        for step_func in self._get_steps().values():
            step_name = step_func.__name__
            for res_def in step_func._step_config.resources:
                res_def.resource.set_type_annotation(res_def.type_annotation)
                try:
                    await self._resource_manager.get(res_def.resource)
                except Exception as e:
                    location = f"step '{step_name}', parameter '{res_def.name}'"
                    errors.append(f"In {location}: {e}")
        return errors

    def validate(
        self,
        *,
        validate_resource_configs: bool = True,
        validate_resources: bool = False,
    ) -> bool:
        """
        Validate the workflow to ensure it's well-formed.

        This method validates the event graph and optionally validates resources:
        - Resource configs (JSON files with Pydantic validation) are validated by default
        - Resource factories are not validated by default (may require env vars)
        - Circular resource dependencies are caught when validate_resources=True

        Args:
            validate_resource_configs: If True (default), validate that resource
                config files exist and contain valid data for their Pydantic models.
            validate_resources: If False (default), skip resolving resource factories
                during validation. Set to True to also validate that resource
                factories can be resolved and detect circular dependencies
                (may require environment variables or external connections).

        Returns:
            True if the workflow uses human-in-the-loop, False otherwise.
        """
        return self._validate(
            validate_resource_configs=validate_resource_configs,
            validate_resources=validate_resources,
            force=True,  # Explicit validate() call should always run
        )

    def _validate(
        self,
        *,
        validate_resource_configs: bool = True,
        validate_resources: bool = False,
        force: bool = False,
    ) -> bool:
        if self._disable_validation and not force:
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

        # Resource validation
        if validate_resource_configs:
            if errors := self._validate_resource_configs():
                raise WorkflowValidationError(
                    "Resource config validation failed:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )

        if validate_resources:
            # Python 3.9 compat: asyncio.run() closes the loop, must restore it
            errors = asyncio.run(self._validate_resources())
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            if errors:
                raise WorkflowValidationError(
                    "Resource validation failed:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )

        # Check if the workflow uses human-in-the-loop
        return (
            InputRequiredEvent in produced_events
            or HumanResponseEvent in consumed_events
        )


@dataclass
class _ResourceValidationContext:
    """Tracks context for resource validation to provide clear error messages."""

    resource: ResourceDescriptor
    step_name: str
    param_name: str
    resource_chain: list[str] = field(default_factory=list)

    def format_location(self) -> str:
        """Format the location string for error messages."""
        if len(self.resource_chain) > 1:
            chain_str = " -> ".join(self.resource_chain)
            return (
                f"step '{self.step_name}', parameter '{self.param_name}' ({chain_str})"
            )
        return f"step '{self.step_name}', parameter '{self.param_name}'"

    def with_dependency(self, dep: ResourceDescriptor) -> _ResourceValidationContext:
        """Create a new context for a dependency resource."""
        return _ResourceValidationContext(
            resource=dep,
            step_name=self.step_name,
            param_name=self.param_name,
            resource_chain=[*self.resource_chain, dep.name],
        )
