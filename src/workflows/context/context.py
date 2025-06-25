# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import functools
import json
import time
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Generic,
    Tuple,
    Type,
    TypeVar,
)

from workflows.decorators import StepConfig
from workflows.errors import (
    ContextSerdeError,
    WorkflowCancelledByUser,
    WorkflowDone,
    WorkflowRuntimeError,
)
from workflows.events import Event, InputRequiredEvent
from workflows.resource import ResourceManager
from workflows.service import ServiceManager
from workflows.types import RunResultT

from .serializers import BaseSerializer, JsonSerializer
from .state_store import InMemoryStateStore, MODEL_T, DictState

if TYPE_CHECKING:  # pragma: no cover
    from workflows import Workflow
    from workflows.checkpointer import CheckpointCallback

T = TypeVar("T", bound=Event)
EventBuffer = dict[str, list[Event]]


# Only warn once about unserializable keys
class UnserializableKeyWarning(Warning):
    pass


warnings.simplefilter("once", UnserializableKeyWarning)


class Context(Generic[MODEL_T]):
    """
    A global object representing a context for a given workflow run.

    The Context object can be used to store data that needs to be available across iterations during a workflow
    execution, and across multiple workflow runs.
    Every context instance offers two type of data storage: a global one, that's shared among all the steps within a
    workflow, and private one, that's only accessible from a single step.

    Both `set` and `get` operations on global data are governed by a lock, and considered coroutine-safe.
    """

    # These keys are set by pre-built workflows and
    # are known to be unserializable in some cases.
    known_unserializable_keys = ("memory",)

    def __init__(
        self,
        workflow: "Workflow",
        stepwise: bool = False,
    ) -> None:
        self.stepwise = stepwise
        self.is_running = False
        # Store the step configs of this workflow, to be used in send_event
        self._step_configs: dict[str, StepConfig | None] = {}
        for step_name, step_func in workflow._get_steps().items():
            self._step_configs[step_name] = getattr(step_func, "__step_config", None)

        # Init broker machinery
        self._init_broker_data()

        # Global data storage
        self._lock = asyncio.Lock()
        self._state_store: InMemoryStateStore[MODEL_T] | None = None

        # instrumentation
        self._dispatcher = workflow._dispatcher

    async def _init_state_store(self, state_class: MODEL_T) -> None:
        # If a state manager already exists, ensure the requested state type is compatible
        if self._state_store is not None:
            existing_state = await self._state_store.get_state()
            if type(state_class) is not type(existing_state):
                # Existing state type differs from the requested one – this is not allowed
                raise ValueError(
                    f"Cannot initialize with state class {type(state_class)} because it already has a state class {type(existing_state)}"
                )

            # State manager already initialised and compatible – nothing to do
            return

        # First-time initialisation
        self._state_store = InMemoryStateStore(state_class)

    @property
    def store(self) -> InMemoryStateStore[MODEL_T]:
        # Default to DictState if no state manager is initialized
        if self._state_store is None:
            self._state_store = InMemoryStateStore(DictState())

        return self._state_store

    def _init_broker_data(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._tasks: set[asyncio.Task] = set()
        self._broker_log: list[Event] = []
        self._cancel_flag: asyncio.Event = asyncio.Event()
        self._step_flags: dict[str, asyncio.Event] = {}
        self._step_events_holding: list[Event] | None = None
        self._step_lock: asyncio.Lock = asyncio.Lock()
        self._step_condition: asyncio.Condition = asyncio.Condition(
            lock=self._step_lock
        )
        self._step_event_written: asyncio.Condition = asyncio.Condition(
            lock=self._step_lock
        )
        self._accepted_events: list[Tuple[str, str]] = []
        self._retval: RunResultT = None
        # Map the step names that were executed to a list of events they received.
        # This will be serialized, and is needed to resume a Workflow run passing
        # an existing context.
        self._in_progress: dict[str, list[Event]] = defaultdict(list)
        # Keep track of the steps currently running. This is only valid when a
        # workflow is running and won't be serialized. Note that a single step
        # might have multiple workers, so we keep a counter.
        self._currently_running_steps: DefaultDict[str, int] = defaultdict(int)
        # Streaming machinery
        self._streaming_queue: asyncio.Queue = asyncio.Queue()
        # Step-specific instance
        self._event_buffers: dict[str, EventBuffer] = {}

    def _serialize_queue(self, queue: asyncio.Queue, serializer: BaseSerializer) -> str:
        queue_items = list(queue._queue)  # type: ignore
        queue_objs = [serializer.serialize(obj) for obj in queue_items]
        return json.dumps(queue_objs)  # type: ignore

    def _deserialize_queue(
        self,
        queue_str: str,
        serializer: BaseSerializer,
        prefix_queue_objs: list[Any] = [],
    ) -> asyncio.Queue:
        queue_objs = json.loads(queue_str)
        queue_objs = prefix_queue_objs + queue_objs
        queue = asyncio.Queue()  # type: ignore
        for obj in queue_objs:
            event_obj = serializer.deserialize(obj)
            queue.put_nowait(event_obj)
        return queue

    def _deserialize_globals(
        self, serialized_globals: dict[str, Any], serializer: BaseSerializer
    ) -> dict[str, Any]:
        """
        DEPRECATED: Kept to support reloading a Context from an old serialized payload.

        This method is deprecated and will be removed in a future version.
        """
        deserialized_globals = {}
        for key, value in serialized_globals.items():
            try:
                deserialized_globals[key] = serializer.deserialize(value)
            except Exception as e:
                raise ValueError(f"Failed to deserialize value for key {key}: {e}")
        return deserialized_globals

    def to_dict(self, serializer: BaseSerializer | None = None) -> dict[str, Any]:
        serializer = serializer or JsonSerializer()

        # Serialize state using the state manager's method
        state_data = {}
        if self._state_store is not None:
            state_data = self._state_store.to_dict(serializer)

        return {
            "state": state_data,  # Use state manager's serialize method
            "streaming_queue": self._serialize_queue(self._streaming_queue, serializer),
            "queues": {
                k: self._serialize_queue(v, serializer) for k, v in self._queues.items()
            },
            "stepwise": self.stepwise,
            "event_buffers": {
                k: {
                    inner_k: [serializer.serialize(ev) for ev in inner_v]
                    for inner_k, inner_v in v.items()
                }
                for k, v in self._event_buffers.items()
            },
            "in_progress": {
                k: [serializer.serialize(ev) for ev in v]
                for k, v in self._in_progress.items()
            },
            "accepted_events": self._accepted_events,
            "broker_log": [serializer.serialize(ev) for ev in self._broker_log],
            "is_running": self.is_running,
        }

    @classmethod
    def from_dict(
        cls,
        workflow: "Workflow",
        data: dict[str, Any],
        serializer: BaseSerializer | None = None,
    ) -> "Context":
        serializer = serializer or JsonSerializer()

        try:
            context = cls(workflow, stepwise=data["stepwise"])

            # Deserialize state manager using the state manager's method
            if "state" in data:
                context._state_store = InMemoryStateStore.from_dict(
                    data["state"], serializer
                )
            elif "globals" in data:
                # Deserialize legacy globals for backward compatibility
                globals = context._deserialize_globals(data["globals"], serializer)
                context._state_store = InMemoryStateStore(DictState(**globals))

            context._streaming_queue = context._deserialize_queue(
                data["streaming_queue"], serializer
            )

            context._event_buffers = {}
            for buffer_id, type_events_map in data["event_buffers"].items():
                context._event_buffers[buffer_id] = {}
                for event_type, events_list in type_events_map.items():
                    context._event_buffers[buffer_id][event_type] = [
                        serializer.deserialize(ev) for ev in events_list
                    ]

            context._accepted_events = data["accepted_events"]
            context._broker_log = [
                serializer.deserialize(ev) for ev in data["broker_log"]
            ]
            context.is_running = data["is_running"]
            # load back up whatever was in the queue as well as the events whose steps
            # were in progress when the serialization of the Context took place
            context._queues = {
                k: context._deserialize_queue(
                    v, serializer, prefix_queue_objs=data["in_progress"].get(k, [])
                )
                for k, v in data["queues"].items()
            }
            context._in_progress = defaultdict(list)
            return context
        except KeyError as e:
            msg = "Error creating a Context instance: the provided payload has a wrong or old format."
            raise ContextSerdeError(msg) from e

    async def set(self, key: str, value: Any, make_private: bool = False) -> None:
        """
        Store `value` into the Context under `key`.

        DEPRECATED: Use `await ctx.store.set(key, value)` instead.
        This method is deprecated and will be removed in a future version.

        Args:
            key: A unique string to identify the value stored.
            value: The data to be stored.

        Raises:
            ValueError: When make_private is True but a key already exists in the global storage.

        """
        warnings.warn(
            "Context.set(key, value) is deprecated. Use 'await ctx.store.set(key, value)' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if make_private:
            warnings.warn(
                "`make_private` is deprecated and will be ignored", DeprecationWarning
            )

        # Delegate to state manager
        await self.store.set(key, value)

    async def mark_in_progress(self, name: str, ev: Event) -> None:
        """
        Add input event to in_progress dict.

        Args:
            name (str): The name of the step that is in progress.
            ev (Event): The input event that kicked off this step.

        """
        async with self.lock:
            self._in_progress[name].append(ev)

    async def remove_from_in_progress(self, name: str, ev: Event) -> None:
        """
        Remove input event from active steps.

        Args:
            name (str): The name of the step that has been completed.
            ev (Event): The associated input event that kicked of this completed step.

        """
        async with self.lock:
            events = [e for e in self._in_progress[name] if e != ev]
            self._in_progress[name] = events

    async def add_running_step(self, name: str) -> None:
        async with self.lock:
            self._currently_running_steps[name] += 1

    async def remove_running_step(self, name: str) -> None:
        async with self.lock:
            self._currently_running_steps[name] -= 1
            if self._currently_running_steps[name] == 0:
                del self._currently_running_steps[name]

    async def running_steps(self) -> list[str]:
        async with self.lock:
            return list(self._currently_running_steps)

    async def get(self, key: str, default: Any | None = Ellipsis) -> Any:
        """
        Get the value corresponding to `key` from the Context.

        DEPRECATED: Use `await ctx.store.get(key)` instead.
        This method is deprecated and will be removed in a future version.

        Args:
            key: A unique string to identify the value stored.
            default: The value to return when `key` is missing instead of raising an exception.

        Raises:
            ValueError: When there's not value accessible corresponding to `key`.

        """
        warnings.warn(
            "Context.get() is deprecated. Use 'await ctx.store.get()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return await self.store.get(key, default=default)

    @property
    def lock(self) -> asyncio.Lock:
        """Returns a mutex to lock the Context."""
        return self._lock

    @property
    def session(self) -> "Context":  # pragma: no cover
        """This property is provided for backward compatibility."""
        msg = "`session` is deprecated, please use the Context instance directly."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self

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

    def collect_events(
        self, ev: Event, expected: list[Type[Event]], buffer_id: str | None = None
    ) -> list[Event] | None:
        """
        Collects events for buffering in workflows.

        This method adds the current event to the internal buffer and attempts to collect all
        expected event types. If all expected events are found, they will be returned in order.
        Otherwise, it returns None and restores any collected events back to the buffer.

        Args:
            ev (Event): The current event to add to the buffer.
            expected (list[Type[Event]]): list of expected event types to collect.
            buffer_id (str): A unique identifier for the events collected. Ideally this should be
            the step name, so to avoid any interference between different steps. If not provided,
            a stable identifier will be created using the list of expected events.

        Returns:
            list[Event] | None: list of collected events in the order of expected types if all
                                  expected events are found; otherwise None.

        """
        buffer_id = buffer_id or self._get_event_buffer_id(expected)

        if buffer_id not in self._event_buffers:
            self._event_buffers[buffer_id] = defaultdict(list)

        event_type_path = self._get_full_path(type(ev))
        self._event_buffers[buffer_id][event_type_path].append(ev)

        retval: list[Event] = []
        for e_type in expected:
            e_type_path = self._get_full_path(e_type)
            e_instance_list = self._event_buffers[buffer_id].get(e_type_path, [])
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
            self._event_buffers[buffer_id][e_type_path].append(ev_to_restore)

        return None

    def add_holding_event(self, event: Event) -> None:
        """
        Add an event to the list of those collected in current step.

        This is only relevant for stepwise execution.
        """
        if self.stepwise:
            if self._step_events_holding is None:
                self._step_events_holding = []

            self._step_events_holding.append(event)

    def get_holding_events(self) -> list[Event]:
        """Returns a copy of the list of events holding the stepwise execution."""
        if self._step_events_holding is None:
            return []

        return list(self._step_events_holding)

    def send_event(self, message: Event, step: str | None = None) -> None:
        """
        Sends an event to a specific step in the workflow.

        If step is None, the event is sent to all the receivers and we let
        them discard events they don't want.
        """
        self.add_holding_event(message)

        if step is None:
            for queue in self._queues.values():
                queue.put_nowait(message)
        else:
            if step not in self._step_configs:
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            step_config = self._step_configs[step]
            if step_config and type(message) in step_config.accepted_events:
                self._queues[step].put_nowait(message)
            else:
                raise WorkflowRuntimeError(
                    f"Step {step} does not accept event of type {type(message)}"
                )

        self._broker_log.append(message)

    async def wait_for_event(
        self,
        event_type: Type[T],
        waiter_event: Event | None = None,
        waiter_id: str | None = None,
        requirements: dict[str, Any] | None = None,
        timeout: float | None = 2000,
    ) -> T:
        """
        Asynchronously wait for a specific event type to be received.

        If provided, `waiter_event` will be written to the event stream to let the caller know that we're waiting for a response.

        Args:
            event_type: The type of event to wait for
            waiter_event: The event to emit to the event stream to let the caller know that we're waiting for a response
            waiter_id: A unique identifier for this specific wait call. It helps ensure that we only send one `waiter_event` for each `waiter_id`.
            requirements: Optional dict of requirements the event must match
            timeout: Optional timeout in seconds. Defaults to 2000s.

        Returns:
            The event type that was requested.

        Raises:
            asyncio.TimeoutError: If the timeout is reached before receiving matching event

        """
        requirements = requirements or {}

        # Generate a unique key for the waiter
        event_str = self._get_full_path(event_type)
        requirements_str = str(requirements)
        waiter_id = waiter_id or f"waiter_{event_str}_{requirements_str}"

        if waiter_id not in self._queues:
            self._queues[waiter_id] = asyncio.Queue()

        # send the waiter event if it's not already sent
        if waiter_event is not None:
            is_waiting = await self.get(waiter_id, default=False)
            if not is_waiting:
                await self.set(waiter_id, True)
                self.write_event_to_stream(waiter_event)

        while True:
            try:
                event = await asyncio.wait_for(
                    self._queues[waiter_id].get(), timeout=timeout
                )
                if type(event) is event_type:
                    if all(
                        event.get(k, default=None) == v for k, v in requirements.items()
                    ):
                        return event
                    else:
                        continue
            finally:
                await self.set(waiter_id, False)

    def write_event_to_stream(self, ev: Event | None) -> None:
        self._streaming_queue.put_nowait(ev)

    def get_result(self) -> RunResultT:
        """Returns the result of the workflow."""
        return self._retval

    @property
    def streaming_queue(self) -> asyncio.Queue:
        return self._streaming_queue

    def clear(self) -> None:
        """Clear any data stored in the context.

        DEPRECATED: Use `await ctx.store.set(StateCLS())` instead.
        This method is deprecated and will be removed in a future version.
        """
        warnings.warn(
            "Context.clear() is deprecated. Use 'await ctx.store.set(StateCLS())' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Clear the user data storage
        if self._state_store is not None:
            self._state_store._state = self._state_store._state.__class__()

    async def shutdown(self) -> None:
        """
        To be called when a workflow ends.

        We clear all the tasks and set the is_running flag. Note that we
        don't clear _globals or _queues so that the context can be still
        used after the shutdown to fetch data or consume leftover events.
        """
        self.is_running = False
        # Cancel all running tasks
        for task in self._tasks:
            task.cancel()
        # Wait for all tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def add_step_worker(
        self,
        name: str,
        step: Callable,
        config: StepConfig,
        stepwise: bool,
        verbose: bool,
        checkpoint_callback: "CheckpointCallback | None",
        run_id: str,
        service_manager: ServiceManager,
        resource_manager: ResourceManager,
    ) -> None:
        self._tasks.add(
            asyncio.create_task(
                self._step_worker(
                    name=name,
                    step=step,
                    config=config,
                    stepwise=stepwise,
                    verbose=verbose,
                    checkpoint_callback=checkpoint_callback,
                    run_id=run_id,
                    service_manager=service_manager,
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
        stepwise: bool,
        verbose: bool,
        checkpoint_callback: "CheckpointCallback | None",
        run_id: str,
        service_manager: ServiceManager,
        resource_manager: ResourceManager,
    ) -> None:
        while True:
            ev = await self._queues[name].get()
            if type(ev) not in config.accepted_events:
                continue

            # do we need to wait for the step flag?
            if stepwise:
                await self._step_flags[name].wait()

                # clear all flags so that we only run one step
                for flag in self._step_flags.values():
                    flag.clear()

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
                        state_instance = config.context_state_type()
                        await self._init_state_store(state_instance)
                    except Exception as e:
                        raise WorkflowRuntimeError(
                            f"Failed to initialize state of type {config.context_state_type}: {e}"
                        ) from e
                else:
                    # Initialize state manager with DictState by default
                    await self._init_state_store(DictState())

            kwargs: dict[str, Any] = {}
            if config.context_parameter:
                kwargs[config.context_parameter] = self
            for service_definition in config.requested_services:
                service = service_manager.get(
                    service_definition.name, service_definition.default_value
                )
                kwargs[service_definition.name] = service
            for resource in config.resources:
                kwargs[resource.name] = await resource_manager.get(
                    resource=resource.resource
                )
            kwargs[config.event_name] = ev

            # wrap the step with instrumentation
            instrumented_step = self._dispatcher.span(step)

            # - check if its async or not
            # - if not async, run it in an executor
            if asyncio.iscoroutinefunction(step):
                retry_start_at = time.time()
                attempts = 0
                while True:
                    await self.mark_in_progress(name=name, ev=ev)
                    await self.add_running_step(name)
                    try:
                        new_ev = await instrumented_step(**kwargs)
                        kwargs.clear()
                        break  # exit the retrying loop

                    except WorkflowDone:
                        await self.remove_from_in_progress(name=name, ev=ev)
                        raise
                    except Exception as e:
                        if config.retry_policy is None:
                            raise WorkflowRuntimeError(
                                f"Error in step '{name}': {e!s}"
                            ) from e

                        delay = config.retry_policy.next(
                            retry_start_at + time.time(), attempts, e
                        )
                        if delay is None:
                            # We're done retrying
                            raise WorkflowRuntimeError(
                                f"Error in step '{name}': {e!s}"
                            ) from e

                        attempts += 1
                        if verbose:
                            print(
                                f"Step {name} produced an error, retry in {delay} seconds"
                            )
                        await asyncio.sleep(delay)
                    finally:
                        await self.remove_running_step(name)

            else:
                try:
                    run_task = functools.partial(instrumented_step, **kwargs)
                    kwargs.clear()
                    new_ev = await asyncio.get_event_loop().run_in_executor(
                        None, run_task
                    )
                except WorkflowDone:
                    await self.remove_from_in_progress(name=name, ev=ev)
                    raise
                except Exception as e:
                    raise WorkflowRuntimeError(f"Error in step '{name}': {e!s}") from e

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

            if stepwise:
                async with self._step_condition:
                    await self._step_condition.wait()

                    if new_ev is not None:
                        self.add_holding_event(new_ev)
                    self._step_event_written.notify()  # shares same lock

                    await self.remove_from_in_progress(name=name, ev=ev)

                    # for stepwise Checkpoint after handler.run_step() call
                    if checkpoint_callback:
                        await checkpoint_callback(
                            run_id=run_id,
                            ctx=self,
                            last_completed_step=name,
                            input_ev=ev,
                            output_ev=new_ev,
                        )
            else:
                # for regular execution, Checkpoint just before firing the next event
                await self.remove_from_in_progress(name=name, ev=ev)
                if checkpoint_callback:
                    await checkpoint_callback(
                        run_id=run_id,
                        ctx=self,
                        last_completed_step=name,
                        input_ev=ev,
                        output_ev=new_ev,
                    )

                # InputRequiredEvent's are special case and need to be written to the stream
                # this way, the user can access and respond to the event
                if isinstance(new_ev, InputRequiredEvent):
                    self.write_event_to_stream(new_ev)
                elif new_ev is not None:
                    self.send_event(new_ev)

    def add_cancel_worker(self) -> None:
        self._tasks.add(asyncio.create_task(self._cancel_worker()))

    async def _cancel_worker(self) -> None:
        try:
            await self._cancel_flag.wait()
            raise WorkflowCancelledByUser
        except asyncio.CancelledError:
            return
