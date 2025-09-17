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
    cast,
)

from workflows.decorators import StepConfig
from workflows.errors import (
    ContextSerdeError,
    WorkflowCancelledByUser,
    WorkflowDone,
    WorkflowRuntimeError,
)
from workflows.events import (
    Event,
    InputRequiredEvent,
    EventsQueueChanged,
    StepStateChanged,
    StepState,
)
from workflows.resource import ResourceManager
from workflows.types import RunResultT

from .serializers import BaseSerializer, JsonSerializer
from .state_store import MODEL_T, DictState, InMemoryStateStore

if TYPE_CHECKING:  # pragma: no cover
    from workflows import Workflow

T = TypeVar("T", bound=Event)
EventBuffer = dict[str, list[Event]]


# Only warn once about unserializable keys
class UnserializableKeyWarning(Warning):
    pass


warnings.simplefilter("once", UnserializableKeyWarning)


class Context(Generic[MODEL_T]):
    """
    Global, per-run context and event broker for a `Workflow`.

    The `Context` coordinates event delivery between steps, tracks in-flight work,
    exposes a global state store, and provides utilities for streaming and
    synchronization. It is created by a `Workflow` at run time and can be
    persisted and restored.

    Args:
        workflow (Workflow): The owning workflow instance. Used to infer
            step configuration and instrumentation.

    Attributes:
        is_running (bool): Whether the workflow is currently running.
        store (InMemoryStateStore[MODEL_T]): Type-safe, async state store shared
            across steps. See also
            [InMemoryStateStore][workflows.context.state_store.InMemoryStateStore].

    Examples:
        Basic usage inside a step:

        ```python
        from workflows import step
        from workflows.events import StartEvent, StopEvent

        @step
        async def start(self, ctx: Context, ev: StartEvent) -> StopEvent:
            await ctx.store.set("query", ev.topic)
            ctx.write_event_to_stream(ev)  # surface progress to UI
            return StopEvent(result="ok")
        ```

        Persisting the state of a workflow across runs:

        ```python
        from workflows import Context

        # Create a context and run the workflow with the same context
        ctx = Context(my_workflow)
        result_1 = await my_workflow.run(..., ctx=ctx)
        result_2 = await my_workflow.run(..., ctx=ctx)

        # Serialize the context and restore it
        ctx_dict = ctx.to_dict()
        restored_ctx = Context.from_dict(my_workflow, ctx_dict)
        result_3 = await my_workflow.run(..., ctx=restored_ctx)
        ```


    See Also:
        - [Workflow][workflows.Workflow]
        - [Event][workflows.events.Event]
        - [InMemoryStateStore][workflows.context.state_store.InMemoryStateStore]
    """

    # These keys are set by pre-built workflows and
    # are known to be unserializable in some cases.
    known_unserializable_keys = ("memory",)

    def __init__(self, workflow: "Workflow") -> None:
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
        self._waiting_ids: set[str] = set()

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
        """Typed, process-local state store shared across steps.

        If no state was initialized yet, a default
        [DictState][workflows.context.state_store.DictState] store is created.

        Returns:
            InMemoryStateStore[MODEL_T]: The state store instance.
        """
        # Default to DictState if no state manager is initialized
        if self._state_store is None:
            # DictState is designed to be compatible with any MODEL_T as the default fallback
            default_store = InMemoryStateStore(DictState())
            self._state_store = cast(InMemoryStateStore[MODEL_T], default_store)

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
        return json.dumps(queue_objs)

    def _deserialize_queue(
        self,
        queue_str: str,
        serializer: BaseSerializer,
        prefix_queue_objs: list[Any] = [],
    ) -> asyncio.Queue:
        queue_objs = json.loads(queue_str)
        queue_objs = prefix_queue_objs + queue_objs
        queue: asyncio.Queue = asyncio.Queue()
        for obj in queue_objs:
            event_obj = serializer.deserialize(obj)
            queue.put_nowait(event_obj)
        return queue

    def to_dict(self, serializer: BaseSerializer | None = None) -> dict[str, Any]:
        """Serialize the context to a JSON-serializable dict.

        Persists the global state store, event queues, buffers, accepted events,
        broker log, and running flag. This payload can be fed to
        [from_dict][workflows.context.context.Context.from_dict] to resume a run
        or carry state across runs.

        Args:
            serializer (BaseSerializer | None): Value serializer used for state
                and event payloads. Defaults to
                [JsonSerializer][workflows.context.serializers.JsonSerializer].

        Returns:
            dict[str, Any]: A dict suitable for JSON encoding and later
            restoration via `from_dict`.

        See Also:
            - [InMemoryStateStore.to_dict][workflows.context.state_store.InMemoryStateStore.to_dict]

        Examples:
            ```python
            ctx_dict = ctx.to_dict()
            my_db.set("key", json.dumps(ctx_dict))

            ctx_dict = my_db.get("key")
            restored_ctx = Context.from_dict(my_workflow, json.loads(ctx_dict))
            result = await my_workflow.run(..., ctx=restored_ctx)
            ```
        """
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
            "waiting_ids": list(self._waiting_ids),
        }

    @classmethod
    def from_dict(
        cls,
        workflow: "Workflow",
        data: dict[str, Any],
        serializer: BaseSerializer | None = None,
    ) -> "Context[MODEL_T]":
        """Reconstruct a `Context` from a serialized payload.

        Args:
            workflow (Workflow): The workflow instance that will own this
                context.
            data (dict[str, Any]): Payload produced by
                [to_dict][workflows.context.context.Context.to_dict].
            serializer (BaseSerializer | None): Serializer used to decode state
                and events. Defaults to JSON.

        Returns:
            Context[MODEL_T]: A context instance initialized with the persisted
                state and queues.

        Raises:
            ContextSerdeError: If the payload is missing required fields or is
                in an incompatible format.

        Examples:
            ```python
            ctx_dict = ctx.to_dict()
            my_db.set("key", json.dumps(ctx_dict))

            ctx_dict = my_db.get("key")
            restored_ctx = Context.from_dict(my_workflow, json.loads(ctx_dict))
            result = await my_workflow.run(..., ctx=restored_ctx)
            ```
        """
        serializer = serializer or JsonSerializer()

        try:
            context = cls(workflow)

            # Deserialize state manager using the state manager's method
            if "state" in data:
                context._state_store = cast(
                    InMemoryStateStore[MODEL_T],
                    InMemoryStateStore.from_dict(data["state"], serializer),
                )

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

            # restore waiting ids for hitl
            context._waiting_ids = set(data["waiting_ids"])

            return context
        except KeyError as e:
            msg = "Error creating a Context instance: the provided payload has a wrong or old format."
            raise ContextSerdeError(msg) from e

    async def mark_in_progress(self, name: str, ev: Event, worker_id: str = "") -> None:
        """
        Add input event to in_progress dict.

        Args:
            name (str): The name of the step that is in progress.
            ev (Event): The input event that kicked off this step.

        """
        async with self.lock:
            self.write_event_to_stream(
                StepStateChanged(
                    step_state=StepState.IN_PROGRESS,
                    name=name,
                    input_event_name=(str(type(ev))),
                    worker_id=worker_id,
                )
            )
            self._in_progress[name].append(ev)

    async def remove_from_in_progress(
        self, name: str, ev: Event, worker_id: str = ""
    ) -> None:
        """
        Remove input event from active steps.

        Args:
            name (str): The name of the step that has been completed.
            ev (Event): The associated input event that kicked off this completed step.

        """
        async with self.lock:
            self.write_event_to_stream(
                StepStateChanged(
                    step_state=StepState.NOT_IN_PROGRESS,
                    name=name,
                    input_event_name=(str(type(ev))),
                    worker_id=worker_id,
                )
            )
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
        """Return the list of currently running step names.

        Returns:
            list[str]: Names of steps that have at least one active worker.
        """
        async with self.lock:
            return list(self._currently_running_steps)

    @property
    def lock(self) -> asyncio.Lock:
        """Returns a mutex to lock the Context."""
        return self._lock

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
        Buffer events until all expected types are available, then return them.

        This utility is helpful when a step can receive multiple event types
        and needs to proceed only when it has a full set. The returned list is
        ordered according to `expected`.

        Args:
            ev (Event): The incoming event to add to the buffer.
            expected (list[Type[Event]]): Event types to collect, in order.
            buffer_id (str | None): Optional stable key to isolate buffers across
                steps or workers. Defaults to an internal key derived from the
                task name or expected types.

        Returns:
            list[Event] | None: The events in the requested order when complete,
            otherwise `None`.

        Examples:
            ```python
            @step
            async def synthesize(
                self, ctx: Context, ev: QueryEvent | RetrieveEvent
            ) -> StopEvent | None:
                events = ctx.collect_events(ev, [QueryEvent, RetrieveEvent])
                if events is None:
                    return None
                query_ev, retrieve_ev = events
                # ... proceed with both inputs present ...
            ```

        See Also:
            - [Event][workflows.events.Event]
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

    def send_event(self, message: Event, step: str | None = None) -> None:
        """Dispatch an event to one or all workflow steps.

        If `step` is omitted, the event is broadcast to all step queues and
        non-matching steps will ignore it. When `step` is provided, the target
        step must accept the event type or a
        [WorkflowRuntimeError][workflows.errors.WorkflowRuntimeError] is raised.

        Args:
            message (Event): The event to enqueue.
            step (str | None): Optional step name to target.

        Raises:
            WorkflowRuntimeError: If the target step does not exist or does not
                accept the event type.

        Examples:
            It's common to use this method to fan-out events:

            ```python
            @step
            async def my_step(self, ctx: Context, ev: StartEvent) -> WorkerEvent | GatherEvent:
                for i in range(10):
                    ctx.send_event(WorkerEvent(msg=i))
                return GatherEvent()
            ```

            You also see this method used from the caller side to send events into the workflow:

            ```python
            handler = my_workflow.run(...)
            async for ev in handler.stream_events():
                if isinstance(ev, SomeEvent):
                    handler.ctx.send_event(SomeOtherEvent(msg="Hello!"))

            result = await handler
            ```
        """
        if step is None:
            for name, queue in self._queues.items():
                queue.put_nowait(message)
                self.write_event_to_stream(
                    EventsQueueChanged(name=name, size=queue.qsize())
                )
        else:
            if step not in self._step_configs:
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            step_config = self._step_configs[step]
            if step_config and type(message) in step_config.accepted_events:
                self._queues[step].put_nowait(message)
                self.write_event_to_stream(
                    EventsQueueChanged(name=step, size=self._queues[step].qsize())
                )
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
        """Wait for the next matching event of type `event_type`.

        Optionally emits a `waiter_event` to the event stream once per `waiter_id` to
        inform callers that the workflow is waiting for external input.
        This helps to prevent duplicate waiter events from being sent to the event stream.

        Args:
            event_type (type[T]): Concrete event class to wait for.
            waiter_event (Event | None): Optional event to write to the stream
                once when the wait begins.
            waiter_id (str | None): Stable identifier to avoid emitting multiple
                waiter events for the same logical wait.
            requirements (dict[str, Any] | None): Key/value filters that must be
                satisfied by the event via `event.get(key) == value`.
            timeout (float | None): Max seconds to wait. `None` means no
                timeout. Defaults to 2000 seconds.

        Returns:
            T: The received event instance of the requested type.

        Raises:
            asyncio.TimeoutError: If the timeout elapses.

        Examples:
            ```python
            @step
            async def my_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
                response = await ctx.wait_for_event(
                    HumanResponseEvent,
                    waiter_event=InputRequiredEvent(msg="What's your name?"),
                    waiter_id="user_name",
                    timeout=60,
                )
                return StopEvent(result=response.response)
            ```
        """
        requirements = requirements or {}

        # Generate a unique key for the waiter
        event_str = self._get_full_path(event_type)
        requirements_str = str(requirements)
        waiter_id = waiter_id or f"waiter_{event_str}_{requirements_str}"

        if waiter_id not in self._queues:
            self._queues[waiter_id] = asyncio.Queue()
            self.write_event_to_stream(
                EventsQueueChanged(name=waiter_id, size=self._queues[waiter_id].qsize())
            )

        # send the waiter event if it's not already sent
        if waiter_event is not None:
            is_waiting = waiter_id in self._waiting_ids
            if not is_waiting:
                self._waiting_ids.add(waiter_id)
                self.write_event_to_stream(waiter_event)

        while True:
            event = await asyncio.wait_for(
                self._queues[waiter_id].get(), timeout=timeout
            )
            if type(event) is event_type:
                if all(getattr(event, k, None) == v for k, v in requirements.items()):
                    if waiter_id in self._waiting_ids:
                        self._waiting_ids.remove(waiter_id)
                    return event
                else:
                    continue
            self.write_event_to_stream(
                EventsQueueChanged(
                    name=waiter_id,
                    size=self._queues[waiter_id].qsize(),
                )
            )

    def write_event_to_stream(self, ev: Event | None) -> None:
        """Enqueue an event for streaming to [WorkflowHandler]](workflows.handler.WorkflowHandler).

        Args:
            ev (Event | None): The event to stream. `None` can be used as a
                sentinel in some streaming modes.

        Examples:
            ```python
            @step
            async def my_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
                ctx.write_event_to_stream(ev)
                return StopEvent(result="ok")
            ```
        """
        self._streaming_queue.put_nowait(ev)

    def get_result(self) -> RunResultT:
        """Return the final result of the workflow run.

        Examples:
            ```python
            result = await my_workflow.run(..., ctx=ctx)
            result_agent = ctx.get_result()
            ```

        Returns:
            RunResultT: The value provided via a `StopEvent`.
        """
        return self._retval

    @property
    def streaming_queue(self) -> asyncio.Queue:
        """The internal queue used for streaming events to callers."""
        return self._streaming_queue

    async def shutdown(self) -> None:
        """Shut down the workflow run and clean up background tasks.

        Cancels all outstanding workers, waits for them to finish, and marks the
        context as not running. Queues and state remain available so callers can
        inspect or drain leftover events.
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
            ev = await self._queues[name].get()
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
                retry_start_at = time.time()
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
                self.send_event(new_ev)

    def add_cancel_worker(self) -> None:
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
