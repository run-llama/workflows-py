# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import functools
import uuid
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generic,
    Type,
    TypeVar,
    cast,
)

from pydantic import BaseModel, ValidationError

from workflows.context.context_types import SerializedContext
from workflows.decorators import StepConfig
from workflows.errors import (
    ContextSerdeError,
    WorkflowRuntimeError,
)
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
)
from workflows.handler import WorkflowHandler
from workflows.plugins.basic import basic_runtime
from workflows.runtime.broker import WorkflowBroker
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import Plugin, WorkflowRuntime
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
    Global, per-run context for a `Workflow`. Provides an interface into the
    underlying broker run, for both external (workflow run oberservers) and
    internal consumption by workflow steps.

    The `Context` coordinates event delivery between steps, tracks in-flight work,
    exposes a global state store, and provides utilities for streaming and
    synchronization. It is created by a `Workflow` at run time and can be
    persisted and restored.

    Args:
        workflow (Workflow): The owning workflow instance. Used to infer
            step configuration and instrumentation.
        previous_context: A previous context snapshot to resume from.
        serializer: A serializer to use for serializing and deserializing the current and previous context snapshots.

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

    # Backing state store; serialized as `state`
    _state_store: InMemoryStateStore[MODEL_T]
    _broker_run: WorkflowBroker[MODEL_T] | None
    _plugin: Plugin
    _workflow: Workflow

    def __init__(
        self,
        workflow: Workflow,
        previous_context: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
        plugin: Plugin = basic_runtime,
    ) -> None:
        self._serializer = serializer or JsonSerializer()
        self._broker_run = None
        self._plugin = plugin
        self._workflow = workflow

        # parse the serialized context
        serializer = serializer or JsonSerializer()
        if previous_context is not None:
            try:
                # Auto-detect and convert V0 to V1 if needed
                previous_context_parsed = SerializedContext.from_dict_auto(
                    previous_context
                )
                # validate it fully parses synchronously to avoid delayed validation errors
                BrokerState.from_serialized(
                    previous_context_parsed, workflow, serializer
                )
            except ValidationError as e:
                raise ContextSerdeError(
                    f"Context dict specified in an invalid format: {e}"
                ) from e
        else:
            previous_context_parsed = SerializedContext()

        self._init_snapshot = previous_context_parsed

        # initialization of the state store is a bit complex, due to inferring and validating its type from the
        # provided workflow context args

        state_types: set[Type[BaseModel]] = set()
        for _, step_func in workflow._get_steps().items():
            step_config: StepConfig = step_func._step_config
            if (
                step_config.context_state_type is not None
                and step_config.context_state_type != DictState
                and issubclass(step_config.context_state_type, BaseModel)
            ):
                state_type = step_config.context_state_type
                state_types.add(state_type)

        if len(state_types) > 1:
            raise ValueError(
                "Multiple state types are not supported. Make sure that each Context[...] has the same generic state type. Found: "
                + ", ".join([state_type.__name__ for state_type in state_types])
            )
        state_type = state_types.pop() if state_types else DictState
        if previous_context_parsed.state:
            # perhaps offer a way to clear on invalid
            store_state = InMemoryStateStore.from_dict(
                previous_context_parsed.state, serializer
            )
            if store_state.state_type != state_type:
                raise ValueError(
                    f"State type mismatch. Workflow context expected {state_type.__name__}, got {store_state.state_type.__name__}"
                )
            self._state_store = cast(InMemoryStateStore[MODEL_T], store_state)
        else:
            try:
                state_instance = cast(MODEL_T, state_type())
                self._state_store = InMemoryStateStore(state_instance)
            except Exception as e:
                raise WorkflowRuntimeError(
                    f"Failed to initialize state of type {state_type}. Does your state define defaults for all fields? Original error:\n{e}"
                ) from e

    @property
    def is_running(self) -> bool:
        """Whether the workflow is currently running."""
        if self._broker_run is None:
            return self._init_snapshot.is_running
        else:
            return self._broker_run.is_running

    def _init_broker(
        self, workflow: Workflow, plugin: WorkflowRuntime | None = None
    ) -> WorkflowBroker[MODEL_T]:
        if self._broker_run is not None:
            raise WorkflowRuntimeError("Broker already initialized")
        # Initialize a runtime plugin (asyncio-based by default)
        runtime: WorkflowRuntime = plugin or self._plugin.new_runtime(str(uuid.uuid4()))
        # Initialize the new broker implementation (broker2)
        broker: WorkflowBroker[MODEL_T] = WorkflowBroker(
            workflow=workflow,
            context=cast("Context[MODEL_T]", self),
            runtime=runtime,
            plugin=self._plugin,
        )
        self._broker_run = broker
        return broker

    def _workflow_run(
        self,
        workflow: Workflow,
        start_event: StartEvent | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> WorkflowHandler:
        """
        called by package internally from the workflow to run it
        """
        prev_broker: WorkflowBroker[MODEL_T] | None = None
        if self._broker_run is not None:
            prev_broker = self._broker_run
            self._broker_run = None

        self._broker_run = self._init_broker(workflow)

        async def before_start() -> None:
            if prev_broker is not None:
                try:
                    await prev_broker.shutdown()
                except Exception:
                    pass
            if semaphore is not None:
                await semaphore.acquire()

        async def after_complete() -> None:
            if semaphore is not None:
                semaphore.release()

        state = BrokerState.from_serialized(
            self._init_snapshot, workflow, self._serializer
        )
        return self._broker_run.start(
            workflow=workflow,
            previous=state,
            start_event=start_event,
            before_start=before_start,
            after_complete=after_complete,
        )

    def _workflow_cancel_run(self) -> None:
        """
        Called internally from the handler to cancel a context's run
        """
        self._running_broker.cancel_run()

    @property
    def _running_broker(self) -> WorkflowBroker[MODEL_T]:
        if self._broker_run is None:
            raise WorkflowRuntimeError(
                "Workflow run is not yet running. Make sure to only call this method after the context has been passed to a workflow.run call."
            )
        return self._broker_run

    @property
    def store(self) -> InMemoryStateStore[MODEL_T]:
        """Typed, process-local state store shared across steps.

        If no state was initialized yet, a default
        [DictState][workflows.context.state_store.DictState] store is created.

        Returns:
            InMemoryStateStore[MODEL_T]: The state store instance.
        """
        return self._state_store

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
        serializer = serializer or self._serializer

        # Serialize state using the state manager's method
        state_data = {}
        if self._state_store is not None:
            state_data = self._state_store.to_dict(serializer)

        # Get the broker state - either from the running broker or from the init snapshot
        if self._broker_run is not None:
            broker_state = self._broker_run._state
        else:
            # Deserialize the init snapshot to get a BrokerState, then re-serialize it
            # This ensures we always output the current format
            broker_state = BrokerState.from_serialized(
                self._init_snapshot, self._workflow, serializer
            )

        context = broker_state.to_serialized(serializer)
        context.state = state_data
        # mode="python" to support pickling over json if one so chooses. This should perhaps be moved into the serializers
        return context.model_dump(mode="python")

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
        try:
            return cls(workflow, previous_context=data, serializer=serializer)
        except KeyError as e:
            msg = "Error creating a Context instance: the provided payload has a wrong or old format."
            raise ContextSerdeError(msg) from e

    async def running_steps(self) -> list[str]:
        """Return the list of currently running step names.

        Returns:
            list[str]: Names of steps that have at least one active worker.
        """
        return await self._running_broker.running_steps()

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
        return self._running_broker.collect_events(ev, expected, buffer_id)

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
        return self._running_broker.send_event(message, step)

    async def wait_for_event(
        self,
        event_type: Type[T],
        waiter_event: Event | None = None,
        waiter_id: str | None = None,
        requirements: dict[str, Any] | None = None,
        timeout: float | None = 2000,
    ) -> T:
        """Wait for the next matching event of type `event_type`.

        The runtime pauses by throwing an internal control-flow exception and replays
        the entire step when the event arrives, so keep this call near the top of the
        step and make any preceding work safe to repeat.

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
        return await self._running_broker.wait_for_event(
            event_type, waiter_event, waiter_id, requirements, timeout
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
        self._running_broker.write_event_to_stream(ev)

    def get_result(self) -> RunResultT:
        """Return the final result of the workflow run.

        Deprecated:
            This method is deprecated and will be removed in a future release.
            Prefer awaiting the handler returned by `Workflow.run`, e.g.:
            `result = await workflow.run(..., ctx=ctx)`.

        Examples:
            ```python
            # Preferred
            result = await my_workflow.run(..., ctx=ctx)

            # Deprecated
            result_agent = ctx.get_result()
            ```

        Returns:
            RunResultT: The value provided via a `StopEvent`.
        """
        _warn_get_result()
        if self._running_broker._handler is None:
            raise WorkflowRuntimeError("Workflow handler is not set")
        return self._running_broker._handler.result()

    def stream_events(self) -> AsyncGenerator[Event, None]:
        """The internal queue used for streaming events to callers."""
        return self._running_broker.stream_published_events()

    @property
    def streaming_queue(self) -> asyncio.Queue:
        """Deprecated queue-based event stream.

        Returns an asyncio.Queue that is populated by iterating this context's
        stream_events(). A deprecation warning is emitted once per process.
        """
        _warn_streaming_queue()
        q: asyncio.Queue[Event] = asyncio.Queue()

        async def _pump() -> None:
            async for ev in self.stream_events():
                await q.put(ev)
                if isinstance(ev, StopEvent):
                    break

        try:
            asyncio.create_task(_pump())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.create_task(_pump())
        return q


@functools.lru_cache(maxsize=1)
def _warn_get_result() -> None:
    warnings.warn(
        (
            "Context.get_result() is deprecated and will be removed in a future "
            "release. Prefer awaiting the WorkflowHandler returned by "
            "Workflow.run: `result = await workflow.run(..., ctx=ctx)`."
        ),
        DeprecationWarning,
        stacklevel=2,
    )


@functools.lru_cache(maxsize=1)
def _warn_streaming_queue() -> None:
    warnings.warn(
        (
            "Context.streaming_queue is deprecated and will be removed in a future "
            "release. Prefer iterating Context.stream_events(): "
            "`async for ev in ctx.stream_events(): ...`"
        ),
        DeprecationWarning,
        stacklevel=2,
    )
