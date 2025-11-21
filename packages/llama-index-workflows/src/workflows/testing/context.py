from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generic,
    Optional,
    Type,
    TypeVar,
    cast,
)

from workflows.context.context import Context
from workflows.context.state_store import MODEL_T, DictState, InMemoryStateStore
from workflows.events import Event

if TYPE_CHECKING:  # pragma: no cover
    from workflows import Workflow

T = TypeVar("T", bound=Event)


class TestStepContext(Context[MODEL_T], Generic[MODEL_T]):
    """
    A test context that tracks interactions for testing workflow steps in isolation.

    This context implementation records all events sent, events written to stream,
    and provides convenient accessors for test assertions. It's designed to be used
    in unit tests where you want to test a single step without running a full workflow.

    Unlike the regular Context, this doesn't require a running broker and can be
    created with an initial state for testing specific scenarios.

    Attributes:
        sent_events (list[tuple[Event, str | None]]): All events sent via send_event,
            stored as (event, target_step) tuples.
        streamed_events (list[Event]): All events written to the stream via
            write_event_to_stream.
        store (InMemoryStateStore[MODEL_T]): The state store, same as regular Context.

    Examples:
        Testing a step with typed state:

        ```python
        from pydantic import BaseModel
        from workflows.testing import TestStepContext

        class MyState(BaseModel):
            counter: int = 0

        # Create context with initial state
        ctx = TestStepContext.create(MyState(counter=5))

        # Test your step
        @step
        async def my_step(self, ctx: Context[MyState], ev: StartEvent) -> StopEvent:
            state = await ctx.store.get_state()
            ctx.send_event(CustomEvent(count=state.counter))
            ctx.write_event_to_stream(ProgressEvent())
            return StopEvent(result=state.counter)

        # Call the step
        result = await my_step(None, ctx, StartEvent())

        # Assert on the interactions
        assert len(ctx.sent_events) == 1
        assert isinstance(ctx.sent_events[0][0], CustomEvent)
        assert len(ctx.streamed_events) == 1
        assert isinstance(ctx.streamed_events[0], ProgressEvent)
        ```

        Testing with DictState:

        ```python
        ctx = TestStepContext.create(DictState(count=10))
        count = await ctx.store.get("count")
        assert count == 10
        ```
    """

    def __init__(
        self,
        workflow: "Workflow",
        initial_state: Optional[MODEL_T] = None,
    ) -> None:
        # Initialize the parent Context but we won't use its broker
        super().__init__(workflow)

        # Override the state store if initial state is provided
        if initial_state is not None:
            self._state_store = InMemoryStateStore(initial_state)

        # Track all interactions for test assertions
        self.sent_events: list[tuple[Event, Optional[str]]] = []
        self.streamed_events: list[Event] = []

    @classmethod
    def create(
        cls, initial_state: Optional[MODEL_T] = None
    ) -> "TestStepContext[MODEL_T]":
        """
        Create a TestStepContext without requiring a workflow instance.

        This is a convenience factory for creating test contexts in isolation.
        A minimal dummy workflow is created internally.

        Args:
            initial_state (MODEL_T | None): Initial state for the context store.
                If None, defaults to DictState().

        Returns:
            TestStepContext[MODEL_T]: A new test context ready for use.

        Examples:
            ```python
            # With typed state
            from pydantic import BaseModel

            class MyState(BaseModel):
                value: int = 0

            ctx = TestStepContext.create(MyState(value=42))

            # With DictState
            ctx = TestStepContext.create(DictState(key="value"))

            # With empty DictState
            ctx = TestStepContext.create()
            ```
        """
        from workflows import Workflow

        # Create a bare Workflow instance using object.__new__() to bypass validation
        workflow = object.__new__(Workflow)
        # Set minimal required attributes
        workflow._timeout = None
        workflow._verbose = False
        workflow._disable_validation = True
        workflow._num_concurrent_runs = None
        workflow._sem = None
        workflow._dispatcher = None

        # If no initial state provided, use empty DictState
        if initial_state is None:
            initial_state = cast(MODEL_T, DictState())

        return cls(workflow, initial_state=initial_state)

    @property
    def is_running(self) -> bool:
        """Always returns True for test contexts since they're always ready to use."""
        return True

    def send_event(self, message: Event, step: Optional[str] = None) -> None:
        """
        Record an event being sent.

        Unlike the real Context, this doesn't dispatch to a broker but instead
        records the event for later assertion in tests.

        Args:
            message (Event): The event being sent.
            step (str | None): The target step name, if any.
        """
        self.sent_events.append((message, step))

    def write_event_to_stream(self, ev: Optional[Event]) -> None:
        """
        Record an event being written to the stream.

        Unlike the real Context, this doesn't write to a broker queue but
        records the event for later assertion in tests.

        Args:
            ev (Event | None): The event to stream. None values are ignored.
        """
        if ev is not None:
            self.streamed_events.append(ev)

    def clear_tracked_events(self) -> None:
        """
        Clear all tracked events.

        Useful when testing multiple operations with the same context and
        you want to reset tracking between operations.

        Examples:
            ```python
            ctx = TestStepContext.create()

            # First operation
            await my_step(ctx, StartEvent())
            assert len(ctx.sent_events) == 1

            # Reset and test second operation
            ctx.clear_tracked_events()
            await another_step(ctx, AnotherEvent())
            assert len(ctx.sent_events) == 1  # Only the second operation
            ```
        """
        self.sent_events.clear()
        self.streamed_events.clear()

    # Methods that should not be called in test contexts
    def collect_events(
        self, ev: Event, expected: list[Type[Event]], buffer_id: Optional[str] = None
    ) -> Optional[list[Event]]:
        """
        Not supported in TestStepContext.

        This requires broker state that isn't available in isolated step testing.
        If you need to test collect_events, use WorkflowTestRunner instead.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "collect_events is not supported in TestStepContext. "
            "Use WorkflowTestRunner for testing steps that use collect_events."
        )

    async def wait_for_event(
        self,
        event_type: Type[T],
        waiter_event: Optional[Event] = None,
        waiter_id: Optional[str] = None,
        requirements: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = 2000,
    ) -> T:
        """
        Not supported in TestStepContext.

        This requires broker state and async coordination that isn't available
        in isolated step testing. If you need to test wait_for_event, use
        WorkflowTestRunner instead.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "wait_for_event is not supported in TestStepContext. "
            "Use WorkflowTestRunner for testing steps that use wait_for_event."
        )

    def stream_events(self) -> AsyncGenerator[Event, None]:
        """
        Not supported in TestStepContext.

        Use the `streamed_events` list attribute instead to inspect events
        that were written to the stream.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "stream_events is not supported in TestStepContext. "
            "Use the streamed_events attribute to inspect written events."
        )

    async def running_steps(self) -> list[str]:
        """
        Not supported in TestStepContext.

        This requires broker state that isn't available in isolated step testing.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "running_steps is not supported in TestStepContext. "
            "Use WorkflowTestRunner for testing steps that need broker state."
        )
