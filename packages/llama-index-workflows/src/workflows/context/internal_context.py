"""Internal context - for step execution within workflows."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Coroutine, Generic, Type, TypeVar, cast

from workflows.context.context_types import MODEL_T
from workflows.context.state_store import StateStore
from workflows.errors import WorkflowRuntimeError
from workflows.runtime.types.results import (
    AddCollectedEvent,
    AddWaiter,
    DeleteCollectedEvent,
    DeleteWaiter,
    StepWorkerContext,
    StepWorkerStateContextVar,
    WaitingForEvent,
)
from workflows.runtime.types.ticks import TickAddEvent

if TYPE_CHECKING:
    from workflows.events import Event
    from workflows.runtime.types.plugin import InternalRunAdapter
    from workflows.workflow import Workflow

T = TypeVar("T", bound="Event")

logger = logging.getLogger(__name__)


class InternalContext(Generic[MODEL_T]):
    """Context for code running inside workflow step functions.

    Provides access to state store, event collection, waiting for events,
    and publishing to the event stream.
    """

    _internal_adapter: InternalRunAdapter
    _workflow: Workflow
    _workers: list[asyncio.Task[Any]]

    def __init__(
        self,
        internal_adapter: InternalRunAdapter,
        workflow: Workflow,
    ) -> None:
        self._internal_adapter = internal_adapter
        self._workflow = workflow
        self._workers = []

    def _execute_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Execute a coroutine as a tracked background task."""
        task = asyncio.create_task(coro)
        self._workers.append(task)

        def _on_done(t: asyncio.Task[Any]) -> None:
            try:
                self._workers.remove(t)
            except ValueError:
                # Task was already cleared during shutdown or cleanup.
                pass
            # Log exceptions from fire-and-forget tasks (cancelled is not an error)
            if not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    logger.error(
                        "Background task failed with exception",
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )

        task.add_done_callback(_on_done)
        return task

    def cancel_background_tasks(self) -> None:
        """Cancel all tracked background tasks."""
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()

    async def _finalize_step(self) -> None:
        """Await all background tasks and finalize the step.

        Called after a step function completes to ensure all fire-and-forget
        operations (e.g., write_event_to_stream, send_event) complete before
        returning control to the control loop. This prevents non-deterministic
        ordering of durable operations on replay.
        """
        workers = self._workers[:]
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        await self._internal_adapter.finalize_step()

    @staticmethod
    def _get_step_ctx(fn: str) -> StepWorkerContext:
        """Get the current step worker context. Raises if not in a step."""
        try:
            return StepWorkerStateContextVar.get()
        except LookupError:
            raise WorkflowRuntimeError(
                f"{fn} may only be called from within a step function"
            )

    @property
    def store(self) -> StateStore[MODEL_T]:
        """Access workflow state store."""
        state_store = self._internal_adapter.get_state_store()
        if state_store is None:
            raise RuntimeError("State store not available from adapter")
        return state_store  # type: ignore[return-value]

    def collect_events(
        self,
        ev: Event,
        expected: list[Type[Event]],
        buffer_id: str | None = None,
    ) -> list[Event] | None:
        """Collect events until all expected types are received."""
        step_ctx = self._get_step_ctx(fn="collect_events")

        # If no events are expected, return an empty list immediately
        if not expected:
            return []

        buffer_id = buffer_id or "default"
        collected_events = step_ctx.state.collected_events.get(buffer_id, [])

        remaining_event_types = Counter(expected) - Counter(
            [type(e) for e in collected_events]
        )

        if remaining_event_types != Counter([type(ev)]):
            if type(ev) in remaining_event_types:
                step_ctx.returns.return_values.append(
                    AddCollectedEvent(event_id=buffer_id, event=ev)
                )
            return None

        total = []
        by_type: dict[Type[Event], list[Event]] = defaultdict(list)
        for e in collected_events + [ev]:
            by_type[type(e)].append(e)
        # order by expected type
        for e_type in expected:
            total.append(by_type[e_type].pop(0))
        # Clear the collected events when the step is complete
        step_ctx.returns.return_values.append(DeleteCollectedEvent(event_id=buffer_id))
        return total

    def send_event(self, message: Event, step: str | None = None) -> None:
        """Send an event to trigger another step."""
        if step is not None:
            self._workflow._validate_valid_step_message(step, message)

        self._execute_task(
            self._internal_adapter.send_event(
                TickAddEvent(event=message, step_name=step)
            )
        )

    async def wait_for_event(
        self,
        event_type: Type[T],
        waiter_event: Event | None = None,
        waiter_id: str | None = None,
        requirements: dict[str, Any] | None = None,
        timeout: float | None = 2000,
    ) -> T:
        """Wait for an event of the specified type."""
        step_ctx = self._get_step_ctx(fn="wait_for_event")

        collected_waiters = step_ctx.state.collected_waiters
        requirements = requirements or {}

        # Generate a unique key for the waiter
        event_str = f"{event_type.__module__}.{event_type.__name__}"
        requirements_str = str(requirements)
        waiter_id = waiter_id or f"waiter_{event_str}_{requirements_str}"

        waiter = next((w for w in collected_waiters if w.waiter_id == waiter_id), None)
        if waiter is None or waiter.resolved_event is None:
            raise WaitingForEvent(
                AddWaiter(
                    waiter_id=waiter_id,
                    requirements=requirements,
                    timeout=timeout,
                    event_type=event_type,
                    waiter_event=waiter_event,
                )
            )
        else:
            step_ctx.returns.return_values.append(DeleteWaiter(waiter_id=waiter_id))
            return cast(T, waiter.resolved_event)

    def write_event_to_stream(self, ev: Event | None) -> None:
        """Write an event to the published event stream."""
        if ev is not None:
            self._execute_task(self._internal_adapter.write_to_event_stream(ev))
