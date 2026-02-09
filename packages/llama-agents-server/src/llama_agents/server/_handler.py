# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Awaitable, Callable

from llama_agents.client.protocol import HandlerData
from llama_agents.client.protocol.serializable_events import (
    EventEnvelopeWithMetadata,
)
from llama_index_instrumentation.dispatcher import instrument_tags
from workflows.errors import WorkflowRuntimeError
from workflows.events import (
    Event,
    StepState,
    StepStateChanged,
    StopEvent,
    UnhandledEvent,
    WorkflowIdleEvent,
)
from workflows.handler import WorkflowHandler
from workflows.workflow import Workflow

from ._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    PersistentHandler,
    Status,
)

logger = logging.getLogger()


@dataclass
class _WorkflowHandler:
    """A wrapper around a handler: WorkflowHandler. Necessary to monitor and dispatch events from the handler's stream_events"""

    run_handler: WorkflowHandler
    queue: asyncio.Queue[Event]
    task: asyncio.Task[None] | None
    # only one consumer of the queue at a time allowed
    consumer_mutex: asyncio.Lock

    # metadata
    handler_id: str
    workflow_name: str
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    # Dependencies for persistence
    _workflow_store: AbstractWorkflowStore
    _persistence_backoff: list[float]
    _on_finish: Callable[[], Awaitable[None]] | None = None
    idle_since: datetime | None = None

    # Idle release support
    _idle_release_timeout: timedelta | None = None
    _on_idle_release: Callable[[_WorkflowHandler], Awaitable[None]] | None = None
    _idle_release_timer: asyncio.Task[None] | None = None
    _skip_checkpoint: bool = False  # Set to prevent checkpointing stale handlers

    def _as_persistent(self) -> PersistentHandler:
        """Persist the current handler state immediately to the workflow store."""
        self.updated_at = datetime.now(timezone.utc)
        if self.status in ("completed", "failed", "cancelled"):
            self.completed_at = self.updated_at

        persistent = PersistentHandler(
            handler_id=self.handler_id,
            workflow_name=self.workflow_name,
            status=self.status,
            run_id=self.run_handler.run_id,
            error=self.error,
            result=self.result,
            started_at=self.started_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
            idle_since=self.idle_since,
            ctx=self.run_handler.ctx.to_dict() if self.run_handler.ctx else {},
        )
        return persistent

    async def persist(self, persistent: PersistentHandler) -> None:
        await self._workflow_store.update(persistent)

    async def checkpoint(self) -> None:
        """Persist with retry/backoff; cancel handler when retries exhausted."""
        if self._skip_checkpoint:
            logger.debug(f"Skipping checkpoint for handler {self.handler_id}")
            return
        backoffs = list(self._persistence_backoff)
        try:
            persistent = self._as_persistent()
        except Exception as e:
            logger.error(
                f"Failed to checkpoint handler {self.handler_id} to persistent state. Is there non-serializable state in an event or the state store? {e}",
                exc_info=True,
            )
            raise
        while True:
            try:
                await self.persist(persistent)
                return
            except Exception as e:
                backoff = backoffs.pop(0) if backoffs else None
                if backoff is None:
                    logger.error(
                        f"Failed to checkpoint handler {self.handler_id} after final attempt. Failing the handler.",
                        exc_info=True,
                    )
                    # Cancel the underlying workflow; do not re-raise here to allow callers to decide behavior
                    try:
                        self.run_handler.cancel()
                    except Exception:
                        pass
                    raise
                logger.error(
                    f"Failed to checkpoint handler {self.handler_id}. Retrying in {backoff} seconds: {e}"
                )
                await asyncio.sleep(backoff)

    def to_response_model(self) -> HandlerData:
        """Convert runtime handler to API response model."""
        return HandlerData(
            handler_id=self.handler_id,
            workflow_name=self.workflow_name,
            run_id=self.run_handler.run_id,
            status=self.status,
            started_at=self.started_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            completed_at=self.completed_at.isoformat()
            if self.completed_at is not None
            else None,
            error=self.error,
            result=EventEnvelopeWithMetadata.from_event(self.result)
            if self.result is not None
            else None,
        )

    @staticmethod
    def handler_data_from_persistent(persistent: PersistentHandler) -> HandlerData:
        return HandlerData(
            handler_id=persistent.handler_id,
            workflow_name=persistent.workflow_name,
            run_id=persistent.run_id,
            status=persistent.status,
            started_at=persistent.started_at.isoformat()
            if persistent.started_at is not None
            else datetime.now(timezone.utc).isoformat(),
            updated_at=persistent.updated_at.isoformat()
            if persistent.updated_at is not None
            else None,
            completed_at=persistent.completed_at.isoformat()
            if persistent.completed_at is not None
            else None,
            error=persistent.error,
            result=EventEnvelopeWithMetadata.from_event(persistent.result)
            if persistent.result is not None
            else None,
        )

    @property
    def status(self) -> Status:
        """Get the current status by inspecting the handler state."""
        if not self.run_handler.done():
            return "running"
        # done - check if cancelled first
        if self.run_handler.cancelled():
            return "cancelled"
        # then check for exception
        exc = self.run_handler.exception()
        if exc is not None:
            return "failed"
        return "completed"

    @property
    def error(self) -> str | None:
        if not self.run_handler.done():
            return None
        try:
            exc = self.run_handler.exception()
        except asyncio.CancelledError:
            return None
        return str(exc) if exc is not None else None

    @property
    def result(self) -> StopEvent | None:
        if not self.run_handler.done():
            return None
        try:
            return self.run_handler.get_stop_event()
        except asyncio.CancelledError:
            return None
        except Exception:
            return None

    def _start_idle_release_timer(self) -> None:
        """Start a timer to release this handler after the idle timeout."""
        if self._idle_release_timeout is None or self._on_idle_release is None:
            return

        # Cancel any existing timer first
        self._cancel_idle_release_timer()

        timeout_seconds = self._idle_release_timeout.total_seconds()

        async def release_after_timeout() -> None:
            try:
                await asyncio.sleep(timeout_seconds)
                # Only release if still idle and no active stream consumers
                if self.idle_since is not None:
                    if self.consumer_mutex.locked():
                        # Mutex is locked - reschedule to try again later
                        self._start_idle_release_timer()
                        return
                    if self._on_idle_release is not None:
                        await self._on_idle_release(self)
            except asyncio.CancelledError:
                pass  # Timer was cancelled, nothing to do

        self._idle_release_timer = asyncio.create_task(release_after_timeout())

    def _cancel_idle_release_timer(self, skip_checkpoint: bool = False) -> None:
        """Cancel any pending idle release timer."""
        if skip_checkpoint:
            self._skip_checkpoint = True
        if self._idle_release_timer is not None:
            self._idle_release_timer.cancel()
            self._idle_release_timer = None

    def mark_idle(self, idle_since: datetime | None = None) -> None:
        self.idle_since = idle_since or datetime.now(timezone.utc)
        self._start_idle_release_timer()

    def mark_active(self) -> None:
        """Mark this handler as active (not idle).

        Call this when an event is being sent to prevent premature release.
        """
        if self.idle_since is not None:
            self.idle_since = None
            self._cancel_idle_release_timer()

    def start_streaming(self, on_finish: Callable[[], Awaitable[None]]) -> None:
        """Start streaming events from the handler and managing state."""
        self.task = asyncio.create_task(self._stream_events(on_finish=on_finish))

    async def _stream_events(self, on_finish: Callable[[], Awaitable[None]]) -> None:
        """Internal method that streams events, updates status, and persists state."""
        with instrument_tags({"handler_id": self.handler_id}):
            await self.checkpoint()
            self._on_finish = on_finish
            try:
                async for event in self.run_handler.stream_events(expose_internal=True):
                    # Track idle state transitions and manage release timer
                    if isinstance(event, WorkflowIdleEvent):
                        self.mark_idle()
                    elif isinstance(event, UnhandledEvent):
                        self.mark_idle()
                    elif (
                        isinstance(event, StepStateChanged)
                        and event.step_state == StepState.RUNNING
                    ):
                        self.mark_active()

                    if (  # Watch for a specific internal event that signals the step is complete
                        isinstance(event, StepStateChanged)
                        and event.step_state == StepState.NOT_RUNNING
                    ):
                        state = (
                            self.run_handler.ctx.to_dict()
                            if self.run_handler.ctx
                            else None
                        )
                        if state is None:
                            logger.warning(
                                f"Context state is None for handler {self.handler_id}. This is not expected."
                            )
                            continue
                        await self.checkpoint()

                    self.queue.put_nowait(event)
            except WorkflowRuntimeError:
                # Stream was already consumed - this can happen during handler
                # cancellation when run_handler is cancelled before this task.
                # This is benign; we'll proceed to cleanup.
                pass

            # Workflow is completing - cancel any pending release timer
            self._cancel_idle_release_timer()

            # done when stream events are complete
            try:
                await self.run_handler
            except asyncio.CancelledError:
                # Handler was cancelled - status will be automatically detected via handler.cancelled()
                logger.info(f"Workflow run {self.handler_id} was cancelled")
                # Don't re-raise, just let the task complete
            except Exception as e:
                logger.error(
                    f"Workflow run {self.handler_id} failed! {e}", exc_info=True
                )

            await self.checkpoint()

    async def acquire_events_stream(
        self, timeout: float = 1
    ) -> AsyncGenerator[Event, None]:
        """
        Acquires the lock to iterate over the events, and returns generator of events.
        """
        try:
            await asyncio.wait_for(self.consumer_mutex.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise NoLockAvailable(
                f"No lock available to acquire after {timeout}s timeout"
            )
        return self._iter_events(timeout=timeout)

    async def _iter_events(self, timeout: float = 1) -> AsyncGenerator[Event, None]:
        """
        Converts the queue to an async generator while the workflow is still running, and there are still events.
        For better or worse, multiple consumers will compete for events
        """
        queue_get_task: asyncio.Task[Event] | None = None

        try:
            while not self.queue.empty() or (
                self.task is not None and not self.task.done()
            ):
                available_events = []
                while not self.queue.empty():
                    available_events.append(self.queue.get_nowait())
                for event in available_events:
                    yield event
                queue_get_task = asyncio.create_task(self.queue.get())
                task_waitable = self.task
                done, pending = await asyncio.wait(
                    {queue_get_task, task_waitable}
                    if task_waitable is not None
                    else {queue_get_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queue_get_task in done:
                    yield await queue_get_task
                    queue_get_task = None
                else:  # otherwise task completed, so nothing else will be published to the queue
                    queue_get_task.cancel()
                    queue_get_task = None
                    break
        finally:
            # Cancel any pending queue.get() task to prevent orphaned tasks from
            # consuming events after the consumer disconnects.
            if queue_get_task is not None:
                if not queue_get_task.done():
                    queue_get_task.cancel()
                    try:
                        await queue_get_task
                    except asyncio.CancelledError:
                        pass

            if self._on_finish is not None and self.run_handler.done():
                # clean up the resources if the stream has been consumed
                await self._on_finish()
            self.consumer_mutex.release()

    async def cancel_handlers_and_tasks(self) -> None:
        """Cancel the handler and release it from the store."""
        if not self.run_handler.done():
            try:
                self.run_handler.cancel()
            except Exception:
                pass
            try:
                await self.run_handler.cancel_run()
            except Exception:
                pass
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


class NoLockAvailable(Exception):
    """Raised when no lock is available to acquire after a timeout"""

    pass


@dataclass
class _NamedWorkflow:
    name: str
    workflow: Workflow
