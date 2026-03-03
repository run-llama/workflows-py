# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Generator, Optional

import pytest
from llama_index_instrumentation import get_dispatcher
from llama_index_instrumentation.base import BaseEvent
from llama_index_instrumentation.event_handlers import BaseEventHandler
from llama_index_instrumentation.span import BaseSpan
from llama_index_instrumentation.span_handlers import BaseSpanHandler
from pydantic import PrivateAttr
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.runtime.types.step_function import SpanCancelledEvent
from workflows.workflow import Workflow


class SpanTracker(BaseSpanHandler[BaseSpan]):
    """Track span lifecycle events for testing."""

    _exited_ids: list[str] = PrivateAttr(default_factory=list)
    _dropped_ids: list[tuple[str, Optional[BaseException]]] = PrivateAttr(
        default_factory=list
    )

    @classmethod
    def class_name(cls) -> str:
        return "SpanTracker"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[BaseSpan]:
        return BaseSpan(id_=id_, parent_id=parent_span_id, tags=tags or {})

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[BaseSpan]:
        span = self.open_spans.get(id_)
        self._exited_ids.append(id_)
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[BaseSpan]:
        span = self.open_spans.get(id_)
        self._dropped_ids.append((id_, err))
        return span


class EventTracker(BaseEventHandler):
    """Track dispatched events for testing."""

    _events: list[BaseEvent] = PrivateAttr(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "EventTracker"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        self._events.append(event)


@pytest.fixture
def span_tracker() -> Generator[SpanTracker, None, None]:
    tracker = SpanTracker()
    root = get_dispatcher()
    root.span_handlers.append(tracker)
    yield tracker
    root.span_handlers.remove(tracker)


@pytest.fixture
def event_tracker() -> Generator[EventTracker, None, None]:
    tracker = EventTracker()
    root = get_dispatcher()
    root.event_handlers.append(tracker)
    yield tracker
    root.event_handlers.remove(tracker)


class WaitEvent(Event):
    value: str


async def test_wait_for_event_does_not_produce_dropped_spans(
    span_tracker: SpanTracker,
) -> None:
    """A step using wait_for_event should exit cleanly, not drop with an error."""

    class WaitWorkflow(Workflow):
        @step
        async def wait_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            result = await ctx.wait_for_event(WaitEvent)
            return StopEvent(result=result.value)

    wf = WaitWorkflow()
    handler = wf.run()
    assert handler.ctx is not None

    handler.ctx.send_event(WaitEvent(value="hello"))

    result = await handler
    assert result == "hello"

    step_drops = [
        (id_, err) for id_, err in span_tracker._dropped_ids if "wait_step" in id_
    ]
    step_exits = [id_ for id_ in span_tracker._exited_ids if "wait_step" in id_]

    assert step_drops == [], (
        f"Expected no dropped spans for wait_step, got: {step_drops}"
    )
    # Exited twice: once for the WaitingForEvent invocation, once for the replay
    assert len(step_exits) >= 2, (
        f"Expected at least 2 span exits for wait_step, got {len(step_exits)}: {step_exits}"
    )


async def test_cancel_run_produces_exited_spans_not_dropped(
    span_tracker: SpanTracker,
    event_tracker: EventTracker,
) -> None:
    """Cancelling a workflow should exit spans cleanly (OK) and emit SpanCancelledEvents."""

    class SleepWorkflow(Workflow):
        @step
        async def sleep_step(self, ev: StartEvent) -> StopEvent:
            await asyncio.sleep(3600)
            return StopEvent(result="unreachable")

    wf = SleepWorkflow()
    handler = wf.run()

    await asyncio.sleep(0.05)
    await handler.cancel_run()
    try:
        await handler
    except Exception:
        pass

    step_drops = [
        (id_, err) for id_, err in span_tracker._dropped_ids if "sleep_step" in id_
    ]
    step_exits = [id_ for id_ in span_tracker._exited_ids if "sleep_step" in id_]

    assert step_drops == [], (
        f"Expected no dropped spans for sleep_step, got: {step_drops}"
    )
    assert len(step_exits) == 1, (
        f"Expected 1 span exit for sleep_step, got {len(step_exits)}: {step_exits}"
    )

    # run_workflow span should also be exited, not dropped
    run_drops = [
        (id_, err)
        for id_, err in span_tracker._dropped_ids
        if "SleepWorkflow.run" in id_
    ]
    run_exits = [id_ for id_ in span_tracker._exited_ids if "SleepWorkflow.run" in id_]
    assert run_drops == [], (
        f"Expected no dropped spans for run_workflow, got: {run_drops}"
    )
    # 2 exits: the inner run_workflow span + the outer Workflow.run() span
    assert len(run_exits) == 2, (
        f"Expected 2 span exits for run_workflow, got {len(run_exits)}: {run_exits}"
    )

    # SpanCancelledEvents should have been emitted
    cancel_events = [
        e for e in event_tracker._events if isinstance(e, SpanCancelledEvent)
    ]
    assert len(cancel_events) >= 1, (
        f"Expected at least 1 SpanCancelledEvent, got {len(cancel_events)}"
    )
