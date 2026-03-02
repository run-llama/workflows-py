# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import inspect
from typing import Any, Generator, Optional

import pytest
from llama_index_instrumentation import get_dispatcher
from llama_index_instrumentation.span import BaseSpan
from llama_index_instrumentation.span_handlers import BaseSpanHandler
from pydantic import PrivateAttr
from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
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


@pytest.fixture
def span_tracker() -> Generator[SpanTracker, None, None]:
    tracker = SpanTracker()
    root = get_dispatcher()
    root.span_handlers.append(tracker)
    yield tracker
    root.span_handlers.remove(tracker)


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
