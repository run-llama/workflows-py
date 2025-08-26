# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import pytest

from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.retry_policy import ConstantDelayRetryPolicy
from workflows.workflow import Workflow


@pytest.mark.asyncio
async def test_retry_e2e() -> None:
    class CountEvent(Event):
        """Empty event to signal a step to increment a counter in the Context."""

    class DummyWorkflow(Workflow):
        # Set a small delay to avoid impacting the CI speed too much
        @step(retry_policy=ConstantDelayRetryPolicy(delay=0.2))
        async def flaky_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            count = await ctx.store.get("counter", default=0)
            ctx.send_event(CountEvent())
            if count < 3:
                raise ValueError("Something bad happened!")
            return StopEvent(result="All good!")

        @step
        async def counter(self, ctx: Context, ev: CountEvent) -> None:
            count = await ctx.store.get("counter", default=0)
            await ctx.store.set("counter", count + 1)

    workflow = DummyWorkflow(disable_validation=True)
    assert await workflow.run() == "All good!"


def test_ConstantDelayRetryPolicy_init() -> None:
    p = ConstantDelayRetryPolicy()
    assert p.maximum_attempts == 3
    assert p.delay == 5


def test_ConstantDelayRetryPolicy_next() -> None:
    delay = 4.2
    p = ConstantDelayRetryPolicy(maximum_attempts=5, delay=delay)
    assert p.next(elapsed_time=0.0, attempts=4, error=Exception()) == delay
    assert p.next(elapsed_time=0.0, attempts=5, error=Exception()) is None
    # This should never happen but ensure the code is resilient
    assert p.next(elapsed_time=0.0, attempts=999, error=Exception()) is None
