# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Fan-out HITL workflow for determinism bug reproduction.

This workflow fans out to multiple concurrent worker steps on each round,
creating conditions where after journal exhaustion during recovery, many
DBOS step results are available instantly and asyncio.wait(FIRST_COMPLETED)
picks non-deterministically from the done set.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.workflow import Workflow

WORKER_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon"]
NUM_WORKERS = len(WORKER_NAMES)


class FanOutTickEvent(InputRequiredEvent):
    """Emitted after each round of fan-out work completes."""

    round: int = Field(default=0)


class FanOutContinueEvent(HumanResponseEvent):
    """Human response to continue to the next round."""

    pass


class WorkRequestEvent(Event):
    """Dispatched to a worker."""

    worker_name: str = Field(default="")
    round: int = Field(default=0)


class WorkResultEvent(Event):
    """Result from a worker."""

    worker_name: str = Field(default="")
    round: int = Field(default=0)
    value: str = Field(default="")


class FanOutWorkflow(Workflow):
    """Workflow that fans out to 5 concurrent workers per round.

    Each round:
    1. Dispatch step sends 5 WorkRequestEvents (one per worker type)
    2. 5 worker steps run concurrently as DBOS steps
    3. Collector gathers all 5 results, tracking per-round counts
    4. Emits FanOutTickEvent (HITL pause point)
    5. On FanOutContinueEvent, starts next round

    This creates the conditions for the determinism bug:
    - Many concurrent DBOS step fids per journal entry
    - After journal exhaustion, all workers complete instantly
    - asyncio.wait picks non-deterministically from done set
    """

    def __init__(self, num_rounds: int = 6, **kwargs: Any) -> None:
        super().__init__(timeout=None, **kwargs)
        self._num_rounds = num_rounds

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> WorkRequestEvent:
        """Initialize and dispatch first round of work."""
        print("STEP:start:complete", flush=True)
        for name in WORKER_NAMES[1:]:
            ctx.send_event(WorkRequestEvent(worker_name=name, round=0))
        return WorkRequestEvent(worker_name=WORKER_NAMES[0], round=0)

    @step
    async def worker_alpha(
        self, ctx: Context, ev: WorkRequestEvent
    ) -> WorkResultEvent | None:
        if ev.worker_name != "alpha":
            return None
        # Random sleep creates non-deterministic completion order during fresh
        # execution. During DBOS replay, the step result is returned instantly
        # from the DB (sleep is never reached), so all workers complete at once
        # and asyncio.wait picks non-deterministically from the done set.
        await asyncio.sleep(random.uniform(0.001, 0.02))
        print(f"STEP:worker_alpha:round={ev.round}", flush=True)
        return WorkResultEvent(
            worker_name="alpha", round=ev.round, value=f"alpha-r{ev.round}"
        )

    @step
    async def worker_beta(
        self, ctx: Context, ev: WorkRequestEvent
    ) -> WorkResultEvent | None:
        if ev.worker_name != "beta":
            return None
        await asyncio.sleep(random.uniform(0.001, 0.02))
        print(f"STEP:worker_beta:round={ev.round}", flush=True)
        return WorkResultEvent(
            worker_name="beta", round=ev.round, value=f"beta-r{ev.round}"
        )

    @step
    async def worker_gamma(
        self, ctx: Context, ev: WorkRequestEvent
    ) -> WorkResultEvent | None:
        if ev.worker_name != "gamma":
            return None
        await asyncio.sleep(random.uniform(0.001, 0.02))
        print(f"STEP:worker_gamma:round={ev.round}", flush=True)
        return WorkResultEvent(
            worker_name="gamma", round=ev.round, value=f"gamma-r{ev.round}"
        )

    @step
    async def worker_delta(
        self, ctx: Context, ev: WorkRequestEvent
    ) -> WorkResultEvent | None:
        if ev.worker_name != "delta":
            return None
        await asyncio.sleep(random.uniform(0.001, 0.02))
        print(f"STEP:worker_delta:round={ev.round}", flush=True)
        return WorkResultEvent(
            worker_name="delta", round=ev.round, value=f"delta-r{ev.round}"
        )

    @step
    async def worker_epsilon(
        self, ctx: Context, ev: WorkRequestEvent
    ) -> WorkResultEvent | None:
        if ev.worker_name != "epsilon":
            return None
        await asyncio.sleep(random.uniform(0.001, 0.02))
        print(f"STEP:worker_epsilon:round={ev.round}", flush=True)
        return WorkResultEvent(
            worker_name="epsilon", round=ev.round, value=f"epsilon-r{ev.round}"
        )

    @step
    async def collect(
        self, ctx: Context, ev: WorkResultEvent
    ) -> FanOutTickEvent | StopEvent | None:
        """Collect results from all workers, tracking per-round completion."""
        # Use per-round counters to handle concurrent rounds correctly
        key = f"round_{ev.round}_count"
        count: int = await ctx.store.get(key, default=0)
        count += 1
        await ctx.store.set(key, count)

        print(
            f"STEP:collect:{ev.worker_name}:round={ev.round}:{count}/{NUM_WORKERS}",
            flush=True,
        )

        if count < NUM_WORKERS:
            return None

        # All workers done for this round
        if ev.round + 1 >= self._num_rounds:
            print(f"STEP:collect:all_rounds_complete:{ev.round + 1}", flush=True)
            return StopEvent(result={"rounds": ev.round + 1})

        return FanOutTickEvent(round=ev.round)

    @step
    async def on_continue(
        self, ctx: Context, ev: FanOutContinueEvent
    ) -> WorkRequestEvent:
        """Handle continue event: start next round of workers."""
        current_round: int = await ctx.store.get("next_round", default=0)
        next_round = current_round + 1
        await ctx.store.set("next_round", next_round)
        print(f"STEP:on_continue:round={next_round}", flush=True)

        for name in WORKER_NAMES[1:]:
            ctx.send_event(WorkRequestEvent(worker_name=name, round=next_round))
        return WorkRequestEvent(worker_name=WORKER_NAMES[0], round=next_round)
