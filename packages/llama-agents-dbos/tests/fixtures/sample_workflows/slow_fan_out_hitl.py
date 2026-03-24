# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Fan-out HITL workflow for orphan-purge tests.

Fans out to 5 concurrent workers per round. Used to verify that orphaned
DBOS operation_outputs rows are purged on recovery.
"""

from __future__ import annotations

import asyncio
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


class SlowFanOutTickEvent(InputRequiredEvent):
    round: int = Field(default=0)


class SlowFanOutContinueEvent(HumanResponseEvent):
    pass


class SlowWorkRequestEvent(Event):
    worker_name: str = Field(default="")
    round: int = Field(default=0)


class SlowWorkResultEvent(Event):
    worker_name: str = Field(default="")
    round: int = Field(default=0)
    value: str = Field(default="")


class SlowFanOutWorkflow(Workflow):
    """Fan-out workflow with concurrent workers for orphan-purge testing."""

    def __init__(self, num_rounds: int = 4, **kwargs: Any) -> None:
        super().__init__(timeout=None, **kwargs)
        self._num_rounds = num_rounds

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> SlowWorkRequestEvent:
        print("STEP:start:complete", flush=True)
        for name in WORKER_NAMES[1:]:
            ctx.send_event(SlowWorkRequestEvent(worker_name=name, round=0))
        return SlowWorkRequestEvent(worker_name=WORKER_NAMES[0], round=0)

    @step
    async def worker_alpha(
        self, ctx: Context, ev: SlowWorkRequestEvent
    ) -> SlowWorkResultEvent | None:
        if ev.worker_name != "alpha":
            return None
        await asyncio.sleep(0.01)
        print(f"STEP:worker_alpha:round={ev.round}", flush=True)
        return SlowWorkResultEvent(
            worker_name="alpha", round=ev.round, value=f"alpha-r{ev.round}"
        )

    @step
    async def worker_beta(
        self, ctx: Context, ev: SlowWorkRequestEvent
    ) -> SlowWorkResultEvent | None:
        if ev.worker_name != "beta":
            return None
        await asyncio.sleep(0.01)
        print(f"STEP:worker_beta:round={ev.round}", flush=True)
        return SlowWorkResultEvent(
            worker_name="beta", round=ev.round, value=f"beta-r{ev.round}"
        )

    @step
    async def worker_gamma(
        self, ctx: Context, ev: SlowWorkRequestEvent
    ) -> SlowWorkResultEvent | None:
        if ev.worker_name != "gamma":
            return None
        await asyncio.sleep(0.01)
        print(f"STEP:worker_gamma:round={ev.round}", flush=True)
        return SlowWorkResultEvent(
            worker_name="gamma", round=ev.round, value=f"gamma-r{ev.round}"
        )

    @step
    async def worker_delta(
        self, ctx: Context, ev: SlowWorkRequestEvent
    ) -> SlowWorkResultEvent | None:
        if ev.worker_name != "delta":
            return None
        await asyncio.sleep(0.01)
        print(f"STEP:worker_delta:round={ev.round}", flush=True)
        return SlowWorkResultEvent(
            worker_name="delta", round=ev.round, value=f"delta-r{ev.round}"
        )

    @step
    async def worker_epsilon(
        self, ctx: Context, ev: SlowWorkRequestEvent
    ) -> SlowWorkResultEvent | None:
        if ev.worker_name != "epsilon":
            return None
        await asyncio.sleep(0.01)
        print(f"STEP:worker_epsilon:round={ev.round}", flush=True)
        return SlowWorkResultEvent(
            worker_name="epsilon", round=ev.round, value=f"epsilon-r{ev.round}"
        )

    @step
    async def collect(
        self, ctx: Context, ev: SlowWorkResultEvent
    ) -> SlowFanOutTickEvent | StopEvent | None:
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
        if ev.round + 1 >= self._num_rounds:
            print(f"STEP:collect:all_rounds_complete:{ev.round + 1}", flush=True)
            return StopEvent(result={"rounds": ev.round + 1})
        return SlowFanOutTickEvent(round=ev.round)

    @step
    async def on_continue(
        self, ctx: Context, ev: SlowFanOutContinueEvent
    ) -> SlowWorkRequestEvent:
        current_round: int = await ctx.store.get("next_round", default=0)
        next_round = current_round + 1
        await ctx.store.set("next_round", next_round)
        print(f"STEP:on_continue:round={next_round}", flush=True)
        for name in WORKER_NAMES[1:]:
            ctx.send_event(SlowWorkRequestEvent(worker_name=name, round=next_round))
        return SlowWorkRequestEvent(worker_name=WORKER_NAMES[0], round=next_round)
