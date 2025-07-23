# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.


import pytest

from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent


class SimpleTestWorkflow(Workflow):
    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        message = await ctx.store.get("test_param", None)
        if message is None:
            message = getattr(ev, "message", "default")
        return StopEvent(result=f"processed: {message}")


class ErrorWorkflow(Workflow):
    @step
    async def error_step(self, ev: StartEvent) -> StopEvent:
        raise ValueError("Test error")


@pytest.fixture
def simple_test_workflow() -> Workflow:
    return SimpleTestWorkflow()


@pytest.fixture
def error_workflow() -> Workflow:
    return ErrorWorkflow()
