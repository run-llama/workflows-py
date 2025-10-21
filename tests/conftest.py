# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from typing import AsyncGenerator
import pytest
from pydantic import Field

from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.workflow import Workflow


class OneTestEvent(Event):
    test_param: str = Field(default="test")


class AnotherTestEvent(Event):
    another_test_param: str = Field(default="another_test")


class LastEvent(Event):
    pass


class DummyWorkflow(Workflow):
    @step
    async def start_step(self, ev: StartEvent) -> OneTestEvent:
        return OneTestEvent()

    @step
    async def middle_step(self, ev: OneTestEvent) -> LastEvent:
        return LastEvent()

    @step
    async def end_step(self, ev: LastEvent) -> StopEvent:
        return StopEvent(result="Workflow completed")


@pytest.fixture()
def workflow() -> Workflow:
    return DummyWorkflow()


@pytest.fixture()
def events() -> list:
    return [OneTestEvent, AnotherTestEvent]


@pytest.fixture()
async def ctx(workflow: Workflow) -> AsyncGenerator[Context, None]:
    ctx = Context(workflow=workflow)
    broker = ctx._init_broker(workflow)
    try:
        yield ctx
    finally:
        await broker.shutdown()
