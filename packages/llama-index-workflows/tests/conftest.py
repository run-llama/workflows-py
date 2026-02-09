# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from typing import Any, AsyncGenerator

import pytest
from pydantic import Field
from workflows.context import Context
from workflows.context.external_context import ExternalContext
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.plugins.basic import (
    BasicRuntime,
)
from workflows.runtime.types.internal_state import BrokerState
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
async def ctx(workflow: Workflow) -> AsyncGenerator[Context[Any], None]:
    runtime = BasicRuntime()

    _ = runtime._get_or_create_queues(
        run_id="test-run",
        init_state=BrokerState.from_workflow(workflow),
    )
    ctx = Context._create_external(
        workflow=workflow,
        external_adapter=runtime.get_external_adapter("test-run"),
    )
    assert isinstance(ctx._face, ExternalContext)
    try:
        yield ctx
    finally:
        await ctx._face.shutdown()
