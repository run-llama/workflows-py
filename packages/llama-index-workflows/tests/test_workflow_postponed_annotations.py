# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import pytest

from workflows.decorators import step
from workflows.events import (
    StartEvent,
    StopEvent,
)
from workflows.workflow import Workflow
from workflows.testing import WorkflowTestRunner

from .conftest import OneTestEvent


class PostponedAnnotationsWorkflow(Workflow):
    @step
    async def step1(self, ev: StartEvent) -> OneTestEvent:
        return OneTestEvent(test_param="postponed")

    @step
    async def step2(self, ev: OneTestEvent) -> StopEvent:
        return StopEvent(result=f"Handled {ev.test_param}")


@pytest.mark.asyncio
async def test_workflow_postponed_annotations() -> None:
    r = await WorkflowTestRunner(PostponedAnnotationsWorkflow()).run()
    assert r.result == "Handled postponed"


@pytest.mark.asyncio
async def test_workflow_forward_reference() -> None:
    class ForwardRefWorkflow(Workflow):
        @step
        async def step1(self, ev: StartEvent) -> OneTestEvent:
            return OneTestEvent(test_param="forward")

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result=f"Handled {ev.test_param}")

    r = await WorkflowTestRunner(ForwardRefWorkflow()).run()
    assert r.result == "Handled forward"
