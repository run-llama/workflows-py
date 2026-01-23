# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from llama_agents.client.protocol import HandlerData
from llama_agents.server.memory_workflow_store import MemoryWorkflowStore
from llama_agents.server.server import _WorkflowHandler
from tests.runtime.conftest import MockRunAdapter
from workflows.events import Event, StopEvent
from workflows.handler import WorkflowHandler
from workflows.workflow import Workflow


class MyStopEvent(StopEvent):
    message: str


@pytest.mark.asyncio
async def test_workflow_handler_to_dict_json_roundtrip() -> None:
    workflow = MagicMock(spec=Workflow)
    workflow.workflow_name = "TestWorkflow"
    adapter = MockRunAdapter(run_id="test-run-id")
    # Set the result on the adapter
    stop_event = MyStopEvent(message="ok")
    adapter.set_result(stop_event)

    handler: WorkflowHandler = WorkflowHandler(
        workflow=workflow, external_adapter=adapter
    )
    # Wait for the result task to complete
    await asyncio.sleep(0)

    queue: asyncio.Queue[Event] = asyncio.Queue()

    async def noop() -> None:
        return None

    task: asyncio.Task[None] = asyncio.create_task(noop())
    # Ensure the task is completed so the wrapper mimics a finished handler
    await asyncio.sleep(0)

    now = datetime.now(timezone.utc)
    wrapper = _WorkflowHandler(
        _workflow_store=MemoryWorkflowStore(),
        _persistence_backoff=[0.0, 0.0],
        run_handler=handler,
        queue=queue,
        task=task,
        consumer_mutex=asyncio.Lock(),
        handler_id="handler-1",
        workflow_name="wf",
        started_at=now,
        updated_at=now,
        completed_at=now,
    )

    response_model = wrapper.to_response_model()
    # JSON serialization should not error
    s = json.dumps(response_model.model_dump())
    reparsed_dict = json.loads(s)
    reparsed = HandlerData.model_validate(reparsed_dict)

    # Round-trip consistency
    assert reparsed == response_model
