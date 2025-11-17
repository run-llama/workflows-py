# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from unittest.mock import MagicMock
from workflows.context import Context
from workflows.server.memory_workflow_store import MemoryWorkflowStore
from workflows.events import StopEvent, Event
from workflows.handler import WorkflowHandler
from workflows.protocol import HandlerData
from workflows.server.server import _WorkflowHandler


class MyStopEvent(StopEvent):
    message: str


@pytest.mark.asyncio
async def test__workflow_handler_to_dict_json_roundtrip() -> None:
    ctx = MagicMock(spec=Context)
    handler: WorkflowHandler = WorkflowHandler(ctx=ctx)
    handler.set_result(MyStopEvent(message="ok"))

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
