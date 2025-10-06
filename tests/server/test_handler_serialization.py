# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from workflows.events import StopEvent, Event
from workflows.handler import WorkflowHandler
from workflows.server.server import _WorkflowHandler


class MyStopEvent(StopEvent):
    message: str


@pytest.mark.asyncio
async def test__workflow_handler_to_dict_json_roundtrip() -> None:
    handler: WorkflowHandler = WorkflowHandler()
    handler.set_result(MyStopEvent(message="ok"))

    queue: asyncio.Queue[Event] = asyncio.Queue()

    async def noop() -> None:
        return None

    task: asyncio.Task[None] = asyncio.create_task(noop())
    # Ensure the task is completed so the wrapper mimics a finished handler
    await asyncio.sleep(0)

    now = datetime.now(timezone.utc)
    wrapper = _WorkflowHandler(
        run_handler=handler,
        queue=queue,
        task=task,
        consumer_mutex=asyncio.Lock(),
        handler_id="handler-1",
        workflow_name="wf",
        started_at=now,
        updated_at=now,
        completed_at=now,
        handler_metadata={"test": "metadata"},
    )

    d = wrapper.to_dict()
    # JSON serialization should not error
    s = json.dumps(d)
    reparsed = json.loads(s)

    # Round-trip consistency
    assert reparsed == d
