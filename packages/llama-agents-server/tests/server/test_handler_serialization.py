# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from llama_agents.client.protocol import HandlerData
from llama_agents.server import PersistentHandler
from llama_agents.server._service import handler_data_from_persistent
from workflows.events import StopEvent


class MyStopEvent(StopEvent):
    message: str


@pytest.mark.asyncio
async def test_handler_data_from_persistent_json_roundtrip() -> None:
    now = datetime.now(timezone.utc)
    stop_event = MyStopEvent(message="ok")

    persistent = PersistentHandler(
        handler_id="handler-1",
        workflow_name="wf",
        run_id="test-run-id",
        status="completed",
        started_at=now,
        updated_at=now,
        completed_at=now,
        result=stop_event,
    )

    response_model = handler_data_from_persistent(persistent)
    # JSON serialization should not error
    s = json.dumps(response_model.model_dump())
    reparsed_dict = json.loads(s)
    reparsed = HandlerData.model_validate(reparsed_dict)

    # Round-trip consistency
    assert reparsed == response_model
