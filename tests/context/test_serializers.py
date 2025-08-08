# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import pytest

from workflows.context import Context
from workflows.errors import ContextSerdeError
from workflows.workflow import Workflow


def test_serialization_roundtrip(ctx: Context, workflow: Workflow) -> None:
    assert Context.from_dict(workflow, ctx.to_dict())


def test_old_serialization(ctx: Context, workflow: Workflow) -> None:
    old_payload = {
        "globals": {},
        "streaming_queue": "[]",
        "queues": {"test_id": "[]"},
        "events_buffer": {},
        "in_progress": {},
        "accepted_events": [],
        "broker_log": [],
        "waiter_id": "test_id",
        "is_running": False,
    }
    with pytest.raises(ContextSerdeError):
        Context.from_dict(workflow, old_payload)
