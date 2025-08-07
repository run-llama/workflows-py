# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from unittest import mock

import pytest

from workflows.context import Context
from workflows.errors import WorkflowRuntimeError
from workflows.handler import WorkflowHandler


def test_str() -> None:
    h = WorkflowHandler()
    h.set_result([])
    assert str(h) == "[]"


@pytest.mark.asyncio
async def test_stream_no_context() -> None:
    h = WorkflowHandler()
    with pytest.raises(ValueError, match="Context is not set!"):
        async for ev in h.stream_events():
            pass


@pytest.mark.asyncio
async def test_stream_events_consume_only_once() -> None:
    ctx = mock.MagicMock(spec=Context)

    h = WorkflowHandler(ctx=ctx)
    h._all_events_consumed = True

    with pytest.raises(
        WorkflowRuntimeError,
        match="All the streamed events have already been consumed.",
    ):
        async for _ in h.stream_events():
            pass
