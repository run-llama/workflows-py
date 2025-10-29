# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from unittest import mock

import pytest

from workflows.context import Context
from workflows.errors import WorkflowRuntimeError
from workflows.handler import WorkflowHandler


@pytest.mark.asyncio
async def test_str() -> None:
    ctx = mock.MagicMock(spec=Context)
    h = WorkflowHandler(ctx=ctx)
    h.set_result([])
    assert str(h) == "[]"


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
