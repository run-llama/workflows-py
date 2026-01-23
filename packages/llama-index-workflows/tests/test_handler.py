# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from unittest import mock

import pytest
from workflows.errors import WorkflowRuntimeError
from workflows.handler import WorkflowHandler
from workflows.workflow import Workflow

from tests.runtime.conftest import MockRunAdapter


def _create_mock_handler() -> WorkflowHandler:
    """Create a WorkflowHandler with MockRunAdapter."""
    workflow = mock.MagicMock(spec=Workflow)
    workflow.workflow_name = "TestWorkflow"
    adapter = MockRunAdapter(run_id="test-run-id")
    return WorkflowHandler(workflow=workflow, external_adapter=adapter)


@pytest.mark.asyncio
async def test_str() -> None:
    h = _create_mock_handler()
    # The str representation shows workflow name, run_id, and result
    assert "TestWorkflow" in str(h)
    assert "test-run-id" in str(h)


@pytest.mark.asyncio
async def test_stream_events_consume_only_once() -> None:
    h = _create_mock_handler()
    h._all_events_consumed = True

    with pytest.raises(
        WorkflowRuntimeError,
        match="All the streamed events have already been consumed.",
    ):
        async for _ in h.stream_events():
            pass
