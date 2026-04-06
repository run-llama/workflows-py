import json
from pathlib import Path
from unittest import mock

import pytest
from llama_agents.appserver.deployment import Deployment
from workflows import Context, Workflow
from workflows.handler import WorkflowHandler


@pytest.fixture
def deployment(tmp_path: Path) -> Deployment:
    # minimal Deployment with no workflows yet
    return Deployment(workflows={})


@pytest.mark.asyncio
async def test_run_workflow_without_session_without_kwargs(tmp_path: Path) -> None:
    d = Deployment(workflows={})
    wf = mock.MagicMock(spec=Workflow)
    wf.run = mock.AsyncMock(return_value="ok")
    d._workflow_services = {"svc": wf}

    result = await d.run_workflow("svc")
    assert result == "ok"
    wf.run.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_run_workflow_with_session(tmp_path: Path) -> None:
    d = Deployment(workflows={})
    wf = mock.MagicMock(spec=Workflow)
    wf.run = mock.AsyncMock(return_value="ok2")
    ctx = mock.MagicMock(spec=Context)
    d._workflow_services = {"svc": wf}
    d._contexts = {"sid": ctx}

    result = await d.run_workflow("svc", session_id="sid", foo=1)
    assert result == "ok2"
    wf.run.assert_awaited_once_with(context=ctx, foo=1)


def test_run_workflow_no_wait_creates_session(tmp_path: Path) -> None:
    d = Deployment(workflows={})
    wf = mock.MagicMock(spec=Workflow)
    handler = mock.MagicMock(spec=WorkflowHandler)
    # Avoid constructing real Context; store a simple mock
    handler.ctx = mock.MagicMock(spec=Context)
    wf.run.return_value = handler
    d._workflow_services = {"svc": wf}

    with mock.patch("llama_agents.appserver.deployment.generate_id") as gen:
        gen.side_effect = ["sess1", "handler1"]
        hid, sid = d.run_workflow_no_wait("svc", None, foo="bar")

    assert hid == "handler1"
    assert sid == "sess1"
    assert d._handlers[hid] is handler
    assert json.loads(d._handler_inputs[hid]) == {"foo": "bar"}


def test_run_workflow_no_wait_with_session(tmp_path: Path) -> None:
    d = Deployment(workflows={})
    wf = mock.MagicMock(spec=Workflow)
    handler = mock.MagicMock(spec=WorkflowHandler)
    ctx = mock.MagicMock(spec=Context)
    wf.run.return_value = handler
    d._workflow_services = {"svc": wf}
    d._contexts = {"sid": ctx}

    with mock.patch(
        "llama_agents.appserver.deployment.generate_id", return_value="hid"
    ):
        hid, sid = d.run_workflow_no_wait("svc", "sid", a=1)

    assert hid == "hid"
    assert sid == "sid"
    assert d._handlers[hid] is handler
    wf.run.assert_called_once()
