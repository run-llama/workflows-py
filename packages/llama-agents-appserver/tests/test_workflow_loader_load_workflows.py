from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from llama_agents.appserver.workflow_loader import load_workflows
from llama_agents.core.deployment_config import DeploymentConfig
from llama_agents.server import WorkflowServer


def test_load_workflows_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create a simple module file under rc_path
    pkg_dir = tmp_path / "src"
    pkg_dir.mkdir()
    (pkg_dir / "m.py").write_text("x = 1\n")

    cfg = DeploymentConfig(
        name="n",
        workflows={"svc": "m:workflow"},
    )

    # Mock importlib to confirm the module name used is derived from file name
    with mock.patch("llama_agents.appserver.workflow_loader.importlib") as imp:
        imp.import_module.return_value = mock.MagicMock(workflow=object())
        load_workflows(cfg)
        # import_module called with the module filename
        assert imp.import_module.call_args[0][0] in ("m",)


def test_load_workflows_from_workflow_server_app() -> None:
    """When config.app points to a WorkflowServer, load_workflows uses get_workflows()."""
    sentinel_wf = object()
    fake_server = mock.MagicMock(spec=WorkflowServer)
    fake_server.get_workflows.return_value = {"my_wf": sentinel_wf}

    cfg = DeploymentConfig(name="n", app="fake_mod:app")

    with mock.patch("llama_agents.appserver.workflow_loader.importlib") as imp:
        imp.import_module.return_value = mock.MagicMock(app=fake_server)
        result = load_workflows(cfg)

    assert result == {"my_wf": sentinel_wf}
    fake_server.get_workflows.assert_called_once()
