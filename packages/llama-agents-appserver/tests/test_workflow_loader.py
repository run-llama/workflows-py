from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
from llama_agents.appserver.settings import ApiserverSettings
from llama_agents.appserver.workflow_loader import (
    build_ui,
    load_environment_variables,
    load_workflows,
)
from llama_agents.core.deployment_config import DeploymentConfig, UIConfig


def test_load_workflows_imports(tmp_path: Path) -> None:
    cfg = DeploymentConfig(
        name="n",
        workflows={"svc": "m:mywf"},
    )

    fake_wf = object()
    with mock.patch("llama_agents.appserver.workflow_loader.importlib") as imp:
        imp.import_module.return_value = mock.MagicMock(mywf=fake_wf)
        m = load_workflows(cfg)
    assert m["svc"] is fake_wf


def test_load_environment_variables_merges_env_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("API_KEY=123\n")
    cfg = DeploymentConfig(
        name="n",
        workflows={"svc": "m:mywf"},
        env={"FOO": "bar"},
        env_files=[".env"],
    )
    load_environment_variables(cfg, tmp_path)
    assert os.environ.get("FOO") == "bar"
    assert os.environ.get("API_KEY") == "123"


def test_build_ui_sets_env_and_calls_pnpm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui_root = tmp_path / "ui"
    ui_root.mkdir(parents=True)
    package_json = ui_root / "package.json"
    package_json.write_text('{"scripts": {"build": "echo build"} }')
    cfg = DeploymentConfig(name="n", ui=UIConfig(directory="ui"))
    # Configure UI proxy port via env (used by settings passed to build_ui)
    monkeypatch.setenv("LLAMA_DEPLOY_APISERVER_PROXY_UI_PORT", "4503")
    api_settings = ApiserverSettings()
    # Mock process runner used inside build_ui
    with (
        mock.patch("llama_agents.appserver.workflow_loader.run_process") as run,
    ):
        run.side_effect = lambda *a, **k: None
        build_ui(tmp_path, cfg, api_settings)

        # Validate call
        call = run.call_args
        assert call is not None
        args, kwargs = call
        # first positional arg is the cmd list
        assert args[0][:3] == ["npm", "run", "build"]
        # cwd and env are kwargs
        env = kwargs["env"]
        assert kwargs["cwd"] == ui_root
        assert env["LLAMA_DEPLOY_DEPLOYMENT_URL_ID"] == "n"
        assert env["LLAMA_DEPLOY_DEPLOYMENT_NAME"] == "n"
        assert env["LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH"] == "/deployments/n/ui"
        assert env["PORT"] == "4503"
