from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterator

import pytest
import yaml
from fastapi.testclient import TestClient
from llama_agents.appserver import app as app_mod
from llama_agents.appserver.app import app
from llama_agents.appserver.deployment_config_parser import get_deployment_config
from llama_agents.appserver.settings import settings


@pytest.fixture(autouse=True)
def reset_config_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    # Point rc_path to a temp dir by default and clear cached config between tests
    monkeypatch.setenv("LLAMA_DEPLOY_APISERVER_APP_ROOT", str(tmp_path))
    settings.app_root = tmp_path
    get_deployment_config.cache_clear()
    yield
    # cleanup env
    os.environ.pop("LLAMA_DEPLOY_APISERVER_APP_ROOT", None)


@pytest.fixture
def http_client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def write_yaml(tmp_path: Path) -> Iterator[Callable[[str, str], Path]]:
    def _write(name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content)
        return p

    yield _write


@pytest.fixture(autouse=True)
def no_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prevent opening a real browser window during tests by stubbing the helper
    monkeypatch.setattr(
        app_mod, "open_browser_async", lambda *a, **k: None, raising=True
    )


@pytest.fixture
def make_deployment_file(tmp_path: Path) -> Callable[[str, bool], Path]:
    def _make(name: str = "myapp", with_ui: bool = False) -> Path:
        content = {
            "name": name,
            "services": {},
        }
        if with_ui:
            content["ui"] = {
                "source": {
                    "location": "ui",
                },
                "proxy_port": 3000,
            }
        path = tmp_path / "llama_deploy.yaml"
        path.write_text(yaml.dump(content))
        return path

    return _make


@pytest.fixture
def process_stub() -> object:
    class _Proc:
        def __init__(self) -> None:
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

    return _Proc()


@pytest.fixture
def proc_with_poll_wait() -> object:
    class _Proc:
        def __init__(self) -> None:
            self._ret = 0

        def wait(self) -> int:
            return self._ret

        def poll(self) -> int:
            return self._ret

    return _Proc()
