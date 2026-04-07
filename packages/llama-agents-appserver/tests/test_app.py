from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from llama_agents.appserver.app import app
from llama_agents.appserver.deployment_config_parser import get_deployment_config
from llama_agents.appserver.settings import settings


@pytest.fixture(autouse=True)
def _reset_deployment_config() -> None:
    get_deployment_config.cache_clear()


def _write_deployment(
    tmp_path: Path, name: str = "myapp", with_ui: bool = True
) -> None:
    content = {
        "name": name,
        "workflows": {},
    }
    if with_ui:
        content["ui"] = {
            "directory": "ui",
            "proxy_port": 3010,
        }
    (tmp_path / "llama_deploy.yaml").write_text(yaml.dump(content))


def test_root_redirect_and_metrics(tmp_path: Path) -> None:
    settings.app_root = tmp_path
    settings.deployment_file_path = Path("llama_deploy.yaml")
    _write_deployment(tmp_path)

    with TestClient(app) as client:
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (302, 307)
        # target should be deployments/{name}/ui
        assert "/deployments/myapp/" in r.headers.get("location", "")

        # metrics exposed
        m = client.get("/metrics")
        assert m.status_code == 200
        assert b"apiserver_state" in m.content

        # CORS header present when Origin is sent
        r2 = client.get("/health", headers={"Origin": "http://example.com"})
        assert r2.headers.get("access-control-allow-origin") == "*"


def test_static_ui_mount_serves_dist(tmp_path: Path) -> None:
    # create dist assets
    ui_dist = tmp_path / "ui" / "dist"
    ui_dist.mkdir(parents=True)
    (ui_dist / "index.html").write_text("<html>ok</html>")

    settings.app_root = tmp_path
    settings.deployment_file_path = Path("llama_deploy.yaml")
    settings.proxy_ui = False
    _write_deployment(tmp_path)

    with TestClient(app) as client:
        r = client.get("/deployments/myapp/ui/index.html")
        assert r.status_code == 200
        assert r.text.strip() == "<html>ok</html>"


def test_workflowserver_endpoint_available(tmp_path: Path) -> None:
    # Real mount with no workflows configured should still expose the index endpoint
    settings.app_root = tmp_path
    settings.deployment_file_path = Path("llama_deploy.yaml")
    (tmp_path / "llama_deploy.yaml").write_text("name: myapp\nservices: {}\n")

    with TestClient(app) as client:
        # Starlette WorkflowServer index
        r = client.get("/deployments/myapp/workflows")
        assert r.status_code == 200
        _ = r.json()

        # Legacy FastAPI sessions endpoint (should return empty list when no sessions)
        r2 = client.get("/deployments/myapp/sessions")
        assert r2.status_code == 200
        assert r2.json() == []
