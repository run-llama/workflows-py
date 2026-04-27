import os
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from llama_agents.core.schema.deployments import DeploymentResponse

"""Test session configuration for isolating per-worker state.

Each xdist worker gets a unique HOME and LLAMACTL_CONFIG_DIR so migrations and
SQLite DBs do not clash across processes.
"""

# Base temp dir for the whole session
_TEST_HOME = tempfile.mkdtemp(prefix="llamactl_test_home_")

# Worker-specific subdir (e.g. gw0, gw1). Falls back to 'main' without xdist.
_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "main")
_WORKER_HOME = os.path.join(_TEST_HOME, _WORKER_ID)
_WORKER_CONFIG = os.path.join(_WORKER_HOME, ".config", "llamactl")

# Ensure directories exist
os.makedirs(_WORKER_CONFIG, exist_ok=True)

# Isolate HOME and config per worker
os.environ["HOME"] = _WORKER_HOME
os.environ.setdefault("TERM", "xterm")
os.environ["LLAMACTL_CONFIG_DIR"] = _WORKER_CONFIG


def pytest_sessionfinish(session: pytest.Session, exitstatus: pytest.ExitCode) -> None:
    try:
        shutil.rmtree(_TEST_HOME, ignore_errors=True)
    except Exception:
        pass


def make_deployment(
    deployment_id: str = "my-app", **overrides: Any
) -> DeploymentResponse:
    """Build a DeploymentResponse with sensible defaults for command tests."""
    base: dict[str, Any] = {
        "id": deployment_id,
        "display_name": deployment_id,
        "repo_url": "https://github.com/example/repo",
        "deployment_file_path": "llama_deploy.yaml",
        "git_ref": "main",
        "git_sha": "abc1234567890",
        "project_id": "proj_default",
        "secret_names": [],
        "apiserver_url": None,
        "status": "Running",
    }
    base.update(overrides)
    return DeploymentResponse.model_validate(base)


def patch_project_client(client_mock: MagicMock) -> Any:
    """Patch ProjectClient construction inside ``cli.client.get_project_client``."""
    return patch(
        "llama_agents.core.client.manage_client.ProjectClient",
        return_value=client_mock,
    )


@pytest.fixture
def fake_profile() -> SimpleNamespace:
    return SimpleNamespace(
        api_url="http://test:8011",
        project_id="proj_default",
        api_key="key",
        device_oidc=None,
        name="prof",
    )


@pytest.fixture
def patched_auth(fake_profile: SimpleNamespace) -> Any:
    """Patch the env service so commands use a fake authenticated profile."""
    with patch("llama_agents.cli.config.env_service.service") as mock_service:
        mock_auth_svc = MagicMock()
        mock_auth_svc.get_current_profile.return_value = fake_profile
        mock_auth_svc.list_profiles.return_value = [fake_profile]
        mock_auth_svc.env = SimpleNamespace(requires_auth=True)
        mock_auth_svc.auth_middleware.return_value = None
        mock_service.current_auth_service.return_value = mock_auth_svc
        yield mock_service
