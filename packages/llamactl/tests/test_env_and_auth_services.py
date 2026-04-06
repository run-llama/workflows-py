from __future__ import annotations

import tempfile
from pathlib import Path
from types import TracebackType
from typing import Generator, Type

import httpx
import pytest
import respx
from llama_agents.cli.config._config import ConfigManager
from llama_agents.cli.config.auth_service import AuthService
from llama_agents.cli.config.env_service import EnvService
from llama_agents.cli.config.schema import DeviceOIDC, Environment
from llama_agents.core.client.manage_client import ControlPlaneClient
from llama_agents.core.schema.projects import ProjectSummary
from llama_agents.core.schema.public import VersionResponse


@pytest.fixture
def temp_config() -> Generator[ConfigManager, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = ConfigManager()
        cfg.config_dir = Path(temp_dir)
        cfg.db_path = cfg.config_dir / "profiles.db"
        cfg._ensure_config_dir()
        cfg._init_database()
        yield cfg


@pytest.fixture
def env_svc(temp_config: ConfigManager) -> EnvService:
    return EnvService(lambda: temp_config)


def test_env_create_update_and_switch_clears_current_profile(
    env_svc: EnvService, temp_config: ConfigManager
) -> None:
    # Seed a profile and set it current
    env_url_a = "https://api.a.local"
    env_url_b = "https://api.b.local"
    temp_config.create_or_update_environment(env_url_a, requires_auth=False)
    temp_config.create_profile("p1", env_url_a, "proj-a")
    temp_config.set_settings_current_profile("p1")

    # Creating/updating environment B should switch current env and clear current profile
    env_svc.create_or_update_environment(
        Environment(api_url=env_url_b, requires_auth=True)
    )
    assert temp_config.get_settings_current_profile_name() is None
    assert temp_config.get_current_environment().api_url == env_url_b

    # Switching to A without existing env row should raise
    with pytest.raises(ValueError):
        env_svc.switch_environment("https://does-not-exist")

    # Create env A row and then switch; should also clear profile
    temp_config.create_or_update_environment(env_url_a, requires_auth=False)
    temp_config.set_settings_current_profile("p1")
    switched = env_svc.switch_environment(env_url_a)
    assert switched.api_url == env_url_a
    assert temp_config.get_settings_current_profile_name() is None


def test_env_auto_update_env_persists_changes(
    env_svc: EnvService, monkeypatch: pytest.MonkeyPatch, temp_config: ConfigManager
) -> None:
    # Seed an env row to be updated
    env = Environment(api_url="https://api.auto.local", requires_auth=False)
    temp_config.create_or_update_environment(env.api_url, env.requires_auth)

    # Monkeypatch version fetch to return updated values
    def fake_fetch(self: AuthService) -> VersionResponse:
        return VersionResponse(
            version="1.2.3", requires_auth=True, min_llamactl_version="0.3.0a99"
        )

    monkeypatch.setattr(AuthService, "fetch_server_version", fake_fetch)

    updated = env_svc.auto_update_env(env)
    assert updated.requires_auth is True
    assert updated.min_llamactl_version == "0.3.0a99"

    # Verify persisted
    from_db = temp_config.get_environment(env.api_url)
    assert from_db is not None
    assert from_db.requires_auth is True
    assert from_db.min_llamactl_version == "0.3.0a99"


def test_env_probe_environment_uses_config_manager_instance(
    env_svc: EnvService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(self: AuthService) -> VersionResponse:
        return VersionResponse(
            version="0.0.1", requires_auth=False, min_llamactl_version=None
        )

    monkeypatch.setattr(AuthService, "fetch_server_version", fake_fetch)

    probed = env_svc.probe_environment("https://api.probe.local/")
    assert probed.api_url == "https://api.probe.local"
    assert probed.requires_auth is False


@pytest.fixture
def device_oidc() -> DeviceOIDC:
    return DeviceOIDC(
        device_name="my-device",
        user_id="user-123",
        email="test@example.com",
        client_id="client-123",
        discovery_url="https://auth.local/.well-known/openid-configuration",
        device_access_token="at-1",
        device_refresh_token="rt-1",
        device_id_token="idt-1",
    )


@pytest.mark.asyncio
@respx.mock
async def test_auth_middleware_refresh_updates_db(
    temp_config: ConfigManager, device_oidc: DeviceOIDC, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_url = "https://api.auth.local"
    temp_config.create_or_update_environment(env_url, requires_auth=True)
    prof = temp_config.create_profile("p", env_url, "proj-1", device_oidc=device_oidc)
    temp_config.set_settings_current_profile("p")

    svc = AuthService(temp_config, Environment(api_url=env_url, requires_auth=True))
    auth = svc.auth_middleware()
    assert isinstance(auth, httpx.Auth)

    # Mock OIDC discovery and refresh token exchange
    discovery_url = device_oidc.discovery_url
    token_endpoint = "https://auth.local/oauth/token"
    respx.get(discovery_url).mock(
        return_value=respx.MockResponse(200, json={"token_endpoint": token_endpoint})
    )
    respx.post(token_endpoint).mock(
        return_value=respx.MockResponse(
            200,
            json={
                "access_token": "at-2",
                "refresh_token": "rt-2",
                "id_token": "idt-2",
                "token_type": "Bearer",
            },
        )
    )

    # Target request: first 401, then 200 with updated Authorization header
    calls = {"count": 0}

    def sequenced_response(req: httpx.Request) -> respx.MockResponse:
        if calls["count"] == 0:
            calls["count"] += 1
            return respx.MockResponse(401)
        assert req.headers.get("Authorization") == "Bearer at-2"
        return respx.MockResponse(200)

    respx.get("https://example.com/x").mock(side_effect=sequenced_response)

    # Execute request through httpx with the auth middleware
    async with httpx.AsyncClient(auth=auth) as client:
        response = await client.get("https://example.com/x")
        assert response.status_code == 200

    # Verify DB updated via on_refresh callback
    from_db = temp_config.get_profile_by_id(prof.id)
    assert from_db is not None and from_db.device_oidc is not None
    assert from_db.device_oidc.device_access_token == "at-2"
    assert from_db.device_oidc.device_refresh_token == "rt-2"
    assert from_db.device_oidc.device_id_token == "idt-2"


def test_auth_profile_creation_helpers(
    temp_config: ConfigManager, device_oidc: DeviceOIDC
) -> None:
    env_url = "https://api.create.local"
    temp_config.create_or_update_environment(env_url, requires_auth=True)
    svc = AuthService(temp_config, Environment(api_url=env_url, requires_auth=True))

    # From token
    prof1 = svc.create_profile_from_token("proj-1", api_key="abc 123 456 789 000")
    assert prof1.api_url == env_url
    assert temp_config.get_settings_current_profile_name() == prof1.name
    # Masked name should include **** and use first and last chars from token sans spaces
    assert "****" in prof1.name

    # From OIDC
    prof2 = svc.create_or_update_profile_from_oidc("proj-2", device_oidc)
    assert prof2.device_oidc is not None
    assert temp_config.get_settings_current_profile_name() == prof2.name


def test_auth_fetch_server_version_and_list_projects(
    monkeypatch: pytest.MonkeyPatch, temp_config: ConfigManager
) -> None:
    env_url = "https://api.manage.local"
    temp_config.create_or_update_environment(env_url, requires_auth=False)
    svc = AuthService(temp_config, Environment(api_url=env_url, requires_auth=False))

    class DummyCtx:
        async def __aenter__(self) -> "DummyCtx":
            return self

        async def __aexit__(
            self,
            exc_type: Type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

        async def server_version(self) -> VersionResponse:
            return VersionResponse(version="9.9.9", requires_auth=False)

        async def list_projects(self) -> list[ProjectSummary]:
            return [
                ProjectSummary(
                    project_id="p1",
                    project_name="Proj One",
                    deployment_count=0,
                )
            ]

    def fake_ctx(
        base_url: str, api_key: str | None = None, auth: httpx.Auth | None = None
    ) -> DummyCtx:
        return DummyCtx()

    monkeypatch.setattr(
        ControlPlaneClient,
        "ctx",
        classmethod(
            lambda cls, base_url, api_key=None, auth=None: fake_ctx(
                base_url, api_key, auth
            )
        ),
    )

    ver = svc.fetch_server_version()
    assert ver.version == "9.9.9"

    projects = svc._validate_token_and_list_projects("ignored-token")
    assert len(projects) == 1
    assert projects[0].project_id == "p1"
