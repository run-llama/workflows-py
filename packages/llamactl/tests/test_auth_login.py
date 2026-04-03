from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from llama_agents.cli.commands.auth import _create_device_profile


def _make_fake_device_oidc() -> SimpleNamespace:
    return SimpleNamespace(
        device_name="dev",
        user_id="user-1",
        email="user@example.com",
        client_id="cli",
        discovery_url="https://example.com/.well-known/openid-configuration",
        device_access_token="access-token",
        device_refresh_token="refresh-token",
        device_id_token="id-token",
    )


def _make_fake_auth_profile() -> SimpleNamespace:
    return SimpleNamespace(
        id="id-1",
        name="test-profile",
        api_url="https://api.example.com",
        project_id="proj-1",
        api_key=None,
        api_key_id=None,
        device_oidc=None,
    )


def test_create_device_profile_cleans_up_on_api_key_failure() -> None:
    """If API key provisioning fails, the partially created profile should be deleted."""

    fake_profile = _make_fake_auth_profile()

    mock_auth_svc = MagicMock()
    mock_auth_svc.create_or_update_profile_from_oidc.return_value = fake_profile
    mock_auth_svc.delete_profile = AsyncMock(return_value=True)

    with (
        patch("llama_agents.cli.config.env_service.service") as mock_service,
        patch(
            "llama_agents.cli.commands.auth._run_device_authentication",
            return_value=_make_fake_device_oidc(),
        ),
        patch(
            "llama_agents.cli.commands.auth._list_projects",
            return_value=[SimpleNamespace(project_id="proj-1")],
        ),
        patch(
            "llama_agents.cli.commands.auth._select_or_enter_project",
            return_value="proj-1",
        ),
        patch(
            "llama_agents.cli.commands.auth._create_or_update_agent_api_key",
            side_effect=click.ClickException("network error"),
        ),
    ):
        mock_service.current_auth_service.return_value = mock_auth_svc

        with pytest.raises(click.ClickException, match="network error"):
            _create_device_profile()

    mock_auth_svc.delete_profile.assert_awaited_once_with(fake_profile.name)
