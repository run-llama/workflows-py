from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner
from llama_agents.cli.app import app
from llama_agents.cli.commands.auth import _create_or_update_agent_api_key
from llama_agents.cli.config.schema import Auth


def test_auth_create_api_key_profile_non_interactive_validation() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["auth", "token", "--no-interactive"])
    assert result.exit_code != 0
    assert (
        "--api-key and --project-id are required in non-interactive mode"
        in result.output
    )


def test_auth_create_api_key_profile_non_interactive_success() -> None:
    runner = CliRunner()
    with patch("llama_agents.cli.config.env_service.service") as mock_service:
        mock_auth_svc = MagicMock()
        mock_auth_svc.create_profile_from_token.return_value = SimpleNamespace(
            name="prof"
        )
        mock_service.current_auth_service.return_value = mock_auth_svc

        result = runner.invoke(
            app,
            [
                "auth",
                "token",
                "--no-interactive",
                "--project-id",
                "p",
                "--api-key",
                "key",
            ],
        )
        assert result.exit_code == 0
        mock_auth_svc.create_profile_from_token.assert_called_once_with("p", "key")


def test_auth_list_profiles_no_profiles() -> None:
    runner = CliRunner()
    with patch("llama_agents.cli.config.env_service.service") as mock_service:
        mock_auth_svc = MagicMock()
        mock_auth_svc.list_profiles.return_value = []
        mock_auth_svc.get_current_profile.return_value = None
        mock_service.current_auth_service.return_value = mock_auth_svc
        result = runner.invoke(app, ["auth", "list"])
        assert result.exit_code == 0
        assert "No profiles found" in result.output


def test_auth_switch_profile_success_and_missing() -> None:
    runner = CliRunner()
    with (
        patch("llama_agents.cli.commands.auth._select_profile") as mock_select,
        patch("llama_agents.cli.config.env_service.service") as mock_service,
    ):
        mock_auth_svc = MagicMock()
        mock_service.current_auth_service.return_value = mock_auth_svc
        mock_select.return_value = SimpleNamespace(name="p1")
        result = runner.invoke(app, ["auth", "switch", "p1"])
        assert result.exit_code == 0
        mock_auth_svc.set_current_profile.assert_called_once_with("p1")

    with (
        patch("llama_agents.cli.commands.auth._select_profile", return_value=None),
        patch("llama_agents.cli.config.env_service.service") as mock_service2,
    ):
        mock_service2.current_auth_service.return_value = MagicMock()
        result = runner.invoke(app, ["auth", "switch", "doesnt-exist"])
        assert result.exit_code == 0
        assert "No profile selected" in result.output


def test_auth_logout_existing() -> None:
    runner = CliRunner()
    with (
        patch("llama_agents.cli.commands.auth._select_profile") as mock_select,
        patch("llama_agents.cli.config.env_service.service") as mock_service,
    ):
        mock_auth_svc = MagicMock()
        mock_service.current_auth_service.return_value = mock_auth_svc
        mock_select.return_value = SimpleNamespace(name="p1")
        mock_auth_svc.delete_profile = AsyncMock(return_value=True)
        result = runner.invoke(app, ["auth", "logout", "p1"])
        assert result.exit_code == 0


def test_auth_logout_missing() -> None:
    runner = CliRunner()
    with (
        patch("llama_agents.cli.commands.auth._select_profile", return_value=None),
        patch("llama_agents.cli.config.env_service.service") as mock_service2,
    ):
        mock_service2.current_auth_service.return_value = MagicMock()
        result = runner.invoke(app, ["auth", "logout", "missing"])
        assert result.exit_code == 0
        assert "No profile selected" in result.output


def test_auth_project_non_interactive_requires_arg() -> None:
    runner = CliRunner()
    with patch(
        "llama_agents.cli.commands.auth.validate_authenticated_profile",
        return_value=MagicMock(name="p", project_id="x"),
    ):
        result = runner.invoke(app, ["auth", "project", "--no-interactive"])
        assert result.exit_code != 0
        assert "No --project-id provided" in result.output


def test_auth_project_interactive_sets_selected() -> None:
    runner = CliRunner()
    with (
        patch(
            "llama_agents.cli.commands.auth.validate_authenticated_profile",
            return_value=MagicMock(name="p"),
        ),
        patch(
            "llama_agents.cli.commands.auth._list_projects",
            return_value=[
                MagicMock(project_id="proj", project_name="Proj", deployment_count=1)
            ],
        ),
        patch("questionary.select") as mock_select,
        patch("llama_agents.cli.config.env_service.service") as mock_service,
    ):
        mock_auth_svc = MagicMock()
        mock_service.current_auth_service.return_value = mock_auth_svc
        mock_select.return_value.ask.return_value = "proj"
        result = runner.invoke(app, ["auth", "project", "--interactive"])
        assert result.exit_code == 0
        mock_auth_svc.set_project.assert_called_once()


@pytest.mark.asyncio
async def test_create_or_update_agent_api_key_raises_on_network_error_without_retry() -> (
    None
):
    """Network errors while provisioning API tokens should surface immediately
    with a clear message, without auto-retrying.

    create_agent_api_key is a non-idempotent POST. Auto-retrying on a transient
    network error can produce duplicate API keys per llamactl login. See the
    2026-04-08 document-extraction rollout plan Phase 3 client retry audit.
    """
    profile = Auth(
        id="id-1",
        name="test",
        api_url="https://example.com",
        project_id="proj",
        api_key=None,
        api_key_id=None,
        device_oidc=None,
    )

    mock_auth_svc = MagicMock()
    mock_client_cm = AsyncMock()
    mock_client = MagicMock()
    mock_client_cm.__aenter__.return_value = mock_client
    mock_auth_svc.profile_client.return_value = mock_client_cm

    mock_client.create_agent_api_key = AsyncMock(side_effect=httpx.RequestError("boom"))

    with pytest.raises(Exception) as exc_info:
        await _create_or_update_agent_api_key(mock_auth_svc, profile)

    msg = str(exc_info.value)
    assert "Network error while provisioning an API token" in msg
    # Should have tried exactly once — no silent retry amplification on
    # non-idempotent POST.
    assert mock_client.create_agent_api_key.await_count == 1
