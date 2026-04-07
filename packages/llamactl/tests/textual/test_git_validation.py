import asyncio
from pathlib import Path
from typing import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from llama_agents.cli.textual.git_validation import (
    GitValidationWidget,
    ValidationCancelMessage,
    ValidationResultMessage,
)
from llama_agents.core.schema.git_validation import RepositoryValidationResponse
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button

T = RepositoryValidationResponse


async def _fake_run_with_network_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
) -> T:
    """Fake retry loop that actually retries up to max_attempts."""
    attempts = 0
    while True:
        try:
            attempts += 1
            return await operation()
        except httpx.RequestError:
            if attempts >= max_attempts:
                raise


async def _passthrough_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
) -> T:
    """Passthrough that calls the operation once with no retries."""
    return await operation()


class GitValidationTestApp(App):
    """Helper app for testing GitValidationWidget with proper container and CSS context."""

    # Use the same CSS as the real app
    CSS_PATH = (
        Path(__file__).parent.parent.parent
        / "src"
        / "llama_agents"
        / "cli"
        / "textual"
        / "styles.tcss"
    )

    def __init__(self, widget: GitValidationWidget):
        super().__init__()
        self.widget = widget
        self.posted_messages: list[object] = []

    def compose(self) -> ComposeResult:
        # Replicate the same container structure as DeploymentEditApp
        with Container(classes="form-container"):
            yield self.widget

    def on_validation_result_message(self, message: ValidationResultMessage) -> None:
        self.posted_messages.append(message)

    def on_validation_cancel_message(self, message: ValidationCancelMessage) -> None:
        self.posted_messages.append(message)


@pytest.mark.asyncio
async def test_initial_validation_flow_success() -> None:
    """Test widget starts validation and transitions to success on immediate success."""
    mock_response = RepositoryValidationResponse(
        accessible=True,
        message="Repository accessible",
        github_app_installation_url=None,
        pat_is_obsolete=False,
    )

    with patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Widget should start in validating state
            assert widget.current_state == "validating"

            # Wait for validation to complete
            await pilot.pause()

            # Should post ValidationResultMessage and not change state (exits immediately)
            assert len(app.posted_messages) == 1
            message = app.posted_messages[0]
            assert isinstance(message, ValidationResultMessage)
            assert message.repo_url == "https://github.com/user/repo"
            assert message.pat is None


@pytest.mark.asyncio
async def test_initial_validation_flow_failure() -> None:
    """Test widget transitions to options state on validation failure."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_authorization_url="https://github.com/login/oauth/authorize?client_id=test",
        github_app_installation_url="https://github.com/apps/llama-deploy/installations/new",
        pat_is_obsolete=False,
    )

    with patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for validation to complete
            await pilot.pause()

            # Should transition to options state
            assert widget.current_state == "options"
            assert widget.validation_response == mock_response

            # Should have GitHub App and PAT buttons
            github_button = app.query_one("#install_github_app", Button)
            pat_button = app.query_one("#use_pat", Button)
            save_anyway_button = app.query_one("#save_anyway", Button)

            assert github_button is not None
            assert pat_button is not None
            assert save_anyway_button is not None

            # With authorization URL present, button should say "Connect GitHub"
            assert github_button.label == "Connect GitHub (Recommended)"


@pytest.mark.asyncio
async def test_options_state_github_app_button() -> None:
    """Test Install GitHub App button starts GitHub auth flow."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url="https://github.com/apps/llama-deploy/installations/new",
        pat_is_obsolete=False,
    )

    with (
        patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client,
        patch(
            "llama_agents.cli.textual.git_validation.webbrowser.open"
        ) as mock_webbrowser,
        patch(
            "llama_agents.cli.textual.git_validation.GitHubCallbackServer"
        ) as mock_callback_server,
    ):
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        # Mock callback server
        mock_server_instance = MagicMock()
        continue_signal = asyncio.Event()

        async def start_and_wait(timeout: float = 300) -> None:
            await continue_signal.wait()

        mock_server_instance.start_and_wait = (
            start_and_wait  # delay completion so we can assert things
        )
        mock_server_instance.stop = AsyncMock()

        mock_callback_server.return_value = mock_server_instance

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation
            await pilot.pause()

            # Click GitHub App button
            await pilot.click("#install_github_app")

            # Should transition to github_auth state
            # Take a screenshot before assertion to debug state

            assert widget.current_state == "github_auth"

            # Should open browser and start callback server
            mock_webbrowser.assert_called_once_with(
                "https://github.com/apps/llama-deploy/installations/new"
            )
            mock_callback_server.assert_called_once()
            valid = mock_response.model_copy(deep=True)
            valid.accessible = True
            mock_client.validate_repository.return_value = valid
            continue_signal.set()
            await pilot.pause()

            # After callback completes and re-validation succeeds,
            # a ValidationResultMessage should be posted
            assert len(app.posted_messages) == 1
            message = app.posted_messages[0]
            assert isinstance(message, ValidationResultMessage)
            assert message.repo_url == "https://github.com/user/repo"


@pytest.mark.asyncio
async def test_options_state_pat_button() -> None:
    """Test Use PAT button transitions to PAT input state."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url="https://github.com/apps/llama-deploy/installations/new",
        pat_is_obsolete=False,
    )

    with patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation
            await pilot.pause()

            # Click PAT button
            await pilot.click("#use_pat")
            await pilot.pause()

            # Should transition to pat_input state
            assert widget.current_state == "pat_input"
            assert widget.error_message == ""

            # Should have PAT input field
            pat_input = app.query_one("#pat_input")
            assert pat_input is not None


@pytest.mark.asyncio
async def test_options_state_save_anyway_button() -> None:
    """Test Save Anyway button posts ValidationResultMessage."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url=None,
        pat_is_obsolete=False,
    )

    with patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation
            await pilot.pause()

            # Click Save Anyway button
            await pilot.click("#save_anyway")
            await pilot.pause()

            # Should post ValidationResultMessage
            assert len(app.posted_messages) == 1
            message = app.posted_messages[0]
            assert isinstance(message, ValidationResultMessage)
            assert message.repo_url == "https://github.com/user/repo"
            assert message.pat is None


@pytest.mark.asyncio
async def test_pat_input_validation_empty() -> None:
    """Test PAT input validation with empty value."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url=None,
        pat_is_obsolete=False,
    )

    with patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation and go to PAT input
            await pilot.pause()
            await pilot.click("#use_pat")
            await pilot.pause()

            # Leave PAT input empty and click validate
            await pilot.click("#validate_pat")
            await pilot.pause()

            # Should show error and stay in pat_input state
            assert widget.current_state == "pat_input"
            assert widget.error_message == "Please enter a Personal Access Token"


@pytest.mark.asyncio
async def test_pat_input_validation_success() -> None:
    """Test successful PAT validation."""
    # Initial failure response
    mock_fail_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url=None,
        pat_is_obsolete=False,
    )

    # Success response with PAT
    mock_success_response = RepositoryValidationResponse(
        accessible=True,
        message="Repository accessible with PAT",
        github_app_installation_url=None,
        pat_is_obsolete=False,
    )

    with patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(
            side_effect=[mock_fail_response, mock_success_response]
        )
        mock_get_client.return_value = mock_client

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation and go to PAT input
            await pilot.pause()
            await pilot.click("#use_pat")
            await pilot.pause()

            # Enter PAT and validate
            pat_input = app.query_one("#pat_input")
            pat_input.value = "test-pat-token"  # type: ignore

            await pilot.click("#validate_pat")
            await pilot.pause()

            # Should post ValidationResultMessage with PAT
            assert len(app.posted_messages) == 1
            message = app.posted_messages[0]
            assert isinstance(message, ValidationResultMessage)
            assert message.repo_url == "https://github.com/user/repo"
            assert message.pat == "test-pat-token"


@pytest.mark.asyncio
async def test_pat_input_back_button() -> None:
    """Test back button returns to options state."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url="https://github.com/apps/llama-deploy/installations/new",
        pat_is_obsolete=False,
    )

    with patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation and go to PAT input
            await pilot.pause()
            await pilot.click("#use_pat")
            await pilot.pause()

            # Click back button
            await pilot.click("#back_to_options")
            await pilot.pause()

            # Should return to options state
            assert widget.current_state == "options"
            assert widget.error_message == ""


@pytest.mark.asyncio
async def test_github_auth_cancel_button() -> None:
    """Test cancel button in GitHub auth state returns to options."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url="https://github.com/apps/llama-deploy/installations/new",
        pat_is_obsolete=False,
    )

    with (
        patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client,
        patch("llama_agents.cli.textual.git_validation.webbrowser.open"),
        patch(
            "llama_agents.cli.textual.git_validation.GitHubCallbackServer"
        ) as mock_callback_server,
    ):
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        # Mock callback server
        mock_server_instance = MagicMock()
        continue_signal = asyncio.Event()

        async def start_and_wait(timeout: float = 300) -> None:
            await continue_signal.wait()

        mock_server_instance.start_and_wait = (
            start_and_wait  # delay completion so we can assert things
        )
        mock_server_instance.stop = AsyncMock()
        mock_callback_server.return_value = mock_server_instance

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation and start GitHub auth
            await pilot.pause()
            await pilot.click("#install_github_app")
            await pilot.pause()

            # Click cancel button
            await pilot.click("#cancel_github_auth")
            await pilot.pause()

            # Should return to options state and stop server
            assert widget.current_state == "options"
            mock_server_instance.stop.assert_called_once()


@pytest.mark.asyncio
async def test_github_auth_recheck_button() -> None:
    """Test recheck button in GitHub auth state re-validates repository."""
    mock_fail_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url="https://github.com/apps/llama-deploy/installations/new",
        pat_is_obsolete=False,
    )

    mock_success_response = RepositoryValidationResponse(
        accessible=True,
        message="Repository accessible via GitHub App",
        github_app_installation_url=None,
        pat_is_obsolete=False,
    )

    with (
        patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client,
        patch("llama_agents.cli.textual.git_validation.webbrowser.open"),
        patch(
            "llama_agents.cli.textual.git_validation.GitHubCallbackServer"
        ) as mock_callback_server,
    ):
        mock_client = MagicMock()

        mock_client.validate_repository = AsyncMock(
            side_effect=[mock_fail_response, mock_success_response]
        )
        mock_get_client.return_value = mock_client

        # Mock callback server
        mock_server_instance = MagicMock()
        continue_signal = asyncio.Event()

        async def start_and_wait(timeout: float = 300) -> None:
            await continue_signal.wait()

        mock_server_instance.start_and_wait = (
            start_and_wait  # delay completion so we can assert things
        )
        mock_server_instance.stop = AsyncMock()
        mock_callback_server.return_value = mock_server_instance

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation and start GitHub auth
            await pilot.pause()
            await pilot.click("#install_github_app")
            await pilot.pause()

            # Click recheck button
            await pilot.click("#recheck_github")
            await pilot.pause()

            # Should transition to success and post message
            assert widget.current_state == "success"
            assert len(app.posted_messages) == 1
            message = app.posted_messages[0]
            assert isinstance(message, ValidationResultMessage)
            assert message.repo_url == "https://github.com/user/repo"


@pytest.mark.asyncio
async def test_universal_cancel_button() -> None:
    """Test universal cancel button posts ValidationCancelMessage."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url=None,
        pat_is_obsolete=False,
    )

    with patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation
            await pilot.pause()

            # Click cancel button
            await pilot.click("#cancel")
            await pilot.pause()

            # Should post ValidationCancelMessage
            assert len(app.posted_messages) == 1
            message = app.posted_messages[0]
            assert isinstance(message, ValidationCancelMessage)


@pytest.mark.asyncio
async def test_pat_obsolescence_flow() -> None:
    """Test PAT obsolescence success flow."""
    mock_response = RepositoryValidationResponse(
        accessible=True,
        message="Repository accessible via GitHub App",
        github_app_installation_url=None,
        pat_is_obsolete=True,
    )

    with patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        widget = GitValidationWidget("https://github.com/user/repo", pat="existing-pat")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for validation to complete
            await pilot.pause()

            # Should transition to success state with obsolescence message
            assert widget.current_state == "success"
            assert "obsolete" in widget.error_message

            # Click continue button
            await pilot.click("#continue_success")
            await pilot.pause()

            # Should post ValidationResultMessage with empty PAT to clear it
            assert len(app.posted_messages) == 1
            message = app.posted_messages[0]
            assert isinstance(message, ValidationResultMessage)
            assert message.repo_url == "https://github.com/user/repo"
            assert message.pat == ""  # PAT should be cleared


@pytest.mark.asyncio
async def test_initial_validation_network_error_retries_and_sets_message() -> None:
    """Network errors during initial validation should be retried with a helpful error message."""

    with (
        patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client,
        patch(
            "llama_agents.cli.textual.git_validation.run_with_network_retries"
        ) as mock_run_with_retries,
    ):
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(
            side_effect=httpx.RequestError("net-down")
        )
        mock_get_client.return_value = mock_client

        mock_run_with_retries.side_effect = _fake_run_with_network_retries

        widget = GitValidationWidget("https://github.com/user/repo")

        await widget._validate_repository()

        assert widget.current_state == "options"
        assert (
            "Network error while validating repository access" in widget.error_message
        )
        assert "net-down" in widget.error_message
        assert mock_client.validate_repository.await_count == 3


@pytest.mark.asyncio
async def test_recheck_github_network_error_retries_and_sets_message() -> None:
    """Network errors during GitHub recheck should be retried with a helpful error message."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_installation_url="https://github.com/apps/llama-deploy/installations/new",
        pat_is_obsolete=False,
    )

    with (
        patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client,
        patch(
            "llama_agents.cli.textual.git_validation.run_with_network_retries"
        ) as mock_run_with_retries,
    ):
        mock_client = MagicMock()
        # Fail all recheck attempts with network errors
        mock_client.validate_repository = AsyncMock(
            side_effect=[
                httpx.RequestError("net-down"),
                httpx.RequestError("net-down"),
                httpx.RequestError("net-down"),
            ]
        )
        mock_get_client.return_value = mock_client

        mock_run_with_retries.side_effect = _fake_run_with_network_retries

        widget = GitValidationWidget("https://github.com/user/repo")
        # Seed initial response and state as if GitHub auth has started
        widget.validation_response = mock_response
        widget.current_state = "github_auth"

        await widget._recheck_github_auth()

        assert (
            "Network error while re-checking GitHub App installation"
            in widget.error_message
        )
        assert "net-down" in widget.error_message
        # Three attempts for the recheck
        assert mock_client.validate_repository.await_count == 3


@pytest.mark.asyncio
async def test_authorization_url_triggers_auth_flow_first() -> None:
    """When both authorization and installation URLs are present, clicking the GitHub button
    should open the authorization URL first and set _github_auth_step to 'authorization'."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_authorization_url="https://github.com/login/oauth/authorize?client_id=test",
        github_app_installation_url="https://github.com/apps/test/installations/new",
        pat_is_obsolete=False,
    )

    with (
        patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client,
        patch(
            "llama_agents.cli.textual.git_validation.webbrowser.open"
        ) as mock_webbrowser,
        patch(
            "llama_agents.cli.textual.git_validation.GitHubCallbackServer"
        ) as mock_callback_server,
    ):
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        # Mock callback server
        mock_server_instance = MagicMock()
        continue_signal = asyncio.Event()

        async def start_and_wait(timeout: float = 300) -> None:
            await continue_signal.wait()

        mock_server_instance.start_and_wait = start_and_wait
        mock_server_instance.stop = AsyncMock()
        mock_callback_server.return_value = mock_server_instance

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation
            await pilot.pause()

            # Verify button label
            github_button = app.query_one("#install_github_app", Button)
            assert github_button.label == "Connect GitHub (Recommended)"

            # Click GitHub button
            await pilot.click("#install_github_app")

            assert widget.current_state == "github_auth"
            assert widget._github_auth_step == "authorization"

            # Should open the authorization URL, NOT the installation URL
            mock_webbrowser.assert_called_once_with(
                "https://github.com/login/oauth/authorize?client_id=test"
            )
            mock_callback_server.assert_called_once()


@pytest.mark.asyncio
async def test_both_urls_chains_authorization_then_installation() -> None:
    """When authorization completes but repo is still not accessible, it should
    automatically proceed to the installation step, then succeed after the second callback."""

    # Responses for the three validate_repository calls:
    # 1. Initial validation -> not accessible, both URLs
    # 2. After authorization callback -> not accessible, installation URL only
    # 3. After installation callback -> accessible
    initial_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_authorization_url="https://github.com/login/oauth/authorize?client_id=test",
        github_app_installation_url="https://github.com/apps/test/installations/new",
        pat_is_obsolete=False,
    )
    after_auth_response = RepositoryValidationResponse(
        accessible=False,
        message="Still not accessible, need installation",
        github_app_authorization_url=None,
        github_app_installation_url="https://github.com/apps/test/installations/new",
        pat_is_obsolete=False,
    )
    after_install_response = RepositoryValidationResponse(
        accessible=True,
        message="Repository accessible via GitHub App",
        github_app_installation_url=None,
        pat_is_obsolete=False,
    )

    with (
        patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client,
        patch(
            "llama_agents.cli.textual.git_validation.webbrowser.open"
        ) as mock_webbrowser,
        patch(
            "llama_agents.cli.textual.git_validation.GitHubCallbackServer"
        ) as mock_callback_server,
        patch(
            "llama_agents.cli.textual.git_validation.run_with_network_retries"
        ) as mock_run_with_retries,
    ):
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(
            side_effect=[initial_response, after_auth_response, after_install_response]
        )
        mock_get_client.return_value = mock_client

        mock_run_with_retries.side_effect = _passthrough_retries

        # Two server instances for two callback steps
        mock_server_1 = MagicMock()
        mock_server_1.start_and_wait = AsyncMock()
        mock_server_1.stop = AsyncMock()

        mock_server_2 = MagicMock()
        mock_server_2.start_and_wait = AsyncMock()
        mock_server_2.stop = AsyncMock()

        # _validate_after_callback will call GitHubCallbackServer() for the installation step
        mock_callback_server.return_value = mock_server_2

        widget = GitValidationWidget("https://github.com/user/repo")
        # Seed the initial state as if initial validation completed
        widget.validation_response = initial_response
        widget.current_state = "github_auth"
        widget._github_auth_step = "authorization"
        widget.github_callback_server = mock_server_1

        await widget._wait_for_callback()

        # After the full chain: authorization callback -> re-validate (not accessible)
        # -> installation callback -> re-validate (accessible) -> post message
        mock_server_1.start_and_wait.assert_awaited_once()
        mock_server_1.stop.assert_awaited_once()
        mock_server_2.start_and_wait.assert_awaited_once()
        mock_server_2.stop.assert_awaited_once()

        # Verify installation URL was opened during the chain
        mock_webbrowser.assert_called_once_with(
            "https://github.com/apps/test/installations/new"
        )

        # Verify _github_auth_step progressed to installation
        assert widget._github_auth_step == "installation"


@pytest.mark.asyncio
async def test_authorization_alone_resolves_skips_installation() -> None:
    """When authorization completes and repo IS accessible, it should NOT
    proceed to the installation step and should post ValidationResultMessage."""

    initial_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_authorization_url="https://github.com/login/oauth/authorize?client_id=test",
        github_app_installation_url="https://github.com/apps/test/installations/new",
        pat_is_obsolete=False,
    )
    after_auth_response = RepositoryValidationResponse(
        accessible=True,
        message="Repository accessible via GitHub App",
        github_app_installation_url=None,
        pat_is_obsolete=False,
    )

    with (
        patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client,
        patch("llama_agents.cli.textual.git_validation.webbrowser.open"),
        patch(
            "llama_agents.cli.textual.git_validation.GitHubCallbackServer"
        ) as mock_callback_server,
        patch(
            "llama_agents.cli.textual.git_validation.run_with_network_retries"
        ) as mock_run_with_retries,
    ):
        mock_client = MagicMock()
        # Only one validate_repository call expected: _validate_after_callback
        # returns accessible=True, so no chaining to installation step.
        mock_client.validate_repository = AsyncMock(return_value=after_auth_response)
        mock_get_client.return_value = mock_client

        mock_run_with_retries.side_effect = _passthrough_retries

        mock_server_instance = MagicMock()
        mock_server_instance.start_and_wait = AsyncMock()
        mock_server_instance.stop = AsyncMock()

        widget = GitValidationWidget("https://github.com/user/repo")
        widget.validation_response = initial_response
        widget.current_state = "github_auth"
        widget._github_auth_step = "authorization"
        widget.github_callback_server = mock_server_instance

        # Collect posted messages
        posted_messages: list[object] = []
        widget.post_message = lambda msg: posted_messages.append(msg) or True  # type: ignore

        await widget._wait_for_callback()

        # Should have posted ValidationResultMessage (success)
        assert len(posted_messages) == 1
        message = posted_messages[0]
        assert isinstance(message, ValidationResultMessage)
        assert message.repo_url == "https://github.com/user/repo"
        assert message.pat is None

        # Should NOT have created a second callback server (no installation step)
        mock_callback_server.assert_not_called()

        # Auth step should still be authorization (never progressed)
        assert widget._github_auth_step == "authorization"


@pytest.mark.asyncio
async def test_no_authorization_url_falls_back_to_installation() -> None:
    """When only github_app_installation_url is present (no authorization URL),
    clicking the GitHub button should open the installation URL and set
    _github_auth_step to 'installation'."""
    mock_response = RepositoryValidationResponse(
        accessible=False,
        message="Repository not accessible",
        github_app_authorization_url=None,
        github_app_installation_url="https://github.com/apps/llama-deploy/installations/new",
        pat_is_obsolete=False,
    )

    with (
        patch("llama_agents.cli.textual.git_validation.get_client") as mock_get_client,
        patch(
            "llama_agents.cli.textual.git_validation.webbrowser.open"
        ) as mock_webbrowser,
        patch(
            "llama_agents.cli.textual.git_validation.GitHubCallbackServer"
        ) as mock_callback_server,
    ):
        mock_client = MagicMock()
        mock_client.validate_repository = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        # Mock callback server
        mock_server_instance = MagicMock()
        continue_signal = asyncio.Event()

        async def start_and_wait(timeout: float = 300) -> None:
            await continue_signal.wait()

        mock_server_instance.start_and_wait = start_and_wait
        mock_server_instance.stop = AsyncMock()
        mock_callback_server.return_value = mock_server_instance

        widget = GitValidationWidget("https://github.com/user/repo")
        app = GitValidationTestApp(widget)

        async with app.run_test(size=(100, 40)) as pilot:
            # Wait for initial validation
            await pilot.pause()

            # Verify button label shows old-style text
            github_button = app.query_one("#install_github_app", Button)
            assert github_button.label == "Install GitHub App (Recommended)"

            # Click GitHub button
            await pilot.click("#install_github_app")

            assert widget.current_state == "github_auth"
            assert widget._github_auth_step == "installation"

            # Should open the installation URL directly
            mock_webbrowser.assert_called_once_with(
                "https://github.com/apps/llama-deploy/installations/new"
            )
            mock_callback_server.assert_called_once()
