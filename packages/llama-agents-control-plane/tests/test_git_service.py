from typing import Generator
from unittest.mock import patch

import llama_agents.control_plane.git._git_service as git_service_module
import pytest
import respx
from llama_agents.control_plane.git import GitService, git_service
from llama_agents.control_plane.git._github_auth import GitHubAppAuth

GIT_SERVICE = "llama_agents.control_plane.git._git_service"


@pytest.fixture(autouse=True)
def mock_github_api() -> Generator[respx.Router, None, None]:
    """Just a safeguard to mock GitHub API calls to prevent real HTTP requests during tests."""
    with respx.mock as mock_router:
        # Mock get_owner_info API calls
        yield mock_router


@pytest.fixture(autouse=True)
def default_public_probe_false() -> Generator[None, None, None]:
    """Default to not-public for git probe unless a test overrides it."""
    with patch(f"{GIT_SERVICE}.validate_git_public_access", return_value=False):
        yield


@pytest.fixture
def service() -> GitService:
    return git_service


@pytest.fixture
def project_id() -> str:
    return "test-project"


@pytest.fixture
def github_app_auth() -> Generator[GitHubAppAuth, None, None]:
    # fake key for testing. Somewhat mangled to avoid the pre-commit hook.
    private_key = (
        "-----BEGIN RSA PRIV"
        "ATE KEY-----\n"
        "MIIEowIBAAKCAQEAlbOWExC3QtsEKSArcrogbO8L39M5Mn0fLtEmtzg5RSQ4FtIY\n"
        "ehHc3TAxcCi6haUqsdht2P7rT2Q0aDPhQgG9m+8BpS9LOFOh4cfd99q0lQSlMVCC\n"
        "zV25NuCljdKzT52n0byzVNIUMW9+knrswPNQy/LpEyxLvBzFyskdsoIyOKwT4Nl0\n"
        "L5T0S1pw2fxkhoGBsmIzfkU53k4yIAJ5heVBRJSwa/+RkfXY+7i8/PIixue5tMnJ\n"
        "kA6gcfG9xW7bnCL169wgN3NmM14/wr+3Aysnia0cyBUF/KO1v3bWp/lrjPOpL3uo\n"
        "hcDWdpZot5qt77GkoWZZNimRA6TjBaWCG0O1LwIDAQABAoIBADmcAUp3+dZ4G3tK\n"
        "Hn5Jo3XYbmzlx9KmtQvawDftIpj5jb42fuXnHuReCgB8I/+PZsVHIUrLGzoTuVlK\n"
        "ccrpiZLLIQp1D1DvWlJdjI239BuOzJWUQqoOgdrdC8jux0OBy9XadPbU26GEoyRy\n"
        "us6sNDEwW0KeHs0XE4Ts7YlHMlV9SCyx2kqoicb7vHiDJLhytAJAXNgSbQgWI05a\n"
        "c+gSknN1b4DBumb9x6MST9JBH4FobGtzCkswSNNsyrPZsax3env7bRDYYQosjRaA\n"
        "7Xc243bdkDyD9oAOKffLdCYxuloLxkPlVKXUrTTVGcHKxOoK5oRpuKRBKMZJVrKu\n"
        "aCbS0KkCgYEA0UdDNizillqLKbv+lAczzxOYhcXqaXsX2WhracDyYa8ycppoe6YU\n"
        "5MPlOL1f/R8HCncuwoXcfGdQqsTnrjsaMZDCFR37pCZeSvy7GxPMdsDB6x7bEk/L\n"
        "qQWKfPtAujhdfzdfJQ88cZjIhb1Pg2+lFBnpyLVK8fUxtBHR2LOhcPkCgYEAtx9e\n"
        "rb3jb1vQ2gHX0aDNA/tZ7Q3/Ika/pzDHbliaULm38Ts/yjojE/Mwc0lFLbHTa2vq\n"
        "4kXCDyJuTN5TurZsbg/gowoGQGNao0emR/Vvr4s9NXJFCoTEqX2W8d1DI70azqIt\n"
        "L2ukiPyeMuS6ijx9Sj/PyaK1TwAOL7VdY98qiWcCgYEAtw7svcC5WudMf28QGo/K\n"
        "Q8JSUgFzMF0Z2XQ7MMAzxDqpmBF0f2QhNpIcOWt9QT4YvJDP+Bt7Z94/c4DVX1QX\n"
        "b2++NRaK/WUKafF0ARVqbh3iAjZ1Tik6bliIcRad4cZYEmVu9k3Dg2IvVLzphoDs\n"
        "Fw8rrgLW0Zq2pVpJApLuDpECgYA1dJ/TwfmxWTEXYrBYjkMqpWXz0EEpBVQO/ytI\n"
        "Z+7sH7q1XaFabCwvN69uB/Z8x0s7MW6IjOqANoHSSJhSicwPOO1PSq7WfupHfbPp\n"
        "j5kBunissGW9E1LBU1sL0ZY2yY4YwbjE/fwyzON1YdWeYtgEI6qJZsjcfdymSqAv\n"
        "dkbZgwKBgCeH06jqN8d7+UrO/T8LM60RTRwJoUz5p8MY/qY863//5c79nB+zwGim\n"
        "L4wvm+sCW/GDzIEH+SjVLZXK1SALuQuzoUhfvINbgWqjsBDBzRptA4hSxM0UIHkd\n"
        "hn+lZmeXrFUJJa7VrO+6BlZWm20567cNZbjUKjKeuyI20t/9hVGP\n"
        "-----END RSA PRIV"
        "ATE KEY-----"
    )

    auth = GitHubAppAuth(
        client_id="12345",
        private_key=private_key,
        app_name="TestApp",
    )

    with (
        patch.object(git_service_module, "get_github_app_auth", return_value=auth),
        patch(
            "llama_agents.control_plane.git.github_api_client.get_github_app_auth",
            return_value=auth,
        ),
    ):
        yield auth


@pytest.fixture(autouse=True)
def no_github_app_auth() -> Generator[GitHubAppAuth | None, None, None]:
    with (
        patch.object(git_service_module, "get_github_app_auth", return_value=None),
        patch(
            "llama_agents.control_plane.git.github_api_client.get_github_app_auth",
            return_value=None,
        ),
    ):
        yield None


def mock_github_repo_and_owner(
    router: respx.Router,
    owner: str = "owner",
    repo: str = "repo",
    installation_id: int | None = None,
    is_org: bool = False,
    owner_exists: bool = True,
    repo_private: bool = True,
    repository_selection: str | None = None,
    repo_accessible: bool | None = None,
) -> None:
    """
    Mocks github apis given a scenario of a repo and owner. Mocks full https requests. If this gets
    to be too much, we should add separate tests for the api client, and mock the api client functions
    """
    owner_type = "Organization" if is_org else "User"
    if owner_exists:
        router.get(f"https://api.github.com/users/{owner}").mock(
            return_value=respx.MockResponse(
                200, json={"id": 12345, "login": owner, "type": owner_type}
            )
        )
        if is_org:
            router.get(f"https://api.github.com/orgs/{owner}").mock(
                return_value=respx.MockResponse(
                    200, json={"id": 12345, "login": owner, "type": "Organization"}
                )
            )
    else:
        router.get(f"https://api.github.com/users/{owner}").mock(
            return_value=respx.MockResponse(404)
        )
        if is_org:
            router.get(f"https://api.github.com/orgs/{owner}").mock(
                return_value=respx.MockResponse(404)
            )

    if repo_accessible is None:
        repo_accessible = not repo_private

    if repo_accessible:
        router.get(f"https://api.github.com/repos/{owner}/{repo}").mock(
            return_value=respx.MockResponse(
                200, json={"private": repo_private, "name": repo}
            )
        )
    else:
        router.get(f"https://api.github.com/repos/{owner}/{repo}").mock(
            return_value=respx.MockResponse(404)
        )
    if installation_id and is_org:
        org_installation_payload: dict[str, int | str] = {"id": installation_id}
        if repository_selection is not None:
            org_installation_payload["repository_selection"] = repository_selection
        router.get(f"https://api.github.com/orgs/{owner}/installation").mock(
            return_value=respx.MockResponse(200, json=org_installation_payload)
        )
    else:
        router.get(f"https://api.github.com/orgs/{owner}/installation").mock(
            return_value=respx.MockResponse(404)
        )
    if installation_id and not is_org:
        router.get(f"https://api.github.com/repos/{owner}/{repo}/installation").mock(
            return_value=respx.MockResponse(200, json={"id": installation_id})
        )
    else:
        router.get(f"https://api.github.com/repos/{owner}/{repo}/installation").mock(
            return_value=respx.MockResponse(404)
        )

    if installation_id:
        router.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens"
        ).mock(
            return_value=respx.MockResponse(
                201, json={"token": f"token-{installation_id}"}
            )
        )


def mock_pat_access(
    router: respx.Router, owner: str, repo: str, pat: str, accessible: bool
) -> None:
    router.get(
        f"https://api.github.com/repos/{owner}/{repo}",
        headers={"Authorization": f"token {pat}"},
    ).mock(
        return_value=respx.MockResponse(
            200 if accessible else 404,
            json={"name": repo} if accessible else {"message": "Not Found"},
        )
    )


@pytest.mark.asyncio
async def test_public_github_repo(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    """Test validation of public GitHub repository."""
    mock_github_repo_and_owner(
        mock_github_api, owner="public", repo="repo", repo_private=False
    )

    # Override default to simulate public probe success via git
    with patch(f"{GIT_SERVICE}.validate_git_public_access", return_value=True):
        result = await service.validate_repository(
            "https://github.com/public/repo", project_id=project_id
        )

    assert result.accessible is True
    assert "public repository" in result.message
    # App name and installation URL are always returned for GitHub repos when GitHub App is configured
    assert result.github_app_name == "TestApp"
    assert (
        result.github_app_installation_url
        == "https://github.com/apps/TestApp/installations/new/permissions?target_id=12345"
    )


@pytest.mark.asyncio
async def test_private_github_repo_with_app_access(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    """Test private repo with GitHub App installation."""
    mock_github_repo_and_owner(
        mock_github_api,
        owner="private",
        repo="app-repo",
        repo_private=True,
        installation_id=12345,
    )

    result = await service.validate_repository(
        "https://github.com/private/app-repo", project_id=project_id
    )

    assert result.accessible is True
    assert "GitHub App installation" in result.message
    assert result.github_app_name == "TestApp"
    assert (
        result.github_app_installation_url
        == "https://github.com/apps/TestApp/installations/new/permissions?target_id=12345"
    )
    # Settings URL for user account
    assert (
        result.github_app_settings_url
        == "https://github.com/settings/installations/12345"
    )


@pytest.mark.asyncio
async def test_private_github_repo_with_valid_pat(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    mock_pat_access(mock_github_api, "private", "pat-valid-repo", "valid_token", True)
    mock_github_repo_and_owner(
        mock_github_api,
        owner="private",
        repo="pat-valid-repo",
        repo_private=True,
    )

    result = await service.validate_repository(
        "https://github.com/private/pat-valid-repo",
        project_id=project_id,
        pat="valid_token",
    )

    assert result.accessible is True
    assert "Personal Access Token" in result.message
    # App name and installation URL are always returned for GitHub repos when GitHub App is configured
    assert result.github_app_name == "TestApp"
    assert (
        result.github_app_installation_url
        == "https://github.com/apps/TestApp/installations/new/permissions?target_id=12345"
    )


@pytest.mark.asyncio
async def test_private_github_repo_with_invalid_pat(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    """Test private repo with invalid PAT."""
    mock_pat_access(
        mock_github_api, "private", "pat-invalid-repo", "invalid_token", False
    )
    mock_github_repo_and_owner(
        mock_github_api,
        owner="private",
        repo="pat-invalid-repo",
        repo_private=True,
    )

    result = await service.validate_repository(
        "https://github.com/private/pat-invalid-repo",
        project_id=project_id,
        pat="invalid_token",
    )

    assert result.accessible is False
    assert "does not have access" in result.message
    assert result.github_app_name == "TestApp"


@pytest.mark.asyncio
async def test_inaccessible_github_repo(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    """Test completely inaccessible GitHub repository."""

    mock_github_api.get("https://api.github.com/users/inaccessible").mock(
        return_value=respx.MockResponse(
            200, json={"id": 12345, "login": "inaccessible", "type": "User"}
        )
    )

    mock_github_api.get("https://api.github.com/repos/inaccessible/repo").mock(
        return_value=respx.MockResponse(404)
    )
    mock_github_api.get(
        "https://api.github.com/repos/inaccessible/repo/installation"
    ).mock(return_value=respx.MockResponse(404))
    mock_github_api.get("https://api.github.com/orgs/inaccessible/installation").mock(
        return_value=respx.MockResponse(404)
    )

    result = await service.validate_repository(
        "https://github.com/inaccessible/repo", project_id=project_id
    )

    assert result.accessible is False
    assert "Unable to access GitHub repository 'inaccessible/repo'" in result.message
    assert result.github_app_name == "TestApp"
    assert (
        result.github_app_installation_url
        == "https://github.com/apps/TestApp/installations/new/permissions?target_id=12345"
    )


@pytest.mark.asyncio
async def test_public_generic_repo(
    service: GitService, project_id: str, github_app_auth: GitHubAppAuth
) -> None:
    """Test validation of public non-GitHub repository."""
    with patch(f"{GIT_SERVICE}.validate_git_public_access", return_value=True):
        result = await service.validate_repository(
            "https://gitlab.com/public/repo.git", project_id=project_id
        )

        assert result.accessible is True
        assert "public repository" in result.message
        assert result.github_app_name is None  # No GitHub App for non-GitHub repos


@pytest.mark.asyncio
async def test_private_generic_repo_with_valid_pat(
    service: GitService, project_id: str, github_app_auth: GitHubAppAuth
) -> None:
    """Test private non-GitHub repo with valid credentials."""
    with (
        patch(f"{GIT_SERVICE}.validate_git_public_access", return_value=False),
        patch(f"{GIT_SERVICE}.validate_git_credential_access", return_value=True),
    ):
        result = await service.validate_repository(
            "https://gitlab.com/private/valid-pat-repo.git",
            project_id=project_id,
            pat="valid_token",
        )

        assert result.accessible is True
        assert "Personal Access Token" in result.message
        assert result.github_app_installation_url is None


@pytest.mark.asyncio
async def test_private_generic_repo_with_invalid_pat(
    service: GitService, project_id: str, github_app_auth: GitHubAppAuth
) -> None:
    """Test private non-GitHub repo with invalid credentials."""
    with (
        patch(f"{GIT_SERVICE}.validate_git_public_access", return_value=False),
        patch(f"{GIT_SERVICE}.validate_git_credential_access", return_value=False),
    ):
        result = await service.validate_repository(
            "https://gitlab.com/private/invalid-pat-repo.git",
            project_id=project_id,
            pat="invalid_token",
        )

        assert result.accessible is False
        assert "does not have access" in result.message
        assert result.github_app_installation_url is None


@pytest.mark.asyncio
async def test_inaccessible_generic_repo(
    service: GitService, project_id: str, github_app_auth: GitHubAppAuth
) -> None:
    """Test completely inaccessible non-GitHub repository."""
    with patch(f"{GIT_SERVICE}.validate_git_public_access", return_value=False):
        result = await service.validate_repository(
            "https://gitlab.com/inaccessible/repo.git", project_id=project_id
        )

        assert result.accessible is False
        assert "private or does not exist" in result.message


@pytest.mark.asyncio
async def test_github_detection(
    service: GitService, github_app_auth: GitHubAppAuth
) -> None:
    """Test GitHub repository detection."""
    assert service._is_github_repository("https://github.com/owner/repo") is True
    assert service._is_github_repository("git@github.com:owner/repo.git") is True
    assert service._is_github_repository("github.com/owner/repo") is True


@pytest.mark.asyncio
async def test_non_github_detection(
    service: GitService, github_app_auth: GitHubAppAuth
) -> None:
    """Test non-GitHub repository detection."""
    assert service._is_github_repository("https://gitlab.com/owner/repo") is False
    assert service._is_github_repository("https://bitbucket.org/owner/repo") is False
    assert service._is_github_repository("https://git.example.com/repo") is False


@pytest.mark.asyncio
async def test_is_github_repository_rejects_spoofed_urls(
    service: GitService, github_app_auth: GitHubAppAuth
) -> None:
    """Test that URLs with github.com in the path but not the host are rejected."""
    assert (
        service._is_github_repository("https://evil.com/github.com/owner/repo") is False
    )
    assert service._is_github_repository("https://evil.com?github.com") is False
    assert service._is_github_repository("https://notgithub.com/owner/repo") is False


@pytest.mark.asyncio
async def test_existing_deployment_with_pat(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    """Test using existing deployment's PAT."""
    with patch(
        f"{GIT_SERVICE}.k8s_client.get_deployment_pat",
        return_value="existing_token",
    ):
        mock_pat_access(
            mock_github_api, "existing", "deployment-repo", "existing_token", True
        )
        mock_github_repo_and_owner(
            mock_github_api,
            owner="existing",
            repo="deployment-repo",
            repo_private=True,
        )
        result = await service.validate_repository(
            "https://github.com/existing/deployment-repo",
            project_id=project_id,
            deployment_id="test-deployment",
        )

        assert result.accessible is True
        assert "existing Personal Access Token" in result.message
        # App name and installation URL are always returned for GitHub repos when GitHub App is configured
        assert result.github_app_name == "TestApp"
        assert (
            result.github_app_installation_url
            == "https://github.com/apps/TestApp/installations/new/permissions?target_id=12345"
        )


@pytest.mark.asyncio
async def test_pat_obsolete_detection(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    """Test PAT obsolescence detection."""
    with patch(
        f"{GIT_SERVICE}.k8s_client.get_deployment_pat",
        return_value="existing_token",
    ):
        mock_github_repo_and_owner(
            mock_github_api,
            owner="obsolete",
            repo="pat-repo",
            repo_private=False,
        )
        result = await service.validate_repository(
            "https://github.com/obsolete/pat-repo",
            project_id=project_id,
            deployment_id="test-deployment-obsolete",
        )

        assert result.accessible is True
        assert result.pat_is_obsolete is True
        # App name and installation URL are always returned for GitHub repos when GitHub App is configured
        assert result.github_app_name == "TestApp"
        assert (
            result.github_app_installation_url
            == "https://github.com/apps/TestApp/installations/new/permissions?target_id=12345"
        )


@pytest.mark.asyncio
async def test_invalid_github_repository_url_single_segment(
    service: GitService,
    project_id: str,
) -> None:
    """Ensure we surface a helpful error when repository path lacks a repo segment."""
    result = await service.validate_repository(
        "https://github.com/just-owner", project_id=project_id
    )

    assert result.accessible is False
    assert "Invalid GitHub repository URL" in result.message


@pytest.mark.asyncio
async def test_github_owner_missing(
    service: GitService,
    project_id: str,
    mock_github_api: respx.Router,
) -> None:
    """Return a precise error when the GitHub owner cannot be found."""
    mock_github_api.get("https://api.github.com/users/missing-owner").mock(
        return_value=respx.MockResponse(404)
    )
    mock_github_api.get("https://api.github.com/repos/missing-owner/repo").mock(
        return_value=respx.MockResponse(404)
    )

    result = await service.validate_repository(
        "https://github.com/missing-owner/repo", project_id=project_id
    )

    assert result.accessible is False
    assert "GitHub owner 'missing-owner' does not exist" in result.message


@pytest.mark.asyncio
async def test_github_org_installation_all_repos_accessible(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    """Org-wide installation with all repositories should authenticate successfully."""
    mock_github_repo_and_owner(
        mock_github_api,
        owner="org-owner",
        repo="visible-repo",
        repo_private=True,
        repo_accessible=True,
        owner_exists=True,
        installation_id=500,
        is_org=True,
        repository_selection="all",
    )

    result = await service.validate_repository(
        "https://github.com/org-owner/visible-repo", project_id=project_id
    )

    assert result.accessible is True
    assert "GitHub App installation" in result.message
    assert result.github_app_name == "TestApp"
    assert (
        result.github_app_installation_url
        == "https://github.com/apps/TestApp/installations/new/permissions?target_id=12345"
    )
    # Settings URL for org account
    assert (
        result.github_app_settings_url
        == "https://github.com/organizations/org-owner/settings/installations/500"
    )


@pytest.mark.asyncio
async def test_github_org_installation_repo_missing(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    """Org-wide GitHub App installs should not mark non-existent repos as accessible."""
    mock_github_repo_and_owner(
        mock_github_api,
        owner="org-owner",
        repo="ghost-repo",
        repo_private=True,
        owner_exists=True,
        installation_id=999,
        is_org=True,
        repository_selection="all",
    )

    result = await service.validate_repository(
        "https://github.com/org-owner/ghost-repo", project_id=project_id
    )

    assert result.accessible is False
    assert "GitHub repository 'org-owner/ghost-repo' does not exist" in result.message
    assert result.github_app_name is None
    assert result.github_app_installation_url is None
    assert result.github_app_settings_url is None


@pytest.mark.asyncio
async def test_github_org_installation_selected_repos_without_access(
    service: GitService,
    project_id: str,
    github_app_auth: GitHubAppAuth,
    mock_github_api: respx.Router,
) -> None:
    """Provide targeted guidance when the GitHub App installation omits the requested repo."""
    mock_github_repo_and_owner(
        mock_github_api,
        owner="org-owner",
        repo="restricted-repo",
        repo_private=True,
        owner_exists=True,
        installation_id=1001,
        is_org=True,
        repository_selection="selected",
    )

    result = await service.validate_repository(
        "https://github.com/org-owner/restricted-repo", project_id=project_id
    )

    assert result.accessible is False
    assert "does not currently include 'org-owner/restricted-repo'" in result.message
    assert result.github_app_name == "TestApp"
    assert (
        result.github_app_installation_url
        == "https://github.com/apps/TestApp/installations/new/permissions?target_id=12345"
    )
