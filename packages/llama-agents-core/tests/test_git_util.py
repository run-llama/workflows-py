import socket
import subprocess
from unittest.mock import MagicMock, Mock, patch

import pytest
from llama_agents.core.git.git_util import (
    GitAccessError,
    GitCloneResult,
    clone_repo,
    get_commit_sha_for_ref,
    get_unpushed_commits_count,
    is_git_repo,
    parse_github_repo_url,
    validate_git_public_access,
    validate_git_url,
    validate_git_url_no_ssrf,
    working_tree_has_changes,
)


def emulate_successful_clone() -> Mock:
    return Mock(
        returncode=0,
        stdout="Cloning into 'repo'...\n Resolving deltas: 100% (2/2), done.\n",
        stderr="",
    )


def emulate_failed_clone() -> Mock:
    return Mock(
        returncode=1,
        stdout="",
        stderr="fatal: could not read Username for 'https://github.com': No such file or directory",
    )


def emulate_successful_checkout(ref: str = "main") -> Mock:
    return Mock(
        returncode=0,
        stdout=f"Switched to branch '{ref}'\nYour branch is up to date with 'origin/{ref}'.\n",
        stderr="",
    )


def emulate_failed_checkout(ref: str) -> Mock:
    return Mock(
        returncode=1,
        stdout="",
        stderr=f"error: pathspec '{ref}' did not match any file(s) known to git",
    )


def emulate_successful_rev_parse(sha: str) -> Mock:
    return Mock(returncode=0, stdout=f"{sha}\n", stderr="")


def emulate_successful_fetch() -> Mock:
    return Mock(returncode=0, stdout="", stderr="")


def emulate_failed_fetch() -> Mock:
    return Mock(returncode=1, stdout="", stderr="fatal: unable to access repository")


def test_parse_github_repo_url() -> None:
    """Test GitHub URL parsing with various formats."""
    # Standard HTTPS format
    assert parse_github_repo_url("https://github.com/owner/repo") == ("owner", "repo")

    # HTTPS with .git suffix
    assert parse_github_repo_url("https://github.com/owner/repo.git") == (
        "owner",
        "repo",
    )

    # SSH format
    assert parse_github_repo_url("git@github.com:owner/repo") == ("owner", "repo")
    assert parse_github_repo_url("git@github.com:owner/repo.git") == ("owner", "repo")

    # Without protocol prefix
    assert parse_github_repo_url("github.com/owner/repo") == ("owner", "repo")

    # With trailing slashes
    assert parse_github_repo_url("https://github.com/owner/repo/") == ("owner", "repo")
    assert parse_github_repo_url("https://github.com/owner/repo.git/") == (
        "owner",
        "repo",
    )

    # Edge cases that should fail
    with pytest.raises(ValueError, match="Could not parse GitHub repository URL"):
        parse_github_repo_url("not-a-github-url")

    with pytest.raises(ValueError):
        parse_github_repo_url("https://gitlab.com/owner/repo")

    with pytest.raises(ValueError):
        parse_github_repo_url("https://github.com/")

    with pytest.raises(ValueError):
        parse_github_repo_url("https://github.com/owner")


# Git reference resolution tests moved from test_k8s_client.py
@patch("subprocess.run")
def test_clone_repo_branch_success(mock_run: MagicMock) -> None:
    """Test successful branch resolution"""
    mock_run.side_effect = [
        emulate_successful_clone(),
        emulate_successful_checkout(),
        emulate_successful_rev_parse("abc123def456"),
    ]

    result = clone_repo("https://github.com/user/repo.git", "main")
    assert result == GitCloneResult(git_sha="abc123def456", git_ref="main")

    assert mock_run.call_count == 3
    one, two, three = mock_run.call_args_list
    assert one.args[0][:4] == ["git", "clone", "--", "https://github.com/user/repo.git"]
    assert len(one.args[0]) == 5
    assert "/" in one.args[0][4]  # temp directory
    assert two.args == (["git", "checkout", "main", "--"],)
    assert three.args == (["git", "rev-parse", "HEAD"],)


@patch("subprocess.run")
def test_clone_repo_no_ref(mock_run: MagicMock) -> None:
    """Test successful branch resolution with PAT"""
    mock_run.side_effect = [
        emulate_successful_clone(),
        Mock(returncode=0, stdout="", stderr=""),  # no branch
        Mock(returncode=0, stdout="v1.1.1"),  # on a tag
        emulate_successful_rev_parse("abc123def456"),
    ]

    result = clone_repo("https://github.com/user/repo.git")
    assert result == GitCloneResult(git_sha="abc123def456", git_ref="v1.1.1")

    assert mock_run.call_count == 4


@patch("subprocess.run")
def test_clone_repo_commit_sha_not_found(mock_run: MagicMock) -> None:
    """Test commit SHA not found in repository surfaces stderr details"""
    mock_run.side_effect = [
        emulate_successful_clone(),
        emulate_failed_checkout("abc123f"),
    ]

    with pytest.raises(GitAccessError) as exc:
        clone_repo("https://github.com/user/repo.git", "abc123f")
    # Error should include the command and stderr pathspec message
    message = str(exc.value)
    assert "git checkout abc123f" in message
    assert "pathspec 'abc123f'" in message


@patch("subprocess.run")
def test_clone_repo_network_error(mock_run: MagicMock) -> None:
    """Test network error surfaces stderr from git command"""
    mock_run.side_effect = [
        emulate_successful_clone(),
        # Simulate checkout failing due to network access error
        Mock(returncode=1, stdout="", stderr="fatal: unable to access repository"),
    ]

    with pytest.raises(GitAccessError) as exc:
        clone_repo("https://github.com/user/repo.git", "main")
    message = str(exc.value)
    assert "git checkout main" in message
    assert "fatal: unable to access repository" in message


@patch("subprocess.run")
def test_clone_repo_timeout(mock_run: MagicMock) -> None:
    """Test timeout during git operation emits enriched timeout message"""
    mock_run.side_effect = subprocess.TimeoutExpired(["git", "clone"], timeout=30)

    with pytest.raises(GitAccessError) as exc:
        clone_repo("https://github.com/user/repo.git", "main")
    message = str(exc.value)
    assert "Command timed out after 30s" in message
    assert "git clone" in message


@patch("subprocess.run")
def test_clone_repo_with_pat(mock_run: MagicMock) -> None:
    """Test git reference resolution with PAT"""
    mock_run.side_effect = [
        # successful clone
        emulate_successful_clone(),
        # successful checkout of main
        emulate_successful_checkout(),
        # successful rev-parse
        emulate_successful_rev_parse("abc123def456"),
    ]

    result = clone_repo("https://github.com/user/repo.git", "main", "ghp_token")
    assert result == GitCloneResult(git_sha="abc123def456", git_ref="main")

    assert mock_run.call_count == 3
    one, two, three = mock_run.call_args_list
    assert one.args[0][:4] == [
        "git",
        "clone",
        "--",
        "https://ghp_token@github.com/user/repo.git",
    ]
    assert len(one.args[0]) == 5
    assert "/" in one.args[0][4]  # temp directory
    assert two.args == (["git", "checkout", "main", "--"],)
    assert three.args == (["git", "rev-parse", "HEAD"],)


# Lightweight tests for new git helpers


@patch("subprocess.run")
def test_working_tree_has_changes_true(mock_run: MagicMock) -> None:
    mock_run.return_value = Mock(returncode=0, stdout=" M file.py\n?? new.txt\n")
    assert working_tree_has_changes() is True


@patch("subprocess.run")
def test_working_tree_has_changes_false(mock_run: MagicMock) -> None:
    mock_run.return_value = Mock(returncode=0, stdout="\n")
    assert working_tree_has_changes() is False


@patch("subprocess.run", side_effect=Exception("boom"))
def test_working_tree_has_changes_exception(_: MagicMock) -> None:
    assert working_tree_has_changes() is False


@patch("subprocess.run")
def test_get_unpushed_commits_count_no_upstream(mock_run: MagicMock) -> None:
    # First call (rev-parse @{u}) fails -> no upstream
    mock_run.return_value = Mock(returncode=1, stdout="", stderr="")
    assert get_unpushed_commits_count() is None


@patch("subprocess.run")
def test_get_unpushed_commits_count_ahead(mock_run: MagicMock) -> None:
    # First call succeeds, second returns behind/ahead counts
    mock_run.side_effect = [
        Mock(returncode=0, stdout="origin/main", stderr=""),
        Mock(returncode=0, stdout="3 2\n", stderr=""),  # behind 3, ahead 2
    ]
    assert get_unpushed_commits_count() == 2


@patch("subprocess.run")
def test_get_unpushed_commits_count_empty_output(mock_run: MagicMock) -> None:
    mock_run.side_effect = [
        Mock(returncode=0, stdout="origin/main", stderr=""),
        Mock(returncode=0, stdout="\n", stderr=""),
    ]
    assert get_unpushed_commits_count() == 0


@patch("subprocess.run", side_effect=Exception("boom"))
def test_get_unpushed_commits_count_exception(_: MagicMock) -> None:
    assert get_unpushed_commits_count() == 0


# New tests for validate_git_public_access


@patch("subprocess.run")
def test_validate_git_public_access_true(mock_run: MagicMock) -> None:
    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
    assert validate_git_public_access("https://github.com/public/repo.git") is True
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][:4] == [
        "git",
        "ls-remote",
        "--heads",
        "--",
    ]


@patch("subprocess.run")
def test_validate_git_public_access_false(mock_run: MagicMock) -> None:
    mock_run.return_value = Mock(returncode=2, stdout="", stderr="fatal")
    assert validate_git_public_access("https://github.com/private/repo.git") is False


@patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["git"], timeout=30))
def test_validate_git_public_access_timeout(_: MagicMock) -> None:
    assert validate_git_public_access("https://github.com/slow/repo.git") is False


@patch("subprocess.run")
def test_get_commit_sha_for_ref(mock_run: MagicMock) -> None:
    mock_run.side_effect = [emulate_successful_rev_parse("abcdefghi123456789")]
    commit_sha = get_commit_sha_for_ref("main")
    assert commit_sha == "abcdefghi123456789"


@patch("llama_agents.core.git.git_util.subprocess.run", side_effect=FileNotFoundError())
def test_is_git_repo_returns_false_when_git_missing(_: MagicMock) -> None:
    # When git executable is missing, we should not crash; return False
    assert is_git_repo() is False


# Tests for validate_git_url


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/repo.git",
        "http://example.com/repo.git",
    ],
)
def test_validate_git_url_allows_valid_schemes(url: str) -> None:
    validate_git_url(url)


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("ext::sh -c echo pwned", id="ext-protocol"),
        pytest.param("-evil-url", id="dash-prefix"),
        pytest.param("git@github.com:owner/repo.git", id="ssh-shorthand"),
        pytest.param("ssh://git@github.com/owner/repo.git", id="ssh-scheme"),
        pytest.param("ftp://example.com/repo", id="unknown-scheme"),
    ],
)
def test_validate_git_url_rejects_dangerous_urls(url: str) -> None:
    with pytest.raises(GitAccessError):
        validate_git_url(url)


@pytest.mark.parametrize(
    ("family", "addr"),
    [
        (socket.AF_INET, ("127.0.0.1", 0)),
        (socket.AF_INET, ("10.0.0.1", 0)),
        (socket.AF_INET, ("172.16.0.1", 0)),
        (socket.AF_INET, ("192.168.1.1", 0)),
        (socket.AF_INET, ("169.254.169.254", 0)),
        (socket.AF_INET6, ("::1", 0, 0, 0)),
    ],
    ids=[
        "loopback",
        "private-10",
        "private-172",
        "private-192",
        "link-local",
        "ipv6-loopback",
    ],
)
def test_validate_git_url_no_ssrf_rejects_private_ips(
    family: socket.AddressFamily, addr: tuple[str, ...]
) -> None:
    with patch(
        "llama_agents.core.git.git_util.socket.getaddrinfo",
        return_value=[(family, 0, 0, "", addr)],
    ):
        with pytest.raises(GitAccessError, match="private network"):
            validate_git_url_no_ssrf("https://example.com/repo.git")


def test_validate_git_url_no_ssrf_allows_public_ip() -> None:
    with patch(
        "llama_agents.core.git.git_util.socket.getaddrinfo",
        return_value=[(socket.AF_INET, 0, 0, "", ("140.82.121.3", 0))],
    ):
        validate_git_url_no_ssrf("https://github.com/owner/repo.git")


def test_validate_git_url_no_ssrf_rejects_dns_failure() -> None:
    with patch(
        "llama_agents.core.git.git_util.socket.getaddrinfo",
        side_effect=socket.gaierror("Name resolution failed"),
    ):
        with pytest.raises(GitAccessError, match="DNS resolution failed"):
            validate_git_url_no_ssrf("https://nonexistent.example.com/repo.git")


def test_clone_repo_rejects_dangerous_url() -> None:
    with pytest.raises(GitAccessError):
        clone_repo("ext::sh -c echo pwned")
