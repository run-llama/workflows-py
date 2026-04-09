# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import socket
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dulwich import porcelain
from dulwich.errors import HangupException
from dulwich.refs import Ref
from dulwich.repo import Repo
from llama_agents.core.git.git_util import (
    GitAccessError,
    GitCloneResult,
    _checkout_ref,
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

GIT_UTIL = "llama_agents.core.git.git_util"


def _make_fake_repo(head_sha: str = "abc123def456" * 3 + "0000") -> MagicMock:
    """Build a fake dulwich Repo whose ``.head()`` returns ``head_sha``."""
    fake_repo = MagicMock()
    fake_repo.head.return_value = head_sha.encode()
    fake_repo.refs.read_ref.return_value = b"ref: refs/heads/main"
    fake_repo.close = MagicMock()
    return fake_repo


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


@patch(f"{GIT_UTIL}.porcelain.clone")
@pytest.mark.asyncio
async def test_clone_repo_branch_success(mock_clone: MagicMock) -> None:
    """clone_repo with a branch ref returns the resolved SHA and ref."""
    mock_clone.return_value = _make_fake_repo("a" * 40)

    with tempfile.TemporaryDirectory() as t:
        result = await clone_repo(
            "https://github.com/user/repo.git", "main", dest_dir=Path(t) / "sub"
        )

    assert result == GitCloneResult(git_sha="a" * 40, git_ref="main")
    assert mock_clone.call_count == 1
    call = mock_clone.call_args
    # branch is passed through as bytes
    assert call.kwargs["branch"] == b"main"
    assert call.kwargs["source"] == "https://github.com/user/repo.git"
    assert call.kwargs["depth"] is None


@patch(f"{GIT_UTIL}.porcelain.clone")
@pytest.mark.asyncio
async def test_clone_repo_no_ref(mock_clone: MagicMock) -> None:
    """clone_repo with no ref resolves the remote default branch."""
    mock_clone.return_value = _make_fake_repo("b" * 40)

    with tempfile.TemporaryDirectory() as t:
        result = await clone_repo(
            "https://github.com/user/repo.git", dest_dir=Path(t) / "sub"
        )

    assert result == GitCloneResult(git_sha="b" * 40, git_ref="main")
    assert mock_clone.call_args.kwargs["branch"] is None


@patch(f"{GIT_UTIL}.porcelain.clone")
@pytest.mark.asyncio
async def test_clone_repo_no_ref_detached_resolves_tag(mock_clone: MagicMock) -> None:
    """A detached HEAD that matches a tag returns the tag name as the ref."""
    fake_repo = MagicMock()
    head_sha = b"c" * 40
    fake_repo.head.return_value = head_sha
    # Symbolic HEAD reads as the SHA itself when detached
    fake_repo.refs.read_ref.return_value = head_sha
    fake_repo.refs.__getitem__.return_value = head_sha
    fake_repo.refs.subkeys.return_value = [b"v1.2.3"]
    mock_clone.return_value = fake_repo

    with tempfile.TemporaryDirectory() as t:
        result = await clone_repo(
            "https://github.com/user/repo.git", dest_dir=Path(t) / "sub"
        )

    assert result == GitCloneResult(git_sha=head_sha.decode(), git_ref="v1.2.3")


@patch(f"{GIT_UTIL}.porcelain.clone")
@pytest.mark.asyncio
async def test_clone_repo_full_sha(mock_clone: MagicMock) -> None:
    """A 40-char SHA is fetched as the default ref then checked out."""
    sha = "deadbeef" * 5  # 40 chars
    fake_repo = MagicMock()
    fake_repo.head.return_value = sha.encode()
    fake_repo.refs.read_ref.return_value = b"ref: refs/heads/main"
    fake_repo.refs.__getitem__.return_value = sha.encode()
    mock_clone.return_value = fake_repo

    with patch(f"{GIT_UTIL}._checkout_ref") as mock_checkout:
        with tempfile.TemporaryDirectory() as t:
            result = await clone_repo(
                "https://github.com/user/repo.git",
                git_ref=sha,
                dest_dir=Path(t) / "sub",
            )

    # branch=None when ref is a SHA — let dulwich pick the default branch first
    assert mock_clone.call_args.kwargs["branch"] is None
    mock_checkout.assert_called_once()
    assert result.git_sha == sha
    assert result.git_ref == sha


@patch(f"{GIT_UTIL}.porcelain.clone")
@pytest.mark.asyncio
async def test_clone_repo_short_sha_like_ref(mock_clone: MagicMock) -> None:
    """Short SHA-like refs are cloned without branch= and checked out after clone."""
    abbrev_sha = "deadbeef"
    fake_repo = _make_fake_repo("a" * 40)
    mock_clone.return_value = fake_repo

    with patch(f"{GIT_UTIL}._checkout_ref") as mock_checkout:
        with tempfile.TemporaryDirectory() as t:
            result = await clone_repo(
                "https://github.com/user/repo.git",
                git_ref=abbrev_sha,
                dest_dir=Path(t) / "sub",
                depth=1,
            )

    assert mock_clone.call_args.kwargs["branch"] is None
    assert mock_clone.call_args.kwargs["depth"] is None
    mock_checkout.assert_called_once_with(fake_repo, abbrev_sha)
    assert result == GitCloneResult(git_sha="a" * 40, git_ref=abbrev_sha)


@patch(f"{GIT_UTIL}.porcelain.clone")
@pytest.mark.asyncio
async def test_clone_repo_network_error(mock_clone: MagicMock) -> None:
    """Network errors from dulwich are normalized to GitAccessError."""
    mock_clone.side_effect = HangupException()

    with tempfile.TemporaryDirectory() as t:
        with pytest.raises(GitAccessError, match="Failed to clone"):
            await clone_repo(
                "https://github.com/user/repo.git", "main", dest_dir=Path(t) / "sub"
            )


@patch(f"{GIT_UTIL}.porcelain.clone")
@pytest.mark.asyncio
async def test_clone_repo_with_basic_auth(mock_clone: MagicMock) -> None:
    """Basic auth gets parsed and forwarded to dulwich as username/password."""
    mock_clone.return_value = _make_fake_repo("e" * 40)

    with tempfile.TemporaryDirectory() as t:
        await clone_repo(
            "https://github.com/user/repo.git",
            "main",
            basic_auth="someuser:tokenvalue",
            dest_dir=Path(t) / "sub",
        )

    call = mock_clone.call_args
    assert call.kwargs["username"] == "someuser"
    assert call.kwargs["password"] == "tokenvalue"


@patch(f"{GIT_UTIL}.porcelain.clone")
@pytest.mark.asyncio
async def test_clone_repo_passes_depth(mock_clone: MagicMock) -> None:
    """Callers can request a shallow clone via the depth parameter."""
    mock_clone.return_value = _make_fake_repo("f" * 40)

    with tempfile.TemporaryDirectory() as t:
        await clone_repo(
            "https://github.com/user/repo.git",
            "main",
            dest_dir=Path(t) / "sub",
            depth=1,
        )

    assert mock_clone.call_args.kwargs["depth"] == 1


@patch(f"{GIT_UTIL}.porcelain.clone")
@pytest.mark.asyncio
async def test_clone_repo_sha_overrides_depth(mock_clone: MagicMock) -> None:
    """When given a SHA-shaped ref, depth is dropped to ensure reachability."""
    sha = "deadbeef" * 5  # 40 chars
    fake_repo = MagicMock()
    fake_repo.head.return_value = sha.encode()
    fake_repo.refs.read_ref.return_value = b"ref: refs/heads/main"
    fake_repo.refs.__getitem__.return_value = sha.encode()
    mock_clone.return_value = fake_repo

    with patch(f"{GIT_UTIL}._checkout_ref"):
        with tempfile.TemporaryDirectory() as t:
            await clone_repo(
                "https://github.com/user/repo.git",
                git_ref=sha,
                dest_dir=Path(t) / "sub",
                depth=1,
            )

    assert mock_clone.call_args.kwargs["depth"] is None
    # branch is None for a SHA ref since we can't pass a SHA as a branch
    assert mock_clone.call_args.kwargs["branch"] is None


def _make_real_repo_with_commit(tmp_path: Path) -> tuple[Repo, str]:
    porcelain.init(str(tmp_path))
    (tmp_path / "f.txt").write_text("hello")
    porcelain.add(str(tmp_path), [str(tmp_path / "f.txt")])
    sha = porcelain.commit(
        str(tmp_path),
        message=b"init",
        author=b"t <t@example.com>",
        committer=b"t <t@example.com>",
    )
    return Repo(str(tmp_path)), sha.decode()


def test_checkout_ref_resolves_short_sha_prefix(tmp_path: Path) -> None:
    """Short SHA prefixes resolve to the matching commit object."""
    repo, sha = _make_real_repo_with_commit(tmp_path)
    try:
        _checkout_ref(repo, sha[:8])
        assert repo.head().decode() == sha
    finally:
        repo.close()


def test_checkout_ref_rejects_ambiguous_short_sha_prefix(tmp_path: Path) -> None:
    """Ambiguous short SHA prefixes raise a user-facing access error."""
    repo, _ = _make_real_repo_with_commit(tmp_path)
    try:
        with patch.object(
            repo.object_store,
            "iter_prefix",
            return_value=iter([b"a" * 40, b"b" * 40]),
        ):
            with pytest.raises(GitAccessError, match="ambiguous"):
                _checkout_ref(repo, "deadbeef")
    finally:
        repo.close()


def test_checkout_ref_rejects_missing_short_sha_prefix(tmp_path: Path) -> None:
    """Missing short SHA prefixes raise a user-facing access error."""
    repo, _ = _make_real_repo_with_commit(tmp_path)
    try:
        with pytest.raises(GitAccessError, match="Git ref not found: deadbeef"):
            _checkout_ref(repo, "deadbeef")
    finally:
        repo.close()


@pytest.mark.asyncio
async def test_clone_repo_rejects_dangerous_url() -> None:
    with pytest.raises(GitAccessError):
        await clone_repo("ext::sh -c echo pwned")


# Lightweight tests for new git helpers


@patch(f"{GIT_UTIL}.porcelain.status")
@patch(f"{GIT_UTIL}.Path.cwd")
def test_working_tree_has_changes_true(_cwd: MagicMock, mock_status: MagicMock) -> None:
    mock_status.return_value = MagicMock(
        staged={"add": [], "delete": [], "modify": ["file.py"]},
        unstaged=[],
        untracked=[],
    )
    assert working_tree_has_changes() is True


@patch(f"{GIT_UTIL}.porcelain.status")
@patch(f"{GIT_UTIL}.Path.cwd")
def test_working_tree_has_changes_false(
    _cwd: MagicMock, mock_status: MagicMock
) -> None:
    mock_status.return_value = MagicMock(
        staged={"add": [], "delete": [], "modify": []},
        unstaged=[],
        untracked=[],
    )
    assert working_tree_has_changes() is False


@patch(f"{GIT_UTIL}.porcelain.status", side_effect=Exception("boom"))
def test_working_tree_has_changes_exception(_: MagicMock) -> None:
    assert working_tree_has_changes() is False


def test_get_unpushed_commits_count_no_upstream(tmp_path: Path) -> None:
    """A fresh repo with no upstream returns None."""
    porcelain.init(str(tmp_path))
    # Create a single commit so HEAD resolves
    (tmp_path / "f.txt").write_text("hello")
    porcelain.add(str(tmp_path), [str(tmp_path / "f.txt")])
    porcelain.commit(
        str(tmp_path),
        message=b"init",
        author=b"t <t@example.com>",
        committer=b"t <t@example.com>",
    )

    with patch(f"{GIT_UTIL}.Path.cwd", return_value=tmp_path):
        assert get_unpushed_commits_count() is None


def test_get_unpushed_commits_count_ahead(tmp_path: Path) -> None:
    """When the local branch is ahead of its tracking ref, count the diff."""
    porcelain.init(str(tmp_path))
    (tmp_path / "f.txt").write_text("first")
    porcelain.add(str(tmp_path), [str(tmp_path / "f.txt")])
    first_sha = porcelain.commit(
        str(tmp_path),
        message=b"first",
        author=b"t <t@example.com>",
        committer=b"t <t@example.com>",
    )

    # Set up an "upstream" tracking ref pointing at the first commit
    repo = Repo(str(tmp_path))
    try:
        repo.refs[Ref(b"refs/remotes/origin/main")] = first_sha  # type: ignore[index]  # ty: ignore[invalid-assignment]
        # Configure branch.<current>.merge / .remote
        config = repo.get_config()
        head = repo.refs.read_ref(Ref(b"HEAD"))
        assert head is not None and head.startswith(b"ref: ")
        branch_name = head[5:].removeprefix(b"refs/heads/")
        config.set((b"branch", branch_name), b"merge", b"refs/heads/main")
        config.set((b"branch", branch_name), b"remote", b"origin")
        config.write_to_path()
    finally:
        repo.close()

    # Add two more commits
    (tmp_path / "f.txt").write_text("second")
    porcelain.add(str(tmp_path), [str(tmp_path / "f.txt")])
    porcelain.commit(
        str(tmp_path),
        message=b"second",
        author=b"t <t@example.com>",
        committer=b"t <t@example.com>",
    )
    (tmp_path / "f.txt").write_text("third")
    porcelain.add(str(tmp_path), [str(tmp_path / "f.txt")])
    porcelain.commit(
        str(tmp_path),
        message=b"third",
        author=b"t <t@example.com>",
        committer=b"t <t@example.com>",
    )

    with patch(f"{GIT_UTIL}.Path.cwd", return_value=tmp_path):
        assert get_unpushed_commits_count() == 2


def test_get_unpushed_commits_count_no_repo(tmp_path: Path) -> None:
    """A non-git directory returns 0 (legacy behavior)."""
    with patch(f"{GIT_UTIL}.Path.cwd", return_value=tmp_path):
        assert get_unpushed_commits_count() == 0


# Tests for validate_git_public_access


@patch(f"{GIT_UTIL}._probe_remote")
@pytest.mark.asyncio
async def test_validate_git_public_access_true(mock_probe: MagicMock) -> None:
    mock_probe.return_value = True
    with patch(f"{GIT_UTIL}.validate_git_url_no_ssrf"):
        assert (
            await validate_git_public_access("https://github.com/public/repo.git")
            is True
        )
    mock_probe.assert_called_once_with("https://github.com/public/repo.git")


@patch(f"{GIT_UTIL}._probe_remote")
@pytest.mark.asyncio
async def test_validate_git_public_access_false(mock_probe: MagicMock) -> None:
    mock_probe.return_value = False
    with patch(f"{GIT_UTIL}.validate_git_url_no_ssrf"):
        assert (
            await validate_git_public_access("https://github.com/private/repo.git")
            is False
        )


def test_get_commit_sha_for_ref(tmp_path: Path) -> None:
    """Resolve a branch ref against a real (test) repo."""
    porcelain.init(str(tmp_path))
    (tmp_path / "f.txt").write_text("hello")
    porcelain.add(str(tmp_path), [str(tmp_path / "f.txt")])
    sha = porcelain.commit(
        str(tmp_path),
        message=b"init",
        author=b"t <t@example.com>",
        committer=b"t <t@example.com>",
    )
    with patch(f"{GIT_UTIL}.Path.cwd", return_value=tmp_path):
        # HEAD resolves
        assert get_commit_sha_for_ref("HEAD") == sha.decode()


def test_is_git_repo_returns_false_when_not_a_repo(tmp_path: Path) -> None:
    """In a directory that is not a git repo, return False."""
    with patch(f"{GIT_UTIL}.Path.cwd", return_value=tmp_path):
        assert is_git_repo() is False


def test_is_git_repo_returns_true_in_real_repo(tmp_path: Path) -> None:
    porcelain.init(str(tmp_path))
    with patch(f"{GIT_UTIL}.Path.cwd", return_value=tmp_path):
        assert is_git_repo() is True


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
