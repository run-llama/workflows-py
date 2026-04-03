import socket
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from llama_agents.control_plane.build_api.build_app import (
    _validate_git_path,
    _validate_url_not_private,
)

# -- git path validation --


@pytest.mark.parametrize(
    "path",
    [
        "info/refs",
        "HEAD",
        "objects/pack/pack-abc123.pack",
        "git-upload-pack",
        "git-receive-pack",
    ],
)
def test_validate_git_path_allows_valid_paths(path: str) -> None:
    _validate_git_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "../../etc/passwd",
        "some/random/path",
    ],
)
def test_validate_git_path_rejects_invalid_paths(path: str) -> None:
    with pytest.raises(HTTPException) as exc:
        _validate_git_path(path)
    assert exc.value.status_code == 400


# -- URL SSRF validation --


@pytest.mark.parametrize(
    ("ip", "family", "addr"),
    [
        ("127.0.0.1", socket.AF_INET, ("127.0.0.1", 0)),
        ("10.0.0.1", socket.AF_INET, ("10.0.0.1", 0)),
        ("172.16.0.1", socket.AF_INET, ("172.16.0.1", 0)),
        ("192.168.1.1", socket.AF_INET, ("192.168.1.1", 0)),
        ("169.254.169.254", socket.AF_INET, ("169.254.169.254", 0)),
        ("::1", socket.AF_INET6, ("::1", 0, 0, 0)),
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
def test_validate_url_blocks_private_ips(
    ip: str, family: socket.AddressFamily, addr: tuple[str, ...]
) -> None:
    with patch(
        "socket.getaddrinfo",
        return_value=[(family, 0, 0, "", addr)],
    ):
        with pytest.raises(HTTPException) as exc:
            _validate_url_not_private("http://example.com/repo.git")
        assert exc.value.status_code == 403


def test_validate_url_allows_public_ip() -> None:
    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, 0, 0, "", ("140.82.121.3", 0))],
    ):
        _validate_url_not_private("https://github.com/owner/repo.git")


def test_validate_url_blocks_dns_failure() -> None:
    with patch(
        "socket.getaddrinfo", side_effect=socket.gaierror("Name resolution failed")
    ):
        with pytest.raises(HTTPException) as exc:
            _validate_url_not_private("http://nonexistent.example.com/repo.git")
        assert exc.value.status_code == 400


def test_validate_url_blocks_no_hostname() -> None:
    with pytest.raises(HTTPException) as exc:
        _validate_url_not_private("not-a-url")
    assert exc.value.status_code == 400
