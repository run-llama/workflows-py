from __future__ import annotations

import pytest
from llama_agents.cli.utils.git_push import internal_push_refspec


def test_internal_push_refspec_defaults_to_main() -> None:
    assert internal_push_refspec(None) == ("main", "refs/heads/main")


def test_internal_push_refspec_uses_private_pin_for_commit_sha() -> None:
    sha = "a" * 40
    assert internal_push_refspec(sha) == (sha, f"refs/llamactl/pins/{sha}")


def test_internal_push_refspec_prefers_existing_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llama_agents.cli.utils.git_push.git_ref_exists",
        lambda ref_name: ref_name == "refs/heads/release",
    )
    assert internal_push_refspec("release") == (
        "refs/heads/release",
        "refs/heads/release",
    )


def test_internal_push_refspec_uses_existing_tag_when_branch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llama_agents.cli.utils.git_push.git_ref_exists",
        lambda ref_name: ref_name == "refs/tags/v1.2.3",
    )
    assert internal_push_refspec("v1.2.3") == (
        "refs/tags/v1.2.3",
        "refs/tags/v1.2.3",
    )


def test_internal_push_refspec_preserves_explicit_refs() -> None:
    assert internal_push_refspec("refs/tags/v2.0.0") == (
        "refs/tags/v2.0.0",
        "refs/tags/v2.0.0",
    )
