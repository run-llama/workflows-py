# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import json
import time

import pytest
from llama_agents.cli.completion_cache import (
    _cache_dir,
    _cache_path,
    is_stale,
    read_cache,
    refresh_cache,
    write_cache,
)


@pytest.fixture(autouse=True)
def _patch_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def test_write_then_read() -> None:
    items = [{"id": "deploy-1", "help": "My App — Running"}]
    write_cache("deployments", items, "abc123")
    result = read_cache("deployments", "abc123")
    assert result == items


def test_read_missing_cache_returns_empty() -> None:
    assert read_cache("nonexistent", "xyz") == []


def test_stale_cache_still_returns_data() -> None:
    items = [{"id": "old-deploy"}]
    write_cache("deployments", items, "abc123")
    # Manually set updated_at to the past
    path = _cache_path("deployments", "abc123")
    data = json.loads(path.read_text())
    data["updated_at"] = time.time() - 600
    path.write_text(json.dumps(data))

    # is_stale should be True
    assert is_stale("deployments", "abc123", ttl=300.0)
    # But read still returns data
    assert read_cache("deployments", "abc123") == items


def test_fresh_cache_is_not_stale() -> None:
    write_cache("deployments", [{"id": "x"}], "abc123")
    assert not is_stale("deployments", "abc123", ttl=300.0)


def test_different_env_hashes_dont_collide() -> None:
    write_cache("deployments", [{"id": "a"}], "env1")
    write_cache("deployments", [{"id": "b"}], "env2")
    assert read_cache("deployments", "env1") == [{"id": "a"}]
    assert read_cache("deployments", "env2") == [{"id": "b"}]


def test_xdg_cache_home_respected(tmp_path: object) -> None:
    # The fixture already sets XDG_CACHE_HOME to tmp_path
    d = _cache_dir()
    assert str(tmp_path) in str(d)


def test_refresh_cache_success() -> None:
    items = [{"id": "fresh-deploy"}]
    result = refresh_cache("deployments", lambda: items, "abc123", timeout=5.0)
    assert result == items
    assert read_cache("deployments", "abc123") == items


def test_refresh_cache_timeout_returns_stale() -> None:
    # Pre-populate with stale data
    stale = [{"id": "stale"}]
    write_cache("deployments", stale, "abc123")

    def slow_fetch() -> list[dict[str, str]]:
        import time

        time.sleep(10)
        return [{"id": "never-reached"}]

    result = refresh_cache("deployments", slow_fetch, "abc123", timeout=0.1)
    assert result == stale


def test_refresh_cache_error_returns_stale() -> None:
    stale = [{"id": "stale"}]
    write_cache("deployments", stale, "abc123")

    def failing_fetch() -> list[dict[str, str]]:
        raise SystemExit(1)

    result = refresh_cache("deployments", failing_fetch, "abc123")
    assert result == stale
