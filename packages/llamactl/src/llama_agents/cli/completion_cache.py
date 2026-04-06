# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(base) / "llamactl" / "completions"


def _cache_path(resource_type: str, env_hash: str | None = None) -> Path:
    if env_hash:
        return _cache_dir() / f"{resource_type}_{env_hash}.json"
    return _cache_dir() / f"{resource_type}.json"


def _env_hash() -> str | None:
    """Get a short hash of the current environment's api_url for cache scoping."""
    try:
        from llama_agents.cli.config.env_service import service

        env = service.get_current_environment()
        return hashlib.md5(env.api_url.encode()).hexdigest()[:8]
    except Exception:
        return None


def read_cache(resource_type: str, env_hash: str | None = None) -> list[dict[str, Any]]:
    """Read cached completion items. Returns stale data rather than empty.

    For completions, stale data is better than nothing — the user sees
    outdated suggestions rather than a blank list.
    """
    path = _cache_path(resource_type, env_hash)
    try:
        data = json.loads(path.read_text())
        return data.get("items", [])
    except Exception:
        return []


def write_cache(
    resource_type: str, items: list[dict[str, Any]], env_hash: str | None = None
) -> None:
    """Atomically write completion items to cache."""
    path = _cache_path(resource_type, env_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"items": items, "updated_at": time.time()}
    # Atomic write: write to temp file then rename
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def is_stale(
    resource_type: str, env_hash: str | None = None, ttl: float = 300.0
) -> bool:
    """Check if cache is stale (older than ttl seconds). Default TTL: 5 minutes."""
    path = _cache_path(resource_type, env_hash)
    try:
        data = json.loads(path.read_text())
        return (time.time() - data.get("updated_at", 0)) > ttl
    except Exception:
        return True


def refresh_cache(
    resource_type: str,
    fetch_fn: Callable[[], list[dict[str, Any]]],
    env_hash: str | None = None,
    timeout: float = 2.0,
) -> list[dict[str, Any]]:
    """Fetch fresh data and write to cache. Returns items on success, stale cache on timeout/error.

    Uses a thread pool with a hard timeout so slow API calls don't hang the shell.
    The pool is not joined on timeout — the daemon thread dies with the process.
    """
    try:
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fetch_fn)
        items = future.result(timeout=timeout)
        pool.shutdown(wait=False)
        write_cache(resource_type, items, env_hash)
        return items
    except BaseException:
        return read_cache(resource_type, env_hash)
