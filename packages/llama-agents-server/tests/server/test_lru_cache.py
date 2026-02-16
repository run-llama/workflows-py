# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from llama_agents.server._lru_cache import LRUCache


def test_put_and_get() -> None:
    cache: LRUCache[str, int] = LRUCache()
    cache.put("a", 1)
    assert cache.get("a") == 1


def test_get_missing_returns_none() -> None:
    cache: LRUCache[str, int] = LRUCache()
    assert cache.get("missing") is None


def test_delete() -> None:
    cache: LRUCache[str, int] = LRUCache()
    cache.put("a", 1)
    cache.delete("a")
    assert cache.get("a") is None
    assert len(cache) == 0


def test_delete_missing_key_is_noop() -> None:
    cache: LRUCache[str, int] = LRUCache()
    cache.delete("nope")


def test_len() -> None:
    cache: LRUCache[str, int] = LRUCache()
    assert len(cache) == 0
    cache.put("a", 1)
    cache.put("b", 2)
    assert len(cache) == 2


def test_evicts_lru_when_full() -> None:
    cache: LRUCache[str, int] = LRUCache(maxsize=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_get_refreshes_recency() -> None:
    cache: LRUCache[str, int] = LRUCache(maxsize=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # refresh "a"
    cache.put("c", 3)  # should evict "b"
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_put_existing_key_refreshes_recency() -> None:
    cache: LRUCache[str, int] = LRUCache(maxsize=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)  # update and refresh "a"
    cache.put("c", 3)  # should evict "b"
    assert cache.get("a") == 10
    assert cache.get("b") is None
