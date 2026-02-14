# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Simple thread-safe LRU cache backed by OrderedDict."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Bounded LRU cache.

    On ``get`` or ``put`` the accessed key moves to the end (most-recently
    used).  When the cache exceeds *maxsize*, the least-recently used entry
    is evicted.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def delete(self, key: K) -> None:
        self._data.pop(key, None)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._data)
