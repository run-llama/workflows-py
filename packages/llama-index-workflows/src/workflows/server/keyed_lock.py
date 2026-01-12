# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Keyed lock utility for per-key mutual exclusion with automatic cleanup."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class KeyedLock:
    """A collection of locks keyed by string, with automatic cleanup.

    Locks are created on-demand and automatically removed when no longer
    in use (no waiters or holders). Safe for concurrent asyncio coroutines.

    Usage:
        locks = KeyedLock()

        async with locks("my-key"):
            # critical section for "my-key"
            pass
    """

    def __init__(self) -> None:
        self._main_lock = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._refs: dict[str, int] = {}

    @asynccontextmanager
    async def __call__(self, key: str) -> AsyncIterator[None]:
        """Acquire a lock for the given key.

        The lock is created if it doesn't exist and removed when the last
        holder/waiter releases it.
        """
        # Get or create lock and register interest.
        # No await between these lines = atomic in asyncio.
        async with self._main_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
                self._refs[key] = 0
            self._refs[key] += 1

        try:
            async with self._locks[key]:
                yield
        finally:
            # Deregister and cleanup if last.
            # No await between these lines = atomic in asyncio.
            async with self._main_lock:
                self._refs[key] -= 1
                if self._refs[key] == 0:
                    del self._locks[key]
                    del self._refs[key]
