# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Shared test utilities for integration tests that run live HTTP servers."""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def wait_for_passing(
    func: Callable[[], Awaitable[T]],
    max_duration: float = 5.0,
    interval: float = 0.05,
) -> T:
    """Retry an async callable until it stops raising, or time runs out."""
    start_time = time.monotonic()
    last_exception: Exception | None = None
    while time.monotonic() - start_time < max_duration:
        remaining_duration = max_duration - (time.monotonic() - start_time)
        try:
            return await asyncio.wait_for(func(), timeout=remaining_duration)
        except Exception as e:
            last_exception = e
            await asyncio.sleep(interval)
    if last_exception:
        raise last_exception
    raise TimeoutError(f"Timed out after {max_duration}s")
