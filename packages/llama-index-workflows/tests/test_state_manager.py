# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Minimal unit tests for InMemoryStateStore.

Full state store protocol tests are in the integration test package
(llama-index-integration-tests/tests/test_state_store_matrix.py),
which tests InMemoryStateStore alongside SqliteStateStore.

These tests provide fast feedback during development of the base package.
"""

from __future__ import annotations

import pytest
from workflows.context.state_store import DictState, InMemoryStateStore


@pytest.mark.asyncio
async def test_in_memory_state_store_smoke() -> None:
    """Smoke test for basic InMemoryStateStore functionality."""
    store: InMemoryStateStore[DictState] = InMemoryStateStore(DictState())

    # Basic get/set
    await store.set("name", "test")
    assert await store.get("name") == "test"

    # Nested path
    await store.set("nested", {"key": "value"})
    assert await store.get("nested.key") == "value"

    # Default on missing
    assert await store.get("missing", default=None) is None

    # Clear
    await store.clear()
    assert await store.get("name", default=None) is None


@pytest.mark.asyncio
async def test_in_memory_edit_state() -> None:
    """Test edit_state context manager."""
    store: InMemoryStateStore[DictState] = InMemoryStateStore(DictState())

    async with store.edit_state() as state:
        state["counter"] = 1

    assert await store.get("counter") == 1
