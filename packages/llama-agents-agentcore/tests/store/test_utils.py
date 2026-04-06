import time

import pytest
from llama_agents.agentcore.store._utils import check_memory_activity

from .conftest import MockMemoryClient


@pytest.mark.asyncio
async def test_check_for_memory_activity_success() -> None:
    client = MockMemoryClient()
    is_active = await check_memory_activity(client, "test-memory")  # type: ignore
    assert is_active


@pytest.mark.asyncio
async def test_check_for_memory_activity_failure() -> None:
    client = MockMemoryClient()
    client.memories_failed = True
    is_active = await check_memory_activity(client, "test-memory")  # type: ignore
    assert not is_active


@pytest.mark.asyncio
async def test_check_for_memory_activity_retries() -> None:
    client = MockMemoryClient()
    client.memories_pending = 3
    start = time.time()
    is_active = await check_memory_activity(client, "test-memory")  # type: ignore
    end = time.time()
    assert is_active
    assert (
        end - start == pytest.approx(3, abs=0.1)
    )  # linear backoff: (0.5*1) + (0.5*2) + (0.5*3) = 3 (delay is 0.5s every time, for 3 times)
