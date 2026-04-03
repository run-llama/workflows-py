import asyncio
from typing import Literal, cast

from bedrock_agentcore.memory import MemoryClient

ActivityStatus = Literal["ACTIVE", "FAILED", "CREATING", "DELETING"]


async def check_memory_activity(client: MemoryClient, memory_id: str) -> bool:
    max_retries = 120
    base_delay = 0.5
    retries = 0
    while retries < max_retries:
        status = await asyncio.to_thread(
            client.get_memory_status,
            memory_id=memory_id,
        )
        status = cast(ActivityStatus, status)
        if status in ("FAILED", "DELETING"):
            return False
        elif status == "ACTIVE":
            return True
        else:
            if retries < max_retries - 1:
                delay = base_delay * (retries + 1)
                await asyncio.sleep(delay)
            retries += 1
    return False


async def create_memory(client: MemoryClient, memory_name: str, namespace: str) -> str:
    response = await asyncio.to_thread(
        client.create_or_get_memory,
        name=memory_name,
    )
    memory_id = response.get("memoryId", response.get("id"))
    if memory_id is None:
        raise RuntimeError(f"AgentCore failed to create a memory ID for {namespace}")
    return memory_id
