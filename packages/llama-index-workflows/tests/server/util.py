import asyncio
import time
from typing import Callable, Awaitable, TypeVar


T = TypeVar("T")


async def wait_for_passing(
    func: Callable[[], Awaitable[T]],
    max_duration: float = 5.0,
    interval: float = 0.05,
) -> T:
    start_time = time.time()
    last_exception = None
    while time.time() - start_time < max_duration:
        remaining_duration = max_duration - (time.time() - start_time)
        try:
            return await asyncio.wait_for(func(), timeout=remaining_duration)
        except Exception as e:
            last_exception = e
            await asyncio.sleep(interval)
    if last_exception:
        raise last_exception
    else:
        raise TimeoutError(
            f"Function {func.__name__} timed out after {max_duration} seconds"
        )
