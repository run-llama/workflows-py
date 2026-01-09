import asyncio
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def async_yield(iterations: int = 10) -> None:
    """Yield to the event loop multiple times to let async tasks run.

    Use this when you need to let async tasks make progress without
    waiting for real time to pass. Useful with time_machine when time
    is frozen (tick=False).
    """
    for _ in range(iterations):
        await asyncio.sleep(0)


async def wait_for_passing(
    func: Callable[[], Awaitable[T]],
    max_duration: float = 5.0,
    interval: float = 0.05,
) -> T:
    start_time = time.monotonic()
    last_exception = None
    while time.monotonic() - start_time < max_duration:
        remaining_duration = max_duration - (time.monotonic() - start_time)
        try:
            return await asyncio.wait_for(func(), timeout=remaining_duration)
        except Exception as e:
            last_exception = e
            await asyncio.sleep(interval)
    if last_exception:
        raise last_exception
    else:
        func_name = getattr(func, "__name__", repr(func))
        raise TimeoutError(
            f"Function {func_name} timed out after {max_duration} seconds"
        )
