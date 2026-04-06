import asyncio
import signal
from asyncio import Event
from contextlib import suppress
from typing import Any, Coroutine, TypeVar

shutdown_event = Event()


def setup_interrupts() -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)


class OperationAborted(Exception):
    """Raised when an operation is aborted due to shutdown/interrupt."""


T = TypeVar("T")


async def wait_or_abort(
    awaitable: Coroutine[Any, Any, T], shutdown_event: asyncio.Event = shutdown_event
) -> T:
    """Await an operation, aborting early if shutdown is requested.

    If the shutdown event is set before the awaitable completes, cancel the
    awaitable and raise OperationAborted. Otherwise, return the awaitable's result.
    """
    event = shutdown_event
    if event.is_set():
        raise OperationAborted()

    op_task: asyncio.Task[T] = asyncio.create_task(awaitable)
    stop_task = asyncio.create_task(event.wait())
    try:
        done, _ = await asyncio.wait(
            {op_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done:
            op_task.cancel()
            with suppress(asyncio.CancelledError):
                await op_task
            raise OperationAborted()
        # Operation finished first
        stop_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task
        return await op_task
    finally:
        # Ensure no leaked tasks if an exception propagates
        for t in (op_task, stop_task):
            if not t.done():
                t.cancel()
