import asyncio
import socket
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable, TypeVar

import httpx
import uvicorn
from workflows.server import WorkflowServer

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


@asynccontextmanager
async def live_server(
    server_factory: Callable[[], WorkflowServer],
) -> AsyncGenerator[tuple[str, WorkflowServer], None]:
    """Start a live HTTP server for testing with atomic port acquisition.

    This context manager handles:
    - Atomic port acquisition (no race condition with parallel tests)
    - Server startup with health check
    - Graceful shutdown

    Args:
        server_factory: A callable that creates and configures a WorkflowServer.
            This allows tests to customize workflows, idle_release_timeout, etc.

    Yields:
        A tuple of (base_url, server) for making requests and inspecting state.

    Example:
        def make_server() -> WorkflowServer:
            server = WorkflowServer(idle_release_timeout=timedelta(seconds=1))
            server.add_workflow("test", MyWorkflow())
            return server

        async with live_server(make_server) as (base_url, server):
            client = WorkflowClient(base_url=base_url)
            # ... run tests
    """
    # Create socket and bind atomically - prevents race condition in parallel tests
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        port = sock.getsockname()[1]

        server = server_factory()

        config = uvicorn.Config(
            server.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            loop="asyncio",
        )
        uv_server = uvicorn.Server(config)

        # Start server in background task with our pre-bound socket
        task = asyncio.create_task(uv_server.serve(sockets=[sock]))

        # Wait until server responds on /health or timeout
        base_url = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(base_url=base_url, timeout=1.0) as client:
            for _ in range(50):  # ~0.5s max wait
                try:
                    resp = await client.get("/health")
                    if resp.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.01)
            else:
                uv_server.should_exit = True
                await task
                raise RuntimeError("Live server did not start in time")

        try:
            yield base_url, server
        finally:
            uv_server.should_exit = True
            try:
                await task
            finally:
                await server.stop()
    finally:
        # Socket is managed by uvicorn after serve() starts, but close if we fail early
        try:
            sock.close()
        except Exception:
            pass
