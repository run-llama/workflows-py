import logging
import os
import signal
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import FrameType
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from ..lifecycle import shutdown_event
from .backup_v1beta1 import router as backup_v1beta1
from .deployments_v1beta1 import router as deployments_v1beta1

logger = logging.getLogger(__name__)


_PREV_SIGNAL_HANDLERS: dict[int, signal.Handlers] = {}


# Register early signal handlers so long-running generators can exit promptly
def _handle_shutdown_signal(signum: int, frame: FrameType | None = None) -> None:
    logger.info(f"manage_api signal received: setting shutdown_event ({signum})")
    shutdown_event.set()

    # Chain to any previously-registered handler so the server can still shut down
    prev = _PREV_SIGNAL_HANDLERS.get(signum)
    if callable(prev):
        try:
            prev(signum, frame)
        except Exception:
            # Let exceptions propagate to allow normal shutdown behavior
            raise
    elif prev == signal.SIG_DFL:
        # Restore default then re-emit the signal to trigger default termination
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    elif prev == signal.SIG_IGN:
        # Respect ignore; nothing else to do
        pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    global _PREV_SIGNAL_HANDLERS
    shutdown_event.clear()
    for _sig in (signal.SIGINT, signal.SIGTERM):
        prev_handler = signal.getsignal(_sig)
        if prev_handler is not None:
            _PREV_SIGNAL_HANDLERS[_sig] = cast(signal.Handlers, prev_handler)
        signal.signal(_sig, _handle_shutdown_signal)

    yield
    # Ensure shutdown flag is set during app shutdown as a fallback
    shutdown_event.set()


app = FastAPI(title="LlamaDeploy on Cloud", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, include_in_schema=False)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    correlation_id = uuid.uuid4().hex
    logger.exception(
        "Unhandled error on %s %s [correlation_id=%s]",
        request.method,
        request.url.path,
        correlation_id,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "correlation_id": correlation_id,
        },
    )


# Include API routers

app.include_router(deployments_v1beta1)
app.include_router(backup_v1beta1)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
