import logging
import logging.config
import os
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response
from llama_agents.appserver.correlation_id import (
    create_correlation_id,
    get_correlation_id,
    set_correlation_id,
)
from llama_agents.appserver.process_utils import should_use_color
from structlog.dev import RichTracebackFormatter

access_logger = logging.getLogger("app.access")


def _get_or_create_correlation_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", create_correlation_id())


def add_log_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_log_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        set_correlation_id(_get_or_create_correlation_id(request))
        return await call_next(request)

    @app.middleware("http")
    async def access_log_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if _is_proxy_request(request) or _is_health_request(request):
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        dur_ms = (time.perf_counter() - start) * 1000
        qp = str(request.query_params)
        if qp:
            qp = f"?{qp}"
        access_logger.info(
            f"{request.method} {request.url.path}{qp}",
            extra={
                "duration_ms": round(dur_ms, 2),
                "status_code": response.status_code,
            },
        )
        return response


def _add_request_id(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    req_id = get_correlation_id()
    if req_id and "request_id" not in event_dict:
        event_dict["request_id"] = req_id
    return event_dict


def _drop_uvicorn_color_message(
    _: Any, __: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    # Uvicorn injects an ANSI-colored duplicate of the message under this key
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging(level: str = "INFO") -> None:
    """
    Configure console logging via structlog with a compact, dev-friendly format.
    Includes request_id and respects logging.extra.
    """
    # Choose renderer and timestamp format based on LOG_FORMAT
    log_format = os.getenv("LOG_FORMAT", "console").lower()
    is_console = log_format == "console"

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
        timestamper = structlog.processors.TimeStamper(fmt="iso", key="timestamp")
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=should_use_color(),
            exception_formatter=RichTracebackFormatter(
                show_locals=False,
                width=120,
            ),
        )
        timestamper = structlog.processors.TimeStamper(fmt="%H:%M:%S", key="timestamp")

    pre_chain: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
        _add_request_id,
    ]

    # Ensure stdlib logs (foreign to structlog) also include `extra={...}` fields
    # and that exceptions/stack info are rendered nicely (esp. for JSON format)
    foreign_pre_chain = [
        *pre_chain,
        structlog.stdlib.ExtraAdder(),
        *(  # otherwise ConsoleRenderer will render nice rich stack traces
            [
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
            ]
            if not is_console
            else []
        ),
        _drop_uvicorn_color_message,
    ]

    structlog.configure(
        processors=[
            *pre_chain,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.stdlib.ExtraAdder(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = {
        "class": "logging.StreamHandler",
        "level": level,
        "formatter": "console",
        "stream": "ext://sys.stdout",
    }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    # With Rich, let it handle the final formatting; otherwise use our renderer
                    "processor": renderer,
                    "foreign_pre_chain": foreign_pre_chain,
                }
            },
            "handlers": {"console": handler, "default": handler},
            "root": {
                "handlers": ["console"],
                "level": level,
            },
            "loggers": {
                "uvicorn.access": {  # disable access logging, we have our own access log
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    )

    # Reduce noise from httpx globally, with fine-grained suppression controlled per-request
    logging.getLogger("httpx").addFilter(_HttpxProxyNoiseFilter())


#####################################################################################
### Proxying through the fastapi server in dev mode is noisy, various suppressions
###
def _is_proxy_request(request: Request) -> bool:
    parts = request.url.path.split("/")
    return len(parts) >= 4 and parts[1] == "deployments" and parts[3] == "ui"


_HEALTH_PATHS = {"/health", "/healthz", "/livez", "/readyz", "/metrics"}


def _is_health_request(request: Request) -> bool:
    return request.url.path.rstrip("/") in _HEALTH_PATHS


_suppress_httpx_logging: ContextVar[bool] = ContextVar(
    "suppress_httpx_logging", default=False
)


class _HttpxProxyNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to drop httpx info/debug logs when suppression is active."""
        try:
            if record.name.startswith("httpx") and record.levelno <= logging.INFO:
                return not _suppress_httpx_logging.get()
        except Exception:
            return True
        return True


@asynccontextmanager
async def suppress_httpx_logs() -> AsyncGenerator[None, None]:
    _suppress_httpx_logging.set(True)
    yield
    _suppress_httpx_logging.set(False)
