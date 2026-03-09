"""
K8s OTEL Example — Counter + Greeter workflows with DBOS, OpenTelemetry, and structlog.

Serves via FastAPI with the WorkflowServer mounted at /api and custom endpoints at the
top level. FastAPI is instrumented with OpenTelemetry.

Env vars:
  POSTGRES_DSN              — Postgres connection string
  OTEL_EXPORTER_OTLP_ENDPOINT — OTLP gRPC endpoint (e.g. http://phoenix:4317)
  SERVER_PORT               — HTTP port (default 8080)
  IDLE_TIMEOUT              — Seconds before idle release (default 30)
  EXECUTOR_POOL_SIZE        — Number of executor slots (e.g., "2")
"""

from __future__ import annotations

import asyncio
import logging
import logging.config
import os
import signal
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, MutableMapping

import structlog
import uvicorn
from dbos import DBOS
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from llama_agents.dbos import DBOSRuntime
from llama_agents.server import WorkflowServer
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from llama_index_instrumentation import get_dispatcher
from llama_index_instrumentation.dispatcher import active_instrument_tags
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import Field
from starlette.types import ASGIApp, Receive, Scope, Send
from workflows import Context, Workflow, step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://workflows:workflows@localhost:5432/workflows"
)
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))
IDLE_TIMEOUT = float(os.environ.get("IDLE_TIMEOUT", "30"))
EXECUTOR_POOL_SIZE = int(os.environ.get("EXECUTOR_POOL_SIZE", "2"))

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

log = structlog.get_logger("app")
access_logger = logging.getLogger("access")


def _merge_instrument_tags(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    ctx = active_instrument_tags.get()
    if ctx:
        for k, v in ctx.items():
            event_dict.setdefault(k, v)
    return event_dict


def _drop_uvicorn_color_message(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging() -> None:
    """Configure all Python logging (stdlib + structlog) to emit JSON.

    Must be called AFTER DBOS() init so we can clear its custom handlers.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _merge_instrument_tags,
    ]

    # structlog loggers → wrap into a stdlib LogRecord for the formatter
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=[
            structlog.stdlib.add_logger_name,
            *shared_processors,
            _drop_uvicorn_color_message,
            structlog.stdlib.ExtraAdder(),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Clear DBOS's private handler so its logs propagate to root
    dbos_logger = logging.getLogger("dbos")
    dbos_logger.handlers.clear()
    dbos_logger.propagate = True

    # Suppress uvicorn's default access logger (we have our own middleware)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# OTEL setup
# ---------------------------------------------------------------------------
otel_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
instrumentor = LlamaIndexOpenTelemetry(
    span_exporter=otel_exporter,
    service_name_or_resource="k8s-otel-example",
    span_processor="simple",
)
instrumentor.start_registering()

# ---------------------------------------------------------------------------
# DBOS setup — must be called at module level before DBOSRuntime()
# ---------------------------------------------------------------------------
DBOS(
    config={
        "name": "k8s-otel-example",
        "system_database_url": POSTGRES_DSN,
        "run_admin_server": False,
    }
)

# Must come AFTER DBOS() init — DBOS adds its own handlers in __init__
setup_logging()

# ---------------------------------------------------------------------------
# Counter Workflow
# ---------------------------------------------------------------------------


class Tick(Event):
    count: int = Field(description="Current count")


class WaitDone(Event):
    count: int = Field(description="Current count after waiting")


class CounterResult(StopEvent):
    final_count: int = Field(description="Final counter value")


class CounterWorkflow(Workflow):
    """Counts to 20 with 1s delays, emitting Tick stream events."""

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> WaitDone:
        log.info("counter.start")
        return WaitDone(count=0)

    @step
    async def tick(self, ctx: Context, ev: WaitDone) -> Tick | CounterResult:
        count = ev.count + 1
        await ctx.store.set("count", count)
        ctx.write_event_to_stream(Tick(count=count))
        log.info("counter.tick", count=count)
        if count >= 20:
            return CounterResult(final_count=count)
        return Tick(count=count)

    @step
    async def wait(self, ctx: Context, ev: Tick) -> WaitDone:
        await asyncio.sleep(1.0)
        return WaitDone(count=ev.count)


# ---------------------------------------------------------------------------
# Greeter Workflow (HITL with idle release)
# ---------------------------------------------------------------------------


class AskName(InputRequiredEvent):
    prompt: str = Field(default="What is your name?")


class UserInput(HumanResponseEvent):
    response: str = Field(default="")


class GreeterWorkflow(Workflow):
    """Ask for a name, wait (idle-releases), then greet."""

    @step
    async def ask(self, ctx: Context, ev: StartEvent) -> AskName:
        log.info("greeter.ask")
        return AskName()

    @step
    async def greet(self, ctx: Context, ev: UserInput) -> StopEvent:
        greeting = f"Hello, {ev.response}!"
        log.info("greeter.greet", greeting=greeting)
        return StopEvent(result={"greeting": greeting})


# ---------------------------------------------------------------------------
# Workflow Server
# ---------------------------------------------------------------------------

runtime = DBOSRuntime(
    _experimental_executor_lease={
        "pool_size": EXECUTOR_POOL_SIZE,
    },
)

workflow_server = WorkflowServer(
    workflow_store=runtime.create_workflow_store(),
    runtime=runtime.build_server_runtime(idle_timeout=IDLE_TIMEOUT),
)
workflow_server.add_workflow("counter", CounterWorkflow(runtime=runtime))
workflow_server.add_workflow("greeter", GreeterWorkflow(runtime=runtime))

# ---------------------------------------------------------------------------
# FastAPI app with WorkflowServer mounted
# ---------------------------------------------------------------------------


def _flush_and_shutdown() -> None:
    """Flush open spans and shut down the OTel pipeline synchronously."""
    dispatcher = get_dispatcher()
    for h in dispatcher.span_handlers:
        log.info(
            "shutdown.spans",
            handler=h.class_name(),
            open_span_ids=list(h.open_spans.keys()),
        )
        all_spans = getattr(h, "all_spans", None)
        if all_spans is not None:
            log.info(
                "shutdown.otel_spans",
                otel_span_ids=list(all_spans.keys()),
            )
    dispatcher.shutdown()
    log.info("shutdown.dispatcher_done")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Register signal handler so spans flush on SIGTERM even if uvicorn
    # is still waiting for connections to drain.
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _flush_and_shutdown)
    await workflow_server.start()
    try:
        yield
    finally:
        _flush_and_shutdown()
        await workflow_server.stop()


HEALTH_PATHS = {"/health"}


class AccessLogMiddleware:
    """Raw ASGI middleware so it covers mounted sub-apps (e.g. /api)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in HEALTH_PATHS:
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 0

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        dur_ms = (time.perf_counter() - start) * 1000
        method = scope.get("method", "?")
        access_logger.info(
            "%s %s",
            method,
            path,
            extra={"status_code": status_code, "duration_ms": round(dur_ms, 1)},
        )


app = FastAPI(title="K8s OTEL Example", lifespan=lifespan)

# Mount the workflow server's Starlette app under /api
app.mount("/api", workflow_server.app)

# Access log middleware wraps the entire ASGI app including mounts
app.add_middleware(AccessLogMiddleware)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app, excluded_urls="health")


@app.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse(url="/api/?api=/api/")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    log.info(
        "server.starting",
        port=SERVER_PORT,
        executor_pool_size=EXECUTOR_POOL_SIZE,
        idle_timeout=IDLE_TIMEOUT,
    )
    config = uvicorn.Config(
        app, host="0.0.0.0", port=SERVER_PORT, log_config=None, access_log=False
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
