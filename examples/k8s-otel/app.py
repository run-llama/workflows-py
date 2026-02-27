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
import os
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
from llama_index_instrumentation.dispatcher import active_instrument_tags
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
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
# Structlog setup — must happen before any logging
# ---------------------------------------------------------------------------


def merge_custom_context(
    _logger: structlog.BoundLogger,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    ctx = active_instrument_tags.get()
    if ctx:
        for k, v in ctx.items():
            event_dict.setdefault(k, v)
    return event_dict


structlog.configure(
    processors=[
        merge_custom_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.ConsoleRenderer(),
    ],
)

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# OTEL setup — swap between openinference and llama-index-observability-otel
# by commenting/uncommenting the blocks below. Keep BOTH blocks present.
# ---------------------------------------------------------------------------
otel_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)

# --- Option A: openinference (active) ---
tracer_provider = TracerProvider(
    resource=Resource(attributes={SERVICE_NAME: "k8s-otel-example"})
)
tracer_provider.add_span_processor(BatchSpanProcessor(otel_exporter))
set_tracer_provider(tracer_provider)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# --- Option B: llama-index-observability-otel (inactive) ---
# from llama_index.observability.otel import LlamaIndexOpenTelemetry
# instrumentor = LlamaIndexOpenTelemetry(
#     span_exporter=otel_exporter,
#     service_name_or_resource="k8s-otel-example",
# )
# instrumentor.start_registering()

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await workflow_server.start()
    try:
        yield
    finally:
        await workflow_server.stop()


access_log = structlog.get_logger("access")

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
        access_log.info(
            f"{method} {path}",
            status_code=status_code,
            duration_ms=round(dur_ms, 1),
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


@app.get("/info")
async def info() -> dict[str, Any]:
    return {
        "executor_pool_size": EXECUTOR_POOL_SIZE,
        "idle_timeout": IDLE_TIMEOUT,
        "workflows": ["counter", "greeter"],
    }


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
