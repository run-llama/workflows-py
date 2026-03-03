# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""
Verify that wait_for_event and cancel_run produce OK spans, not ERROR spans.

Requires the k8s-otel stack running via Tilt (Jaeger OTLP on localhost:4317,
Jaeger UI on localhost:16686).

Usage:
    cd examples/k8s-otel
    tilt up          # in another terminal
    uv run python verify_spans.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from llama_index.observability.otel import LlamaIndexOpenTelemetry
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import get_tracer_provider
from workflows import Context, Workflow
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent

OTEL_ENDPOINT = "http://localhost:4317"
JAEGER_UI = "http://localhost:16686"


# -- Events ------------------------------------------------------------------


class WaitEvent(Event):
    value: str


# -- Workflows ----------------------------------------------------------------


class WaitForEventWorkflow(Workflow):
    """Step calls wait_for_event, pauses, then resumes when the event arrives."""

    @step
    async def wait_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        result = await ctx.wait_for_event(WaitEvent)
        return StopEvent(result=result.value)


class CancelledWorkflow(Workflow):
    """Step sleeps forever; we cancel it to test CancelledError span handling."""

    @step
    async def sleep_step(self, ev: StartEvent) -> StopEvent:
        await asyncio.sleep(3600)
        return StopEvent(result="unreachable")


# -- Helpers ------------------------------------------------------------------


def setup_otel() -> None:
    exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
    instrumentor = LlamaIndexOpenTelemetry(
        span_exporter=exporter,
        service_name_or_resource="verify-span-fix",
    )
    instrumentor.start_registering()


async def run_wait_for_event_test() -> None:
    print("\n=== Test 1: wait_for_event ===")
    print("Running workflow with wait_for_event...")

    wf = WaitForEventWorkflow()
    handler = wf.run()
    assert handler.ctx is not None

    # Send the waited-for event
    handler.ctx.send_event(WaitEvent(value="hello from verify script"))

    result = await handler
    print(f"Result: {result}")
    print("OK — workflow completed. Check Jaeger for two OK spans on 'wait_step'.")


async def run_cancel_test() -> None:
    print("\n=== Test 2: cancel_run ===")
    print("Running workflow that sleeps, then cancelling...")

    wf = CancelledWorkflow()
    handler = wf.run()

    # Let the step start
    await asyncio.sleep(0.1)

    await handler.cancel_run()
    try:
        await handler
    except Exception as e:
        print(f"Caught expected exception: {type(e).__name__}: {e}")

    print("OK — workflow cancelled. Check Jaeger for OK span on 'sleep_step' (no ERROR).")


async def main() -> None:
    setup_otel()

    await run_wait_for_event_test()
    await run_cancel_test()

    # Flush spans to the OTLP exporter
    print("\nFlushing spans to Jaeger...")
    provider = get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush()

    print(f"\nDone! Open Jaeger UI to verify spans: {JAEGER_UI}")
    print("Look for service 'verify-span-fix' — all step spans should show OK status.")


if __name__ == "__main__":
    asyncio.run(main())
