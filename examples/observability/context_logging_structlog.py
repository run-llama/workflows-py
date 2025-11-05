"""
Demonstrates how dispatcher and workflow context can be automatically added to log entries with structlog.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, MutableMapping
import structlog

from llama_index_instrumentation.dispatcher import (
    active_instrument_tags,
    instrument_tags,
)

from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent


def configure_structlog() -> None:
    def merge_custom_context(
        _logger: structlog.BoundLogger,
        _method_name: str,
        event_dict: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """
        Merge values from your ContextVar dict into structlog's event_dict.
        Later processors (e.g., JSONRenderer) will see these keys as if bound.
        """
        ctx = active_instrument_tags.get()
        if ctx:
            # don't clobber explicitly-set event keys unless you want to:
            for k, v in ctx.items():
                event_dict.setdefault(k, v)
                # or: event_dict[k] = v  # if you want your ctx to win
        return event_dict

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            merge_custom_context,  # <------------- Add this to add llama index dispatcher tags to structlog
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


configure_structlog()
logging.basicConfig(level=logging.INFO, format="%(message)s run_id=%(run_id)s")

structlog_logger = structlog.get_logger()
regular_logger = logging.getLogger()


class LoggingWorkflow(Workflow):
    """A workflow that demonstrates log context."""

    @step
    async def log_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        # Any fields bound here will also appear alongside dispatcher tags
        structlog_logger.info("processing-start")
        regular_logger.info("processing-start", extra={**active_instrument_tags.get()})

        # Simulate a bit of work and access the store to prove context works
        await ctx.store.set("seen", True)

        structlog_logger.info("processing-done")
        regular_logger.info("processing-done", extra={**active_instrument_tags.get()})
        return StopEvent(result="ok")


async def main() -> None:
    # Tags set outside the workflow run will be captured in all logs emitted
    # during the run (together with run_id injected by the broker).
    outer_tags: dict[str, Any] = {"request_id": "req-123", "user": "alice"}

    wf = LoggingWorkflow()

    # The broker sets instrument_tags to include run_id during the run.
    # We add extra outer tags here to demonstrate they are merged too.
    with instrument_tags(outer_tags):
        handler = wf.run()

        # Stream events to ensure the run proceeds fully while we log
        async for _ in handler.stream_events():
            pass

        result = await handler

    # Example emits logs like:
    # 2025-11-05 13:34:13 [info     ] processing-start               request_id=req-123 run_id=a7e4gwViAt user=alice
    # processing-start run_id=a7e4gwViAt
    # 2025-11-05 13:34:13 [info     ] processing-done                request_id=req-123 run_id=a7e4gwViAt user=alice
    # processing-done run_id=a7e4gwViAt
    # Workflow result: ok
    print(f"Workflow result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
