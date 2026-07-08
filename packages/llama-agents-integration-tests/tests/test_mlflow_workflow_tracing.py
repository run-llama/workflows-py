# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

import mlflow
import pytest
from dbos import DBOS, DBOSConfig
from llama_agents.dbos import DBOSRuntime
from llama_index_instrumentation import get_dispatcher
from mlflow.entities import Span
from mlflow.entities.trace import Trace
from mlflow.llama_index.tracer import (
    MlflowSpanHandler,
    remove_llama_index_tracer,
    set_llama_index_tracer,
)
from workflows.decorators import step
from workflows.events import StartEvent, StopEvent
from workflows.runtime.types.internal_state import BrokerState
from workflows.workflow import Workflow


class TracePropagationWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


@pytest.fixture
def mlflow_tracing(tmp_path: Path) -> Generator[None, None, None]:
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.sqlite3'}")
    mlflow.set_experiment("workflow-tracing")
    remove_llama_index_tracer()
    set_llama_index_tracer()
    try:
        yield
    finally:
        dispatcher = get_dispatcher()
        for handler in dispatcher.span_handlers:
            if isinstance(handler, MlflowSpanHandler):
                with suppress(Exception):
                    handler.close()
        remove_llama_index_tracer()
        mlflow.set_tracking_uri(old_tracking_uri)


@pytest.fixture
async def dbos_runtime(
    tmp_path: Path,
) -> AsyncGenerator[DBOSRuntime, None]:
    config: DBOSConfig = {
        "name": "mlflow-trace-test",
        "system_database_url": f"sqlite+pysqlite:///{tmp_path / 'dbos.sqlite3'}?check_same_thread=false",
        "run_admin_server": False,
        "notification_listener_polling_interval_sec": 0.01,
    }
    DBOS(config=config)
    runtime = DBOSRuntime(polling_interval_sec=0.01)
    try:
        yield runtime
    finally:
        await runtime.destroy()
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor())


def _span_named(trace: Trace, name: str) -> Span:
    matches = [span for span in trace.data.spans if span.name == name]
    assert len(matches) == 1, (
        f"expected one span named {name!r}, got {[span.name for span in trace.data.spans]}"
    )
    return matches[0]


@pytest.mark.asyncio
async def test_mlflow_propagated_context_parents_workflow_trace_after_serialization(
    dbos_runtime: DBOSRuntime,
    mlflow_tracing: None,
) -> None:
    workflow = TracePropagationWorkflow(runtime=dbos_runtime)
    await dbos_runtime.launch()

    with mlflow.start_span("request-root") as root:
        root_trace_id = root.trace_id
        root_span_id = root.span_id
        propagation_context: dict[str, Any] = json.loads(
            json.dumps(get_dispatcher().capture_propagation_context())
        )

    registered = dbos_runtime.get_registered(workflow)
    assert registered is not None

    result = await registered.workflow_run_fn(
        BrokerState.from_workflow(workflow),
        StartEvent(),
        propagation_context,
    )

    assert result.result == "done"

    root_trace = mlflow.get_trace(root_trace_id, flush=True)
    assert root_trace is not None

    workflow_span = _span_named(root_trace, "TracePropagationWorkflow.run")
    step_span = _span_named(root_trace, "TracePropagationWorkflow.step_one")

    assert workflow_span.parent_id == root_span_id
    assert step_span.parent_id == workflow_span.span_id
