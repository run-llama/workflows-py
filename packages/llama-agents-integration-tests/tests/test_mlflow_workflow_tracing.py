# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import contextvars
import json
from contextlib import suppress
from pathlib import Path
from typing import Any, Generator, cast

import mlflow
import pytest
from llama_index_instrumentation import get_dispatcher
from mlflow.entities import Span
from mlflow.entities.trace import Trace
from mlflow.llama_index.tracer import (
    MlflowSpanHandler,
    remove_llama_index_tracer,
    set_llama_index_tracer,
)
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.context.state_store import (
    InMemoryStateStore,
    infer_state_type,
    is_durable_serialized_state,
)
from workflows.decorators import step
from workflows.errors import WorkflowRuntimeError
from workflows.events import StartEvent, StopEvent
from workflows.plugins.basic import BasicRuntime, setting_run_id
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import ExternalRunAdapter
from workflows.workflow import Workflow


class TracePropagationWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> StopEvent:
        with mlflow.start_span("manual.step_one") as span:
            span.set_outputs("done")
        return StopEvent(result="done")


class ContextBoundaryRuntime(BasicRuntime):
    def run_workflow(
        self,
        run_id: str,
        workflow: Workflow,
        init_state: BrokerState,
        start_event: StartEvent | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> ExternalRunAdapter:
        if run_id in self._queues:
            raise RuntimeError(f"Workflow run with run_id '{run_id}' already exists.")

        registered = self.get_or_register(workflow)
        active_serializer = serializer or JsonSerializer()
        if serialized_state:
            if is_durable_serialized_state(serialized_state):
                store_type = serialized_state.get("store_type")
                raise WorkflowRuntimeError(
                    f"BasicRuntime cannot restore durable state store '{store_type}'. "
                    "Use the matching durable runtime or pass an in-memory context snapshot."
                )
            state_store = InMemoryStateStore.from_dict(
                serialized_state, active_serializer
            )
        else:
            state_store = InMemoryStateStore(infer_state_type(registered.workflow)())

        queues = self._get_or_create_queues(run_id, init_state)
        queues.state_store = state_store
        captured_tags = json.loads(
            json.dumps(get_dispatcher().capture_propagation_context())
        )

        async def run_with_concurrency_limit() -> StopEvent:
            _ = queues
            with setting_run_id(run_id):
                async with self._maybe_acquire_max_concurrent_runs(workflow, run_id):
                    return await registered.workflow_run_fn(
                        init_state, start_event, captured_tags
                    )

        task = cast(
            asyncio.Task[StopEvent],
            contextvars.Context().run(
                asyncio.create_task,
                run_with_concurrency_limit(),
            ),
        )
        queues.complete = task
        return self.get_external_adapter(run_id)


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


def _span_named(trace: Trace, name: str) -> Span:
    matches = [span for span in trace.data.spans if span.name == name]
    assert len(matches) == 1, (
        f"expected one span named {name!r}, got {[span.name for span in trace.data.spans]}"
    )
    return matches[0]


@pytest.mark.asyncio
async def test_mlflow_propagated_context_parents_workflow_trace_after_serialization(
    mlflow_tracing: None,
) -> None:
    workflow = TracePropagationWorkflow(runtime=ContextBoundaryRuntime())

    with mlflow.start_span("request-root") as root:
        root_trace_id = root.trace_id
        root_span_id = root.span_id

        result = await workflow.run()

        assert result == "done"

    root_trace = mlflow.get_trace(root_trace_id, flush=True)
    assert root_trace is not None

    workflow_span = _span_named(root_trace, "TracePropagationWorkflow.run")
    step_span = _span_named(root_trace, "TracePropagationWorkflow.step_one")
    manual_span = _span_named(root_trace, "manual.step_one")

    assert workflow_span.parent_id == root_span_id
    assert step_span.parent_id == workflow_span.span_id
    assert manual_span.parent_id == step_span.span_id
