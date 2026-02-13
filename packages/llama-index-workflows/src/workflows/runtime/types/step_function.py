# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import functools
import time
from contextvars import copy_context
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generic, Protocol

from llama_index_instrumentation.dispatcher import instrument_tags
from llama_index_instrumentation.span import active_span_id
from workflows.decorators import P, R, StepConfig
from workflows.errors import WorkflowRuntimeError
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
)
from workflows.runtime.control_loop import control_loop
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.plugin import (
    ControlLoopFunction,
    RunContext,
    WorkflowRunFunction,
    run_context,
)
from workflows.runtime.types.results import (
    Returns,
    StepFunctionResult,
    StepWorkerContext,
    StepWorkerFailed,
    StepWorkerResult,
    StepWorkerState,
    StepWorkerStateContextVar,
    WaitingForEvent,
)
from workflows.workflow import Workflow

if TYPE_CHECKING:
    from workflows.context.context import Context


class StepWorkerFunction(Protocol, Generic[R]):
    def __call__(
        self,
        state: StepWorkerState,
        step_name: str,
        event: Event,
        workflow: Workflow,
    ) -> Awaitable[list[StepFunctionResult[R]]]: ...


async def partial(
    func: Callable[..., R],
    step_config: StepConfig,
    event: Event,
    context: Context,
    workflow: Workflow,
) -> Callable[[], R]:
    kwargs: dict[str, Any] = {}
    kwargs[step_config.event_name] = event
    if step_config.context_parameter:
        # Convert to internal face for step execution
        kwargs[step_config.context_parameter] = context
    with workflow._resource_manager.resolution_scope():
        for resource_def in step_config.resources:
            descriptor = resource_def.resource
            descriptor.set_type_annotation(resource_def.type_annotation)
            # Unified resolution through ResourceManager
            resource_value = await workflow._resource_manager.get(resource=descriptor)
            kwargs[resource_def.name] = resource_value
    return functools.partial(func, **kwargs)


def as_step_worker_functions(workflow: Workflow) -> dict[str, StepWorkerFunction]:
    step_funcs = workflow._get_steps()
    step_workers: dict[str, StepWorkerFunction[Any]] = {
        name: as_step_worker_function(getattr(func, "__func__", func))
        for name, func in step_funcs.items()
    }
    return step_workers


def as_step_worker_function(func: Callable[P, Awaitable[R]]) -> StepWorkerFunction[R]:
    """
    Wrap a step function, setting context variables and handling exceptions to instead
    return the appropriate StepFunctionResult.
    """

    # Keep original function reference for free-function steps; for methods we
    # will resolve the currently-bound method from the provided workflow at call time.
    original_func: Callable[..., Awaitable[R]] = func

    # Avoid functools.wraps here because it would set __wrapped__ to the bound
    # method (when present), which would strongly reference the workflow
    # instance and prevent garbage collection under high churn.
    async def wrapper(
        state: StepWorkerState,
        step_name: str,
        event: Event,
        workflow: Workflow,
    ) -> list[StepFunctionResult[R]]:
        from workflows.context.context import Context

        internal_context = Context._create_internal(workflow=workflow)
        returns = Returns[R](return_values=[])

        token = StepWorkerStateContextVar.set(
            StepWorkerContext(state=state, returns=returns)
        )

        try:
            config = workflow._get_steps()[step_name]._step_config
            # Resolve callable at call time:
            # - If the workflow has an attribute with the step name, use it
            #   (this yields a bound method for instance-defined steps).
            # - Otherwise, fall back to the original function (free function step).
            try:
                call_func = getattr(workflow, step_name)
            except AttributeError:
                call_func = original_func
            partial_func = await partial(
                func=workflow._dispatcher.span(call_func),
                step_config=config,
                event=event,
                context=internal_context,
                workflow=workflow,
            )

            try:
                # coerce to coroutine function
                if not asyncio.iscoroutinefunction(call_func):
                    # run_in_executor doesn't accept **kwargs, so we need to use partial
                    copy = copy_context()

                    result: R = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: copy.run(partial_func),  # type: ignore
                    )
                else:
                    result = await partial_func()
                    if result is not None and not isinstance(result, Event):
                        msg = f"Step function {step_name} returned {type(result).__name__} instead of an Event instance."
                        raise WorkflowRuntimeError(msg)
                returns.return_values.append(StepWorkerResult(result=result))
            except WaitingForEvent as e:
                await asyncio.sleep(0)
                returns.return_values.append(e.add)
            except Exception as e:
                returns.return_values.append(
                    StepWorkerFailed(exception=e, failed_at=time.time())
                )

            await internal_context._finalize_step()
            return returns.return_values
        finally:
            try:
                StepWorkerStateContextVar.reset(token)
            except Exception:
                pass

    # Manually set minimal metadata without retaining bound instance references.
    try:
        unbound_for_wrapped = getattr(func, "__func__", func)
        wrapper.__name__ = getattr(func, "__name__", wrapper.__name__)
        wrapper.__qualname__ = getattr(func, "__qualname__", wrapper.__qualname__)
        # Point __wrapped__ to the unbound function when available to avoid
        # strong refs to the instance via a bound method object.
        setattr(wrapper, "__wrapped__", unbound_for_wrapped)
    except Exception:
        # Best-effort; lack of these attributes is non-fatal.
        pass

    return wrapper


def create_workflow_run_function(
    workflow: Workflow, control_loop_fn: ControlLoopFunction = control_loop
) -> WorkflowRunFunction:
    async def run_workflow(
        init_state: BrokerState,
        start_event: StartEvent | None = None,
        tags: dict[str, Any] | None = None,
    ) -> StopEvent:
        from workflows.context.context import Context
        from workflows.context.internal_context import InternalContext

        registered = workflow._runtime.get_or_register(workflow)
        # Set run_id context before creating internal context
        internal_ctx = Context._create_internal(workflow=workflow)
        internal_adapter = workflow._runtime.get_internal_adapter(workflow)

        # Extract parent span ID if present and remove from tags
        tags = tags or {}
        parent_span_id = tags.pop("parent_span_id", None)

        # Set parent span ID context if provided
        parent_span_token = None
        if parent_span_id is not None:
            parent_span_token = active_span_id.set(parent_span_id)

        try:
            with instrument_tags(tags):
                # defer execution to make sure the task can be captured and passed
                # to the handler as async exception, protecting against exceptions from before_start
                await asyncio.sleep(0)

                run_ctx = RunContext(
                    workflow=workflow,
                    run_adapter=internal_adapter,
                    context=internal_ctx,
                    steps=registered.steps,
                )
                try:
                    with run_context(run_ctx):
                        result = await control_loop_fn(
                            start_event,
                            init_state,
                            internal_adapter.run_id,
                        )
                        return result
                finally:
                    # Cancel any background tasks from InternalContext on completion or cancellation
                    if isinstance(internal_ctx._face, InternalContext):
                        internal_ctx._face.cancel_background_tasks()
        finally:
            # Reset parent span ID if it was set
            if parent_span_token is not None:
                active_span_id.reset(parent_span_token)

    return run_workflow
