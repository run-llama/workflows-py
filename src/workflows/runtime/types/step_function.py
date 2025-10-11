# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import asyncio
from contextvars import copy_context
import functools
import time
from typing import Any, Awaitable, Callable, TYPE_CHECKING, Generic, Protocol


from workflows.decorators import P, R, StepConfig
from workflows.errors import WorkflowRuntimeError
from workflows.events import (
    Event,
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
        context: Context,  # TODO - pass an identifier and re-hydrate from the plugin for distributed step workers
        workflow: Workflow,
    ) -> Awaitable[list[StepFunctionResult[R, Any]]]: ...


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
        kwargs[step_config.context_parameter] = context
    for resource in step_config.resources:
        resource_value = await workflow._resource_manager.get(
            resource=resource.resource
        )
        kwargs[resource.name] = resource_value
    return functools.partial(func, **kwargs)


# TODO - make sure this is serializable for distributed step workers
def as_step_worker_function(func: Callable[P, Awaitable[R]]) -> StepWorkerFunction[R]:
    """
    Wrap a step function, setting context variables and handling exceptions to instead
    return the appropriate StepFunctionResult.
    """

    @functools.wraps(func)
    async def wrapper(
        state: StepWorkerState,
        step_name: str,
        event: Event,
        context: Context,
        workflow: Workflow,
    ) -> list[StepFunctionResult[R, Any]]:
        returns = Returns[R](return_values=[])

        token = StepWorkerStateContextVar.set(
            StepWorkerContext(state=state, returns=returns)
        )

        try:
            config = workflow._get_steps()[step_name]._step_config
            partial_func = await partial(
                func=workflow._dispatcher.span(func),
                step_config=config,
                event=event,
                context=context,
                workflow=workflow,
            )

            try:
                # coerce to coroutine function
                if not asyncio.iscoroutinefunction(func):
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
                    StepWorkerFailed(exception=e, failed_at=time.monotonic())
                )
            return returns.return_values
        finally:
            try:
                StepWorkerStateContextVar.reset(token)
            except Exception:
                pass

    return wrapper
