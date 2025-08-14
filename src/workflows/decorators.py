# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Type

from pydantic import BaseModel, ConfigDict, Field

from .errors import WorkflowValidationError
from .resource import ResourceDefinition
from .utils import (
    inspect_signature,
    is_free_function,
    validate_step_signature,
)

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import Workflow
from .retry_policy import RetryPolicy


class StepConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    accepted_events: list[Any]
    event_name: str
    return_types: list[Any]
    context_parameter: str | None
    num_workers: int
    retry_policy: RetryPolicy | None
    resources: list[ResourceDefinition]
    context_state_type: Type[BaseModel] | None = Field(default=None)


def step(
    *args: Any,
    workflow: Type["Workflow"] | None = None,
    num_workers: int = 4,
    retry_policy: RetryPolicy | None = None,
) -> Callable:
    """
    Decorate a callable to declare it as a workflow step.

    The decorator inspects the function signature to infer the accepted event
    type, return event types, optional `Context` parameter (optionally with a
    typed state model), and any resource injections via `typing.Annotated`.

    When applied to free functions, provide the workflow class via
    `workflow=MyWorkflow`. For instance methods, the association is automatic.

    Args:
        workflow (type[Workflow] | None): Workflow class to attach the free
            function step to. Not required for methods.
        num_workers (int): Number of workers for this step. Defaults to 4.
        retry_policy (RetryPolicy | None): Optional retry policy for failures.

    Returns:
        Callable: The original function, annotated with internal step metadata.

    Raises:
        WorkflowValidationError: If signature validation fails or when decorating
            a free function without specifying `workflow`.

    Examples:
        Method step:

        ```python
        class MyFlow(Workflow):
            @step
            async def start(self, ev: StartEvent) -> StopEvent:
                return StopEvent(result="done")
        ```

        Free function step:

        ```python
        class MyWorkflow(Workflow):
            pass

        @step(workflow=MyWorkflow)
        async def generate(ev: StartEvent) -> NextEvent: ...
        ```
    """

    def decorator(func: Callable) -> Callable:
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise WorkflowValidationError(
                "num_workers must be an integer greater than 0"
            )

        # This will raise providing a message with the specific validation failure
        spec = inspect_signature(func)
        validate_step_signature(spec)
        event_name, accepted_events = next(iter(spec.accepted_events.items()))

        # store the configuration in the function object
        func.__step_config = StepConfig(  # type: ignore[attr-defined]
            accepted_events=accepted_events,
            event_name=event_name,
            return_types=spec.return_types,
            context_parameter=spec.context_parameter,
            context_state_type=spec.context_state_type,
            num_workers=num_workers,
            retry_policy=retry_policy,
            resources=spec.resources,
        )

        # If this is a free function, call add_step() explicitly.
        if is_free_function(func.__qualname__):
            if workflow is None:
                msg = f"To decorate {func.__name__} please pass a workflow class to the @step decorator."
                raise WorkflowValidationError(msg)
            workflow.add_step(func)

        return func

    if len(args):
        # The decorator was used without parentheses, like `@step`
        func = args[0]
        decorator(func)
        return func
    return decorator
