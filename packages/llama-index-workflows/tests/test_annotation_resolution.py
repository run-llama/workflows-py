# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

import pytest
from workflows.decorators import step
from workflows.errors import WorkflowValidationError
from workflows.events import StartEvent, StopEvent
from workflows.resource import Resource
from workflows.workflow import Workflow

if TYPE_CHECKING:

    class MissingReturn:  # pragma: no cover
        pass


def test_step_decorator_resolves_local_resource_factory_with_future_annotations() -> (
    None
):
    class Repo:
        pass

    def get_repo() -> Repo:
        return Repo()

    class LocalWorkflow(Workflow):
        @step
        async def start(
            self,
            ev: StartEvent,
            repo: Annotated[Repo, Resource(get_repo)],
        ) -> StopEvent:
            return StopEvent(result="ok")

    resources = LocalWorkflow.start._step_config.resources
    assert len(resources) == 1
    assert resources[0].name == "repo"
    assert resources[0].type_annotation is Repo


def test_step_decorator_resolves_local_return_type_with_future_annotations() -> None:
    class ResultEvent(StopEvent):
        pass

    class LocalWorkflow(Workflow):
        @step
        async def start(self, ev: StartEvent) -> ResultEvent:
            return ResultEvent()

    return_types = LocalWorkflow.start._step_config.return_types
    assert return_types == [ResultEvent]


def test_step_decorator_error_message_for_unresolved_string_annotations() -> None:
    with pytest.raises(
        WorkflowValidationError,
        match="Failed to resolve type annotations",
    ):

        class BadWorkflow(Workflow):
            @step
            async def start(self, ev: StartEvent) -> "MissingReturn":
                return cast("MissingReturn", StopEvent(result="ok"))
