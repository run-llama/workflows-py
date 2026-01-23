# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import pytest
from pydantic import BaseModel
from workflows.decorators import step
from workflows.errors import WorkflowValidationError
from workflows.events import StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig
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


@pytest.mark.asyncio
async def test_resource_config_in_factory_with_future_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ResourceConfig in resource factories should resolve under future annotations."""
    monkeypatch.chdir(tmp_path)

    class SimpleConfig(BaseModel):
        name: str

    class SimpleClient:
        def __init__(self, config: SimpleConfig) -> None:
            self.config = config

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"name": "demo"}))

    def get_client(
        config: Annotated[SimpleConfig, ResourceConfig(config_file=str(config_path))],
    ) -> SimpleClient:
        return SimpleClient(config=config)

    class WorkflowWithConfig(Workflow):
        @step
        async def start_step(
            self,
            ev: StartEvent,
            client: Annotated[SimpleClient, Resource(get_client)],
        ) -> StopEvent:
            assert client.config.name == "demo"
            return StopEvent(result="done")

    wf = WorkflowWithConfig(disable_validation=True)
    await wf.run()


@pytest.mark.asyncio
async def test_nested_resource_in_resource_with_future_annotations() -> None:
    """Nested Resource dependencies should resolve under future annotations."""

    class DBConnection:
        def __init__(self) -> None:
            self.connected = True

    class Repository:
        def __init__(self, db: DBConnection) -> None:
            self.db = db

    def get_db() -> DBConnection:
        return DBConnection()

    def get_repo(
        db: Annotated[DBConnection, Resource(get_db)],
    ) -> Repository:
        return Repository(db=db)

    class WorkflowWithNestedResources(Workflow):
        @step
        async def start_step(
            self,
            ev: StartEvent,
            repo: Annotated[Repository, Resource(get_repo)],
        ) -> StopEvent:
            assert repo.db.connected
            return StopEvent(result="done")

    wf = WorkflowWithNestedResources(disable_validation=True)
    result = await wf.run()
    assert result == "done"
