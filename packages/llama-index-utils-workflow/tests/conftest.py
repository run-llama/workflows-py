from typing import Annotated

import pytest
from pydantic import Field
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource
from workflows.workflow import Workflow


class OneTestEvent(Event):
    test_param: str = Field(default="test")


class AnotherTestEvent(Event):
    another_test_param: str = Field(default="another_test")


class LastEvent(Event):
    pass


class DummyWorkflow(Workflow):
    @step()
    async def start_step(self, ev: StartEvent) -> OneTestEvent:
        return OneTestEvent()

    @step()
    async def middle_step(self, ev: OneTestEvent) -> LastEvent:
        return LastEvent()

    @step()
    async def end_step(self, ev: LastEvent) -> StopEvent:
        return StopEvent(result="Workflow completed")


# --- Resource-based workflow for testing ---


class DatabaseClient:
    """A mock database client for testing resources."""

    pass


def get_database_client() -> DatabaseClient:
    """Factory function to create a database client.

    This is a test docstring that should appear in the resource metadata.
    """
    return DatabaseClient()


class CacheClient:
    """A mock cache client for testing resources."""

    pass


def get_cache_client() -> CacheClient:
    """Factory function to create a cache client."""
    return CacheClient()


class ResourceWorkflow(Workflow):
    """A workflow with resource dependencies for testing visualization."""

    @step()
    async def start_step(self, ev: StartEvent) -> OneTestEvent:
        return OneTestEvent()

    @step()
    async def step_with_db(
        self,
        ev: OneTestEvent,
        db_client: Annotated[DatabaseClient, Resource(get_database_client)],
    ) -> LastEvent:
        return LastEvent()

    @step()
    async def step_with_both_resources(
        self,
        ev: LastEvent,
        db: Annotated[DatabaseClient, Resource(get_database_client)],
        cache: Annotated[CacheClient, Resource(get_cache_client)],
    ) -> StopEvent:
        return StopEvent(result="Workflow completed")


@pytest.fixture()
def workflow() -> Workflow:
    return DummyWorkflow()


@pytest.fixture()
def workflow_with_resources() -> Workflow:
    return ResourceWorkflow()


@pytest.fixture()
def events() -> list[type[Event]]:
    return [OneTestEvent, AnotherTestEvent]


# --- Nested workflow for testing ---


class ChildWorkflowA(Workflow):
    @step()
    async def child_start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Child processed")


class ParentWorkflow(Workflow):
    @step()
    async def parent_start(self, ev: StartEvent) -> Event:
        # Instantiate the nested workflow. The drawing function will inspect this.
        child_wf = ChildWorkflowA()
        # The actual run logic is not important for drawing.
        _ = await child_wf.run(input="dummy")
        return Event(result="some result")

    @step()
    async def parent_end(self, ev: Event) -> StopEvent:
        return StopEvent(result="Final Result")


@pytest.fixture()
def nested_workflow() -> Workflow:
    return ParentWorkflow()
