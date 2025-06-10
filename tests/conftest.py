import pytest
from pydantic import Field

from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.workflow import Workflow


class OneTestEvent(Event):
    test_param: str = Field(default="test")


class AnotherTestEvent(Event):
    another_test_param: str = Field(default="another_test")


class LastEvent(Event):
    pass


class DummyWorkflow(Workflow):
    @step
    async def start_step(self, ev: StartEvent) -> OneTestEvent:
        return OneTestEvent()

    @step
    async def middle_step(self, ev: OneTestEvent) -> LastEvent:
        return LastEvent()

    @step
    async def end_step(self, ev: LastEvent) -> StopEvent:
        return StopEvent(result="Workflow completed")


@pytest.fixture()
def workflow() -> Workflow:
    return DummyWorkflow()


@pytest.fixture()
def events() -> list:
    return [OneTestEvent, AnotherTestEvent]


@pytest.fixture()
def ctx() -> Context:
    return Context(workflow=DummyWorkflow())
