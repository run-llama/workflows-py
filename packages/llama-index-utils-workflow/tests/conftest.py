import pytest

from workflows.workflow import Workflow
from workflows.decorators import step
from workflows.events import StartEvent, StopEvent, Event
from pydantic import Field


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


@pytest.fixture()
def workflow() -> Workflow:
    return DummyWorkflow()


@pytest.fixture()
def events() -> list[type[Event]]:
    return [OneTestEvent, AnotherTestEvent]
