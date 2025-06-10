import pytest

from workflows.context import Context
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.service import ServiceManager, ServiceNotFoundError
from workflows.workflow import Workflow


class ServiceWorkflow(Workflow):
    """This wokflow is only responsible to generate a number, it knows nothing about the caller."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        self._the_answer = kwargs.pop("the_answer", 42)
        super().__init__(*args, **kwargs)

    @step
    async def generate(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=self._the_answer)


class NumGenerated(Event):
    """To be used in the dummy workflow below."""

    num: int


class DummyWorkflow(Workflow):
    """
    This workflow needs a number, and it calls another workflow to get one.
    A service named "service_workflow" must be added to `DummyWorkflow` for
    the step to be able to use it (see below).
    This step knows nothing about the other workflow, it gets an instance
    and it only knows it has to call `run` on that instance.
    """

    @step
    async def get_a_number(
        self,
        ev: StartEvent,
        ctx: Context,
        service_workflow: ServiceWorkflow = ServiceWorkflow(),
    ) -> NumGenerated:
        res = await service_workflow.run()
        return NumGenerated(num=int(res))

    @step
    async def multiply(self, ev: NumGenerated) -> StopEvent:
        return StopEvent(ev.num * 2)


@pytest.mark.asyncio
async def test_e2e() -> None:
    wf = DummyWorkflow()
    # We are responsible for passing the ServiceWorkflow instances to the dummy workflow
    # and give it a name, in this case "service_workflow"
    wf.add_workflows(service_workflow=ServiceWorkflow(the_answer=1337))
    res = await wf.run()
    assert res == 2674


@pytest.mark.asyncio
async def test_default_value_for_service() -> None:
    wf = DummyWorkflow()
    # We don't add any workflow to leverage the default value defined by the user
    res = await wf.run()
    assert res == 84


def test_service_manager_add(workflow: Workflow) -> None:
    s = ServiceManager()
    s.add("test_id", workflow)
    assert s._services["test_id"] == workflow


def test_service_manager_get(workflow: Workflow) -> None:
    s = ServiceManager()
    s._services["test_id"] = workflow
    assert s.get("test_id") == workflow
    with pytest.raises(ServiceNotFoundError):
        s.get("not_found")
