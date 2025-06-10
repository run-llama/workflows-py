import re

import pytest

from workflows.decorators import step
from workflows.errors import WorkflowValidationError
from workflows.events import Event, StartEvent, StopEvent
from workflows.workflow import Workflow


def test_decorated_config(workflow: Workflow) -> None:
    def f(self, ev: Event) -> Event:  # type: ignore
        return Event()

    res = step(workflow=workflow.__class__)(f)
    config = getattr(res, "__step_config")
    assert config.accepted_events == [Event]
    assert config.event_name == "ev"
    assert config.return_types == [Event]


def test_decorate_method() -> None:
    class TestWorkflow(Workflow):
        @step
        def f1(self, ev: StartEvent) -> Event:
            return ev

        @step
        def f2(self, ev: Event) -> StopEvent:
            return StopEvent()

    wf = TestWorkflow()
    assert getattr(wf.f1, "__step_config")
    assert getattr(wf.f2, "__step_config")


def test_decorate_wrong_signature() -> None:
    def f() -> None:
        pass

    with pytest.raises(WorkflowValidationError):
        step()(f)


def test_decorate_free_function() -> None:
    class TestWorkflow(Workflow):
        pass

    @step(workflow=TestWorkflow)
    def f(ev: Event) -> Event:
        return Event()

    assert TestWorkflow._step_functions == {"f": f}


def test_decorate_free_function_wrong_decorator() -> None:
    with pytest.raises(
        WorkflowValidationError,
        match=re.escape(
            "To decorate f please pass a workflow class to the @step decorator."
        ),
    ):

        @step
        def f(ev: Event) -> Event:
            return Event()


def test_decorate_free_function_wrong_num_workers() -> None:
    class TestWorkflow(Workflow):
        pass

    with pytest.raises(
        WorkflowValidationError, match="num_workers must be an integer greater than 0"
    ):

        @step(workflow=TestWorkflow, num_workers=0)
        def f1(ev: Event) -> Event:
            return Event()

    with pytest.raises(
        WorkflowValidationError, match="num_workers must be an integer greater than 0"
    ):

        @step(workflow=TestWorkflow, num_workers=0.5)  # type: ignore
        def f2(ev: Event) -> Event:
            return Event()
