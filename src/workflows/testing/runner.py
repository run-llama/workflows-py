from typing import AsyncContextManager, Any, Optional, TYPE_CHECKING
from types import TracebackType
from collections import Counter
from dataclasses import dataclass

if TYPE_CHECKING:
    from workflows import Workflow
from workflows.events import StartEvent, Event, EventType


@dataclass
class WorkflowTestResult:
    """Container for workflow test results"""

    collected: list[Event]
    event_types: dict[EventType, int]
    result: Any


class WorkflowTestRunner(AsyncContextManager["WorkflowTestRunner"]):
    def __init__(
        self,
        workflow: "Workflow",
        start_event: StartEvent,
        expose_internal: bool = True,
        exclude_events: Optional[list[EventType]] = None,
    ):
        """
        Run a workflow end-to-end and collect the events that are streamed during its execution.

        Args:
            start_event (StartEvent): The input event for the workflow
            expose_internal (bool): Whether or not to expose internal events. Defaults to True if not set.
            exclude_events. (list[EventType]): A list of event types to exclude from the collected events. Defaults to None if not set.

        Returns:
            WorkflowTestResult

        Example:
            ```
            wf = GreetingWorkflow()
            runner = WorkflowTestRunner(wf)

            async with runner as test_wf:
                if test_wf.result:
                    assert test_wf.result.collected == 22
                    assert test_result.result.event_types.get(StepStateChanged, 0) == 8
                    assert str(test_result.result.result) == "hello Adam!"
            ```
        """
        self._workflow = workflow
        self._start_event = start_event
        self._expose_internal = expose_internal
        self._exclude_events = exclude_events
        self._handler = None
        self._result: Optional[WorkflowTestResult] = None

    async def __aenter__(self) -> "WorkflowTestRunner":
        await self._run()
        return self

    @property
    def result(self) -> Optional[WorkflowTestResult]:
        return self._result

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Called when exiting the 'async with' block"""
        return not exc_type and not exc_val and not exc_tb

    async def _run(self) -> None:
        handler = self._workflow.run(start_event=self._start_event)
        collected_events: list[Event] = []
        async for event in handler.stream_events(expose_internal=self._expose_internal):
            if self._exclude_events and type(event) in self._exclude_events:
                continue
            collected_events.append(event)
        result = await handler
        event_freqs: dict[EventType, int] = dict(
            Counter([type(ev) for ev in collected_events])
        )
        self._result = WorkflowTestResult(
            collected=collected_events, result=result, event_types=event_freqs
        )
