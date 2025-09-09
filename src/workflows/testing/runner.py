from typing import AsyncContextManager, Any, Optional, TYPE_CHECKING
from types import TracebackType
from collections import Counter

if TYPE_CHECKING:
    from workflows import Workflow
from workflows.events import StartEvent, Event, EventType


class WorkflowTestRunner(AsyncContextManager["WorkflowTestRunner"]):
    def __init__(self, workflow: "Workflow"):
        self._workflow = workflow

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Called when exiting the 'async with' block"""
        return not exc_type and not exc_val and not exc_tb

    async def run(
        self,
        start_event: StartEvent,
        expose_internal: bool = True,
        exclude_events: Optional[list[EventType]] = None,
    ) -> tuple[list[Event], dict[EventType, int], Any]:
        """
        Run a workflow end-to-end and collect the events that are streamed during its execution.

        Args:
            start_event (StartEvent): The input event for the workflow
            expose_internal (bool): Whether or not to expose internal events. Defaults to True if not set.
            exclude_events. (list[EventType]): A list of event types to exclude from the collected events. Defaults to None if not set.

        Returns:
            A tuple containing the list of collected events, a dictionary mapping each event type with its count in the collected events and the result of the workflow run.

        Example:
            ```
            wf = GreetingWorkflow()

            async with WorkflowTestRunner(wf) as test_runner:
                collected, ev_types, result = await test_runner.run(start_event=StartEvent(name="Adam", greeting="hello"))
                assert len(collected) == 22
                assert ev_types.get(StepStateChanged, 0) == 8
                assert str(result) == "hello Adam!"
            ```
        """
        handler = self._workflow.run(start_event=start_event)
        collected_events: list[Event] = []
        async for event in handler.stream_events(expose_internal=expose_internal):
            if exclude_events and type(event) in exclude_events:
                continue
            collected_events.append(event)
        result = await handler
        event_freqs: dict[EventType, int] = dict(
            Counter([type(ev) for ev in collected_events])
        )
        return collected_events, event_freqs, result
