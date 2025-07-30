import asyncio

from workflows import Workflow, step
from workflows.context import Context
from workflows.events import Event, StartEvent, StopEvent
from workflows.server import WorkflowServer


class StreamEvent(Event):
    sequence: int


class GreetingWorkflow(Workflow):
    @step
    async def greet(self, ctx: Context, ev: StartEvent) -> StopEvent:
        for i in range(3):
            ctx.write_event_to_stream(StreamEvent(sequence=i))
            await asyncio.sleep(0.3)

        name = getattr(ev, "name", "World")
        return StopEvent(result=f"Hello, {name}!")


server = WorkflowServer()
server.add_workflow("greeting", GreetingWorkflow())


# This is to test the server locally, run with `uv run python app.py`
if __name__ == "__main__":
    try:
        asyncio.run(server.serve(host="localhost", port=8080))
    except KeyboardInterrupt:
        pass
