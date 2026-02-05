"""MCP server example: expose workflows as MCP tools.

Run with:
    uv run python examples/server/mcp_server_example.py

Then connect Claude Code (or any MCP client) to:
    http://localhost:8000/mcp/

Claude Code config (~/.claude/mcp.json):
    {
      "mcpServers": {
        "workflows": {
          "url": "http://localhost:8000/mcp/"
        }
      }
    }
"""

import asyncio

from llama_agents.mcp import MCPToolConfig, MCPWorkflowServer
from llama_agents.server import WorkflowServer
from workflows import Workflow, step
from workflows.context import Context
from workflows.events import Event, StartEvent, StopEvent

# ---------------------------------------------------------------------------
# Workflow definitions
# ---------------------------------------------------------------------------


class SearchInput(StartEvent):
    query: str
    limit: int = 10


class SearchWorkflow(Workflow):
    """Simple search that returns results immediately."""

    @step
    async def search(self, ev: SearchInput) -> StopEvent:
        # Replace with actual search logic
        results = [f"Result {i + 1} for '{ev.query}'" for i in range(ev.limit)]
        return StopEvent(result="\n".join(results))


class ProgressEvent(Event):
    step_name: str
    progress: int
    message: str


class AnalyzeInput(StartEvent):
    topic: str
    depth: str = "shallow"


class AnalyzeWorkflow(Workflow):
    """Long-running analysis that streams progress events."""

    @step
    async def analyze(self, ctx: Context, ev: AnalyzeInput) -> StopEvent:
        steps = ["gathering data", "processing", "summarizing"]
        for i, step_name in enumerate(steps):
            ctx.write_event_to_stream(
                ProgressEvent(
                    step_name=step_name,
                    progress=int((i + 1) / len(steps) * 100),
                    message=f"{step_name} for '{ev.topic}' ({ev.depth})",
                )
            )
            await asyncio.sleep(0.5)

        return StopEvent(
            result=f"Analysis complete for '{ev.topic}' at {ev.depth} depth. "
            f"Processed {len(steps)} stages."
        )


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------


async def main() -> None:
    server = WorkflowServer()
    server.add_workflow("search", SearchWorkflow())
    server.add_workflow("analyze", AnalyzeWorkflow())

    mcp = MCPWorkflowServer(
        server,
        name="Example Workflow Tools",
        tools={
            "search": MCPToolConfig(
                description="Search for information by query. Returns matching results.",
            ),
            "analyze": MCPToolConfig(
                mode="async",
                description="Run a deep analysis on a topic. Streams progress updates.",
            ),
        },
    )
    mcp.mount()

    print("Starting server at http://localhost:8000")  # noqa: T201
    print("MCP endpoint: http://localhost:8000/mcp/")  # noqa: T201
    print("HTTP API: http://localhost:8000/workflows")  # noqa: T201
    await server.serve(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
