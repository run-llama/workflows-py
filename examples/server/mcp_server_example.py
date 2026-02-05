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
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)

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


class InterviewInput(StartEvent):
    role: str


class CandidateAnswer(HumanResponseEvent):
    answer: str


class InterviewWorkflow(Workflow):
    """Multi-round interview that asks the user questions and scores answers.

    Demonstrates HITL: the workflow pauses at each question, waits for a
    human response, then continues to the next question or produces a
    final summary.
    """

    QUESTIONS = [
        "Tell me about your experience with {role}.",
        "What's the most challenging {role} problem you've solved?",
        "Where do you see yourself in 5 years as a {role}?",
    ]

    @step
    async def ask_first_question(
        self, ctx: Context, ev: InterviewInput
    ) -> InputRequiredEvent:
        await ctx.store.set("role", ev.role)
        await ctx.store.set("question_index", 0)
        await ctx.store.set("answers", [])
        question = self.QUESTIONS[0].format(role=ev.role)
        return InputRequiredEvent(prefix=question)  # type: ignore[call-arg]

    @step
    async def handle_answer(
        self, ctx: Context, ev: CandidateAnswer
    ) -> InputRequiredEvent | StopEvent:
        role = await ctx.store.get("role")
        idx = await ctx.store.get("question_index")
        answers: list[str] = await ctx.store.get("answers")
        answers.append(ev.answer)
        await ctx.store.set("answers", answers)

        next_idx = idx + 1
        if next_idx < len(self.QUESTIONS):
            await ctx.store.set("question_index", next_idx)
            question = self.QUESTIONS[next_idx].format(role=role)
            return InputRequiredEvent(prefix=question)  # type: ignore[call-arg]

        # All questions answered -- produce summary
        summary_parts = [f"Interview summary for {role} candidate:\n"]
        for i, (q, a) in enumerate(zip(self.QUESTIONS, answers), 1):
            summary_parts.append(f"Q{i}: {q.format(role=role)}")
            summary_parts.append(f"A{i}: {a}\n")
        summary_parts.append(f"Total questions: {len(answers)}")
        return StopEvent(result="\n".join(summary_parts))


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------


async def main() -> None:
    server = WorkflowServer()
    server.add_workflow("search", SearchWorkflow())
    server.add_workflow("analyze", AnalyzeWorkflow())
    server.add_workflow("interview", InterviewWorkflow())

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
            "interview": MCPToolConfig(
                mode="async",
                description=(
                    "Start a mock interview for a given role. "
                    "Asks 3 questions one at a time, waiting for answers. "
                    "Use interview_respond to answer each question."
                ),
            ),
        },
    )
    mcp.mount()

    print("Starting server at http://localhost:8000")  # noqa: T201
    print("MCP endpoint: http://localhost:8000/mcp/")  # noqa: T201
    print("HTTP API: http://localhost:8000/workflows")  # noqa: T201
    print("Tools: search, analyze, interview + interview_respond")  # noqa: T201
    await server.serve(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
