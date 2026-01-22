"""Integration tests for AgentWorkflow (multi-agent) abstraction.

These tests verify that the AgentWorkflow class from llama-index-core
works correctly with the workflows package.
"""

from typing import List

from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool


def _response_generator_from_list(responses: List[ChatMessage]):
    """Helper to create a response generator from a list of responses."""
    index = 0

    def generator(messages: List[ChatMessage]) -> ChatMessage:
        nonlocal index
        if not responses:
            return ChatMessage(role=MessageRole.ASSISTANT, content=None)
        msg = responses[index]
        index = (index + 1) % len(responses)
        return msg

    return generator


async def test_multi_agent_workflow_basic() -> None:
    """Test basic multi-agent workflow execution."""

    def greet(name: str) -> str:
        """Greet a person."""
        return f"Hello, {name}!"

    agent1 = FunctionAgent(
        name="greeter",
        description="Greets people",
        system_prompt="You are a friendly greeter.",
        tools=[FunctionTool.from_defaults(fn=greet)],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Hello! How can I help you today?",
                    )
                ]
            )
        ),
    )

    workflow = AgentWorkflow(agents=[agent1], root_agent="greeter")

    handler = workflow.run(user_msg="Hi there!")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None
    assert isinstance(response, AgentOutput)


async def test_multi_agent_workflow_with_handoff() -> None:
    """Test multi-agent workflow with handoff between agents."""

    def search(query: str) -> str:
        """Search for information."""
        return f"Found results for: {query}"

    def summarize(text: str) -> str:
        """Summarize text."""
        return f"Summary: {text[:50]}..."

    # Create a searcher agent that can hand off to summarizer
    searcher = FunctionAgent(
        name="searcher",
        description="Searches for information",
        system_prompt="You search for information.",
        tools=[FunctionTool.from_defaults(fn=search)],
        can_handoff_to=["summarizer"],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="I found the information you requested.",
                    )
                ]
            )
        ),
    )

    # Create a summarizer agent
    summarizer = FunctionAgent(
        name="summarizer",
        description="Summarizes information",
        system_prompt="You summarize information.",
        tools=[FunctionTool.from_defaults(fn=summarize)],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Here is the summary of the information.",
                    )
                ]
            )
        ),
    )

    workflow = AgentWorkflow(
        agents=[searcher, summarizer],
        root_agent="searcher",
    )

    handler = workflow.run(user_msg="Search for Python tutorials")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None


async def test_multi_agent_workflow_with_memory() -> None:
    """Test multi-agent workflow preserves memory."""

    agent = FunctionAgent(
        name="assistant",
        description="A helpful assistant",
        system_prompt="You are a helpful assistant.",
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="I remember our conversation.",
                    )
                ]
            )
        ),
    )

    workflow = AgentWorkflow(agents=[agent], root_agent="assistant")
    memory = ChatMemoryBuffer.from_defaults()

    handler = workflow.run(user_msg="Remember this: the sky is blue", memory=memory)
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None

    # Memory should contain messages
    messages = await memory.aget()
    assert len(messages) > 0


async def test_multi_agent_workflow_event_streaming() -> None:
    """Test that events are properly streamed from multi-agent workflow."""

    agent = FunctionAgent(
        name="streamer",
        description="Streams events",
        system_prompt="You stream events.",
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Streaming response.",
                    )
                ]
            )
        ),
    )

    workflow = AgentWorkflow(agents=[agent], root_agent="streamer")

    handler = workflow.run(user_msg="Stream some events")

    events = []
    async for event in handler.stream_events():
        events.append(event)

    response = await handler
    assert response is not None
    # Should have received some events during streaming
    assert len(events) > 0
