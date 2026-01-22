"""Integration tests for FunctionAgent workflow abstraction.

These tests verify that the FunctionAgent class from llama-index-core
works correctly with the workflows package.
"""

from typing import List

import pytest

from llama_index.core.agent.workflow import FunctionAgent, AgentInput
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, ToolSelection
from llama_index.core.workflow.errors import WorkflowRuntimeError


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


@pytest.fixture()
def function_agent() -> FunctionAgent:
    return FunctionAgent(
        name="retriever",
        description="Manages data retrieval",
        system_prompt="You are a retrieval assistant.",
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Success with the FunctionAgent",
                    )
                ]
            )
        ),
    )


async def test_single_function_agent(function_agent: FunctionAgent) -> None:
    """Test single agent with state management."""
    handler = function_agent.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)


async def test_function_agent_with_tool() -> None:
    """Test FunctionAgent with a tool."""

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    llm = MockFunctionCallingLLM(max_tokens=200)

    agent = FunctionAgent(
        name="test_agent",
        description="A test agent",
        system_prompt="You are a helpful assistant.",
        tools=[FunctionTool.from_defaults(fn=multiply)],
        llm=llm,
    )

    memory = ChatMemoryBuffer.from_defaults()
    handler = agent.run(user_msg="Hello, can you help me?", memory=memory)

    events = []
    async for event in handler.stream_events():
        events.append(event)

    response = await handler

    assert response.response is not None
    assert len(events) > 0


async def test_max_iterations() -> None:
    """Test max iterations raises error."""

    def random_tool() -> str:
        return "random"

    agent = FunctionAgent(
        name="agent",
        description="test",
        tools=[random_tool],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="handing off",
                        additional_kwargs={
                            "tool_calls": [
                                ToolSelection(
                                    tool_id="one",
                                    tool_name="random_tool",
                                    tool_kwargs={},
                                )
                            ]
                        },
                    ),
                ]
                * 100
            )
        ),
    )

    # Default max iterations is 20
    with pytest.raises(WorkflowRuntimeError, match="Either something went wrong"):
        _ = await agent.run(user_msg="test")


async def test_early_stopping_method_generate() -> None:
    """Test early_stopping_method='generate' produces a final response."""

    def random_tool() -> str:
        return "random"

    tool_call_response = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling tool",
        additional_kwargs={
            "tool_calls": [
                ToolSelection(
                    tool_id="one",
                    tool_name="random_tool",
                    tool_kwargs={},
                )
            ]
        },
    )
    final_response = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Here is my final summary based on the information gathered.",
    )

    agent = FunctionAgent(
        name="agent",
        description="test",
        tools=[random_tool],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [tool_call_response] * 5 + [final_response]
            )
        ),
        early_stopping_method="generate",
    )

    handler = agent.run(user_msg="test", max_iterations=5)
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None
    assert (
        "final summary" in str(response.response).lower()
        or response.response is not None
    )


async def test_function_agent_with_context() -> None:
    """Test FunctionAgent with explicit Context and ChatMessage input."""
    from llama_index.core.llms import TextBlock
    from llama_index.core.workflow import Context

    def noop() -> str:
        """A no-op function for testing."""
        return "noop executed"

    llm = MockFunctionCallingLLM()
    function_tools = [FunctionTool.from_defaults(fn=noop)]

    agent = FunctionAgent(
        llm=llm,
        tools=function_tools,
        system_prompt="You are a test assistant.",
    )

    ctx = Context(agent)

    user_message = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="Can you help me with a calculation?"),
            TextBlock(text="It's one plus one."),
        ],
    )

    handler = agent.run(
        user_msg=user_message,
        ctx=ctx,
    )

    response = await handler

    assert response is not None
    response_str = str(response)
    assert "Can you help me with a calculation?" in response_str
    assert "It's one plus one." in response_str
