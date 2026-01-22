"""Integration tests for ReActAgent workflow abstraction.

These tests verify that the ReActAgent class from llama-index-core
works correctly with the workflows package.
"""

from typing import List

import pytest

from llama_index.core.agent.workflow import ReActAgent, AgentInput
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


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


@pytest.fixture()
def calculator_agent() -> ReActAgent:
    return ReActAgent(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant.",
        tools=[
            FunctionTool.from_defaults(fn=add),
            FunctionTool.from_defaults(fn=subtract),
        ],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content='Thought: I need to add these numbers\nAction: add\nAction Input: {"a": 5, "b": 3}\n',
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=r"Thought: The result is 8\Answer: The sum is 8",
                    ),
                ]
            )
        ),
    )


@pytest.fixture()
def retry_calculator_agent() -> ReActAgent:
    return ReActAgent(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant.",
        tools=[
            FunctionTool.from_defaults(fn=add),
            FunctionTool.from_defaults(fn=subtract),
        ],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content='Thought: I need to add these numbers\nAction: add\n{"a": 5 "b": 3}\n',
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content='Thought: I need to add these numbers\nAction: add\nAction Input: {"a": 5, "b": 3}\n',
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=r"Thought: The result is 8\nAnswer: The sum is 8",
                    ),
                ]
            )
        ),
    )


async def test_single_react_agent(calculator_agent: ReActAgent) -> None:
    """Verify execution of basic ReAct single agent."""
    memory = ChatMemoryBuffer.from_defaults()
    handler = calculator_agent.run(user_msg="Can you add 5 and 3?", memory=memory)

    events = []
    async for event in handler.stream_events():
        events.append(event)

    response = await handler

    assert "8" in str(response.response)


async def test_single_react_agent_retry(retry_calculator_agent: ReActAgent) -> None:
    """Verify execution of basic ReAct single agent with retry due to output parsing error."""
    memory = ChatMemoryBuffer.from_defaults()
    handler = retry_calculator_agent.run(user_msg="Can you add 5 and 3?", memory=memory)

    events = []
    contains_error_message = False
    async for event in handler.stream_events():
        events.append(event)
        if isinstance(event, AgentInput):
            content = event.input[-1].content
            if content and "Error while parsing the output" in content:
                contains_error_message = True

    assert contains_error_message

    response = await handler

    assert "8" in str(response.response)


async def test_react_agent_with_memory() -> None:
    """Test ReActAgent preserves conversation memory."""

    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"The weather in {city} is sunny."

    agent = ReActAgent(
        name="weather_agent",
        description="Gets weather information",
        system_prompt="You are a weather assistant.",
        tools=[FunctionTool.from_defaults(fn=get_weather)],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Thought: I need to get the weather\nAction: get_weather\nAction Input: {\"city\": \"London\"}\n",
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Thought: I have the weather info\nAnswer: The weather in London is sunny.",
                    ),
                ]
            )
        ),
    )

    memory = ChatMemoryBuffer.from_defaults()
    handler = agent.run(user_msg="What's the weather in London?", memory=memory)

    async for _ in handler.stream_events():
        pass

    response = await handler

    assert response is not None
    # Memory should contain the conversation
    messages = await memory.aget()
    assert len(messages) > 0
