"""Test helpers for llama-index workflow integration tests."""

from typing import Callable, List

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)


def response_generator_from_list(responses: List[ChatMessage]) -> Callable:
    """Create a response generator that cycles through a list of responses."""
    index = 0

    def generator(messages: List[ChatMessage]) -> ChatMessage:
        nonlocal index
        if not responses:
            return ChatMessage(role=MessageRole.ASSISTANT, content=None)
        msg = responses[index]
        index = (index + 1) % len(responses)
        return msg

    return generator


def make_tool_call_response(
    tool_name: str,
    tool_kwargs: dict | None = None,
    content: str = "",
    tool_id: str = "call_1",
) -> ChatMessage:
    """Create a ChatMessage with a tool call."""
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=[
            TextBlock(text=content),
            ToolCallBlock(
                tool_call_id=tool_id, tool_name=tool_name, tool_kwargs=tool_kwargs or {}
            ),
        ],
    )


def make_text_response(content: str) -> ChatMessage:
    """Create a simple text response."""
    return ChatMessage(role=MessageRole.ASSISTANT, content=content)
