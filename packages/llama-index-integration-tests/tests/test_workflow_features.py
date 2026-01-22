"""Integration tests for workflow features used with llama-index agents.

These tests verify that core workflow features like wait_for_event,
context state management, and serialization work correctly when used
inside llama-index agent tools.
"""

from typing import List

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import ToolSelection
from workflows import Context
from workflows.context import PickleSerializer
from workflows.events import HumanResponseEvent, InputRequiredEvent


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


async def test_wait_for_event_in_tool() -> None:
    """Test wait_for_event works correctly inside an agent tool function.

    This is a critical pattern for human-in-the-loop workflows where
    a tool needs to pause execution and wait for external input.
    """

    async def human_input_tool(ctx: Context) -> str:
        """Tool that pauses workflow to get human input."""
        resp = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_event=InputRequiredEvent(prefix="Please provide your name"),
        )
        return f"Hello, {resp.response}!"

    agent = FunctionAgent(
        name="greeter",
        description="Greets users by name",
        tools=[human_input_tool],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="I'll ask for your name",
                        additional_kwargs={
                            "tool_calls": [
                                ToolSelection(
                                    tool_id="call_1",
                                    tool_name="human_input_tool",
                                    tool_kwargs={},
                                )
                            ]
                        },
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Greeted the user successfully",
                    ),
                ]
            )
        ),
    )

    workflow = AgentWorkflow(agents=[agent], root_agent="greeter")

    handler = workflow.run(user_msg="Please greet me")

    # Stream events until we hit the pause point
    ctx_dict = None
    async for ev in handler.stream_events():
        if isinstance(ev, InputRequiredEvent):
            assert ev.prefix == "Please provide your name"
            ctx_dict = handler.ctx.to_dict()
            await handler.cancel_run()
            break

    assert ctx_dict is not None, "Should have received InputRequiredEvent"

    # Resume with human response
    new_ctx = Context.from_dict(workflow, ctx_dict)
    handler = workflow.run(user_msg="Please greet me", ctx=new_ctx)
    handler.ctx.send_event(HumanResponseEvent(response="Alice"))

    response = await handler
    assert response is not None


async def test_tool_with_context_state() -> None:
    """Test that tools can access and modify workflow state via context.

    This verifies that the ctx.store pattern works correctly when
    tools need to maintain state across calls.
    """

    async def counter_tool(ctx: Context) -> str:
        """Tool that increments a counter in the workflow state."""
        state = await ctx.store.get("state")
        state["count"] += 1
        await ctx.store.set("state", state)
        return f"Counter is now {state['count']}"

    agent = FunctionAgent(
        name="counter",
        description="Maintains a counter",
        tools=[counter_tool],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Incrementing counter",
                        additional_kwargs={
                            "tool_calls": [
                                ToolSelection(
                                    tool_id="call_1",
                                    tool_name="counter_tool",
                                    tool_kwargs={},
                                )
                            ]
                        },
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Counter incremented",
                    ),
                ]
            )
        ),
    )

    workflow = AgentWorkflow(
        agents=[agent],
        root_agent="counter",
        initial_state={"count": 0},
    )

    handler = workflow.run(user_msg="Increment the counter")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None

    # Verify state was updated
    state = await handler.ctx.store.get("state")
    assert state["count"] == 1


async def test_context_pickle_serialization() -> None:
    """Test that workflow context can be pickle-serialized and restored.

    This is important for workflows that need to persist state across
    process restarts or be distributed across multiple workers.
    """

    async def pausable_tool(ctx: Context) -> str:
        """Tool that pauses for human input."""
        resp = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_event=InputRequiredEvent(prefix="Continue?"),
        )
        return f"Received: {resp.response}"

    agent = FunctionAgent(
        name="pausable",
        description="Can pause for input",
        tools=[pausable_tool],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Pausing for input",
                        additional_kwargs={
                            "tool_calls": [
                                ToolSelection(
                                    tool_id="call_1",
                                    tool_name="pausable_tool",
                                    tool_kwargs={},
                                )
                            ]
                        },
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Resumed successfully",
                    ),
                ]
            )
        ),
    )

    workflow = AgentWorkflow(agents=[agent], root_agent="pausable")

    handler = workflow.run(user_msg="Test pause")

    # Wait for pause, then serialize with pickle
    ctx_dict = None
    async for ev in handler.stream_events():
        if isinstance(ev, InputRequiredEvent):
            serializer = PickleSerializer()
            ctx_dict = handler.ctx.to_dict(serializer=serializer)
            await handler.cancel_run()
            break

    assert ctx_dict is not None

    # Deserialize and resume
    serializer = PickleSerializer()
    new_ctx = Context.from_dict(workflow, ctx_dict, serializer=serializer)

    handler = workflow.run(user_msg="Test pause", ctx=new_ctx)
    handler.ctx.send_event(HumanResponseEvent(response="yes"))

    response = await handler
    assert response is not None


async def test_multiple_tool_calls_with_state() -> None:
    """Test multiple sequential tool calls that share state."""

    async def add_item(item: str, ctx: Context) -> str:
        """Add an item to the list."""
        state = await ctx.store.get("state")
        state["items"].append(item)
        await ctx.store.set("state", state)
        return f"Added {item}"

    agent = FunctionAgent(
        name="list_manager",
        description="Manages a list of items",
        tools=[add_item],
        llm=MockFunctionCallingLLM(
            response_generator=_response_generator_from_list(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Adding first item",
                        additional_kwargs={
                            "tool_calls": [
                                ToolSelection(
                                    tool_id="call_1",
                                    tool_name="add_item",
                                    tool_kwargs={"item": "apple"},
                                )
                            ]
                        },
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Adding second item",
                        additional_kwargs={
                            "tool_calls": [
                                ToolSelection(
                                    tool_id="call_2",
                                    tool_name="add_item",
                                    tool_kwargs={"item": "banana"},
                                )
                            ]
                        },
                    ),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="Added both items to the list",
                    ),
                ]
            )
        ),
    )

    workflow = AgentWorkflow(
        agents=[agent],
        root_agent="list_manager",
        initial_state={"items": []},
    )

    handler = workflow.run(user_msg="Add apple and banana")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert response is not None

    # Verify both items were added
    state = await handler.ctx.store.get("state")
    assert "apple" in state["items"]
    assert "banana" in state["items"]
    assert len(state["items"]) == 2
