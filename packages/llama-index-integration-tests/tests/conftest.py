"""Shared fixtures for llama-index workflow integration tests."""

from typing import Callable, List, Union

import pytest
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import BaseTool
from llama_index_integration_tests.helpers import (
    make_text_response,
    response_generator_from_list,
)


@pytest.fixture
def create_workflow():
    """Factory fixture to create AgentWorkflow with FunctionAgent.

    FunctionAgent is used because it supports function calling via
    tool_calls in additional_kwargs, which our mock responses use.
    """

    def _create(
        name: str = "test_agent",
        tools: List[Union[BaseTool, Callable]] | None = None,
        responses: List[ChatMessage] | None = None,
        initial_state: dict | None = None,
        **kwargs,
    ):
        if responses is None:
            responses = [make_text_response("Done")]

        llm = MockFunctionCallingLLM(
            response_generator=response_generator_from_list(responses)
        )

        agent = FunctionAgent(
            name=name,
            description=f"Test {name}",
            tools=tools or [],
            llm=llm,
        )

        return AgentWorkflow(
            agents=[agent],
            root_agent=name,
            initial_state=initial_state,
            **kwargs,
        )

    return _create


@pytest.fixture(params=["function", "react"])
def agent_type(request) -> str:
    """Parameterized fixture for agent type.

    Use this only for tests that don't involve tool invocations,
    since ReActAgent uses text parsing instead of function calling.
    """
    return request.param


@pytest.fixture
def create_simple_workflow(agent_type: str):
    """Factory for simple workflows (no tools) parameterized by agent type.

    Use this for tests that want to verify behavior across both
    FunctionAgent and ReActAgent without tool invocations.
    """

    def _create(
        name: str = "test_agent",
        responses: List[ChatMessage] | None = None,
        **kwargs,
    ):
        if responses is None:
            responses = [make_text_response("Done")]

        llm = MockFunctionCallingLLM(
            response_generator=response_generator_from_list(responses)
        )

        if agent_type == "function":
            agent = FunctionAgent(
                name=name,
                description=f"Test {name}",
                llm=llm,
            )
        else:
            agent = ReActAgent(
                name=name,
                description=f"Test {name}",
                llm=llm,
            )

        return AgentWorkflow(
            agents=[agent],
            root_agent=name,
            **kwargs,
        )

    return _create
