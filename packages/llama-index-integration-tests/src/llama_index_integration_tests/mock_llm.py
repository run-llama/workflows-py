"""Custom mock LLM for integration tests.

This provides a MockFunctionCallingLLM that works regardless of the
llama-index-core version installed, allowing tests to run against
any compatible version.
"""

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    AsyncGenerator,
    Generator,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import ToolSelection

from pydantic import Field, PrivateAttr


ResponseGenerator = Callable[[Sequence[ChatMessage]], ChatMessage]


def _default_response_generator(messages: Sequence[ChatMessage]) -> ChatMessage:
    """Default response generator that echoes the last message's content."""
    if not messages:
        return ChatMessage(role=MessageRole.ASSISTANT, content="<empty>")

    last_msg = messages[-1]
    content = last_msg.content or "<empty>"
    return ChatMessage(role=MessageRole.ASSISTANT, content=content)


class MockFunctionCallingLLM(FunctionCallingLLM):
    """Mock LLM that supports function calling for testing purposes.

    This is a simplified version that works with the base FunctionCallingLLM API.
    """

    max_tokens: Optional[int] = Field(default=None)

    _response_generator: Optional[ResponseGenerator] = PrivateAttr(default=None)

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        response_generator: Optional[ResponseGenerator] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            callback_manager=callback_manager or CallbackManager([]),
            **kwargs,
        )
        self._response_generator = response_generator

    @classmethod
    def class_name(cls) -> str:
        return "MockFunctionCallingLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            is_function_calling_model=True,
            num_output=self.max_tokens or -1,
        )

    def _get_response(self, messages: Sequence[ChatMessage]) -> ChatMessage:
        """Get response using the generator or default."""
        if self._response_generator is not None:
            return self._response_generator(messages)
        return _default_response_generator(messages)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=prompt)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen() -> Generator[CompletionResponse, None, None]:
            yield CompletionResponse(text=prompt, delta=prompt)

        return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=prompt)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> AsyncGenerator[CompletionResponse, None]:
            yield CompletionResponse(text=prompt, delta=prompt)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        response_msg = self._get_response(messages)
        content = response_msg.content or ""
        return ChatResponse(
            message=response_msg,
            delta=content,
            raw={"content": content},
        )

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return self.chat(messages=messages)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        response_msg = self._get_response(messages)
        content = response_msg.content or ""

        def _gen() -> Generator[ChatResponse, None, None]:
            yield ChatResponse(
                message=response_msg,
                delta=content,
                raw={"content": content},
            )

        return _gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = self._get_response(messages)
        content = response_msg.content or ""

        async def _gen() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(
                message=response_msg,
                delta=content,
                raw={"content": content},
            )

        return _gen()

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Optional[str | ChatMessage] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Prepare the arguments needed to let the LLM chat with tools."""
        messages = list(chat_history or [])

        if user_msg:
            if isinstance(user_msg, str):
                messages.append(ChatMessage(role=MessageRole.USER, content=user_msg))
            else:
                messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tools,
        }

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = False, **kwargs: Any
    ) -> List[ToolSelection]:
        """Extract tool calls from response.

        Checks additional_kwargs for tool_calls (for test mocks).
        """
        if "tool_calls" in response.message.additional_kwargs:
            return response.message.additional_kwargs.get("tool_calls", [])
        return []
