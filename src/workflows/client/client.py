# TOP-LEVEL
import time
import json
import httpx

# GENERATED CLASSES (CLIENTS)
from workflows.openapi_generated_client.workflows_api_client import (
    Client,
    AuthenticatedClient,
)

# GENERATED FUNCTIONS (API)
from workflows.openapi_generated_client.workflows_api_client.api.default.get_health import (
    asyncio as get_health,
)
from workflows.openapi_generated_client.workflows_api_client.api.default.get_workflows import (
    asyncio as get_workflows,
)
from workflows.openapi_generated_client.workflows_api_client.api.default.get_handlers import (
    asyncio as get_handlers,
)
from workflows.openapi_generated_client.workflows_api_client.api.default.get_results_handler_id import (
    asyncio as get_results_handler_id,
)
from workflows.openapi_generated_client.workflows_api_client.api.default.post_workflows_name_run import (
    asyncio as post_workflows_name_run,
)
from workflows.openapi_generated_client.workflows_api_client.api.default.post_workflows_name_run_nowait import (
    asyncio as post_workflows_name_run_nowait,
)
from workflows.openapi_generated_client.workflows_api_client.api.default.post_events_handler_id import (
    asyncio as post_events_handler_id,
)

# GENERATED TYPES (API)
from workflows.openapi_generated_client.workflows_api_client.models.post_workflows_name_run_body import (
    PostWorkflowsNameRunBody,
)
from workflows.openapi_generated_client.workflows_api_client.models.post_workflows_name_run_body_context import (
    PostWorkflowsNameRunBodyContext,
)
from workflows.openapi_generated_client.workflows_api_client.models.post_workflows_name_run_body_start_event import (
    PostWorkflowsNameRunBodyStartEvent,
)
from workflows.openapi_generated_client.workflows_api_client.models.post_workflows_name_run_body_kwargs import (
    PostWorkflowsNameRunBodyKwargs,
)
from workflows.openapi_generated_client.workflows_api_client.models.post_workflows_name_run_nowait_body import (
    PostWorkflowsNameRunNowaitBody,
)
from workflows.openapi_generated_client.workflows_api_client.models.post_workflows_name_run_nowait_body_context import (
    PostWorkflowsNameRunNowaitBodyContext,
)
from workflows.openapi_generated_client.workflows_api_client.models.post_workflows_name_run_nowait_body_kwargs import (
    PostWorkflowsNameRunNowaitBodyKwargs,
)
from workflows.openapi_generated_client.workflows_api_client.models.post_workflows_name_run_nowait_body_start_event import (
    PostWorkflowsNameRunNowaitBodyStartEvent,
)
from workflows.openapi_generated_client.workflows_api_client.models.post_events_handler_id_body import (
    PostEventsHandlerIdBody,
)
from workflows.openapi_generated_client.workflows_api_client.models.post_events_handler_id_response_200 import (
    PostEventsHandlerIdResponse200,
)
from workflows.openapi_generated_client.workflows_api_client.models.handler import (
    Handler,
)
from workflows.openapi_generated_client.workflows_api_client.models.handler_status import (
    HandlerStatus,
)
from workflows.openapi_generated_client.workflows_api_client.types import Unset, UNSET

# MISC
from workflows import Context
from workflows.events import StartEvent, Event
from workflows.context.serializers import JsonSerializer
from .utils import AuthDetails, EventDict
from typing import Literal, Optional, Any, Union, cast, AsyncGenerator


class WorkflowClient:
    def __init__(
        self,
        protocol: Literal["http", "https"] = "http",
        host: str = "localhost",
        port: int = 80,
        auth_details: Optional[AuthDetails] = None,
        raise_on_unexpected_status: bool = True,
        **kwargs: Any,
    ) -> None:
        self.base_url = f"{protocol}://{host}:{port}"
        if auth_details:
            self._client = AuthenticatedClient(
                base_url=self.base_url,
                token=auth_details.token,
                prefix=auth_details.prefix,
                auth_header_name=auth_details.auth_header_name,
                raise_on_unexpected_status=raise_on_unexpected_status,
                cookies=kwargs.get("cookies", {}),
                headers=kwargs.get("headers", {}),
                timeout=kwargs.get("timeout", None),
                verify_ssl=kwargs.get("verify_ssl", True),
                follow_redirects=kwargs.get("follow_redirects", False),
                httpx_args=kwargs.get("httpx_args", {}),
            )
        else:
            self._client = Client(
                base_url=self.base_url,
                raise_on_unexpected_status=raise_on_unexpected_status,
                cookies=kwargs.get("cookies", {}),
                headers=kwargs.get("headers", {}),
                timeout=kwargs.get("timeout", None),
                verify_ssl=kwargs.get("verify_ssl", True),
                follow_redirects=kwargs.get("follow_redirects", False),
                httpx_args=kwargs.get("httpx_args", {}),
            )

    async def is_healthy(self) -> bool:
        response = await get_health(client=self._client)
        if not response:
            return False
        return True

    async def ping(self) -> float:
        start = time.time()
        response = await get_health(client=self._client)
        if not response:
            return -1
        return (time.time() - start) * 1000

    async def list_workflows(self) -> list[str]:
        response = await get_workflows(client=self._client)
        if not response:
            return []
        return response.workflows

    async def run_workflow(
        self,
        workflow_name: str,
        start_event: Union[StartEvent, dict[str, Any], None] = None,
        context: Union[Context, dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Handler:
        if start_event and isinstance(start_event, StartEvent):
            start_event = start_event.model_dump()
        if context and isinstance(context, Context):
            try:
                context = context.to_dict()
            except Exception as e:
                raise ValueError(f"Impossible to serialize context because of: {e}")
        response = await post_workflows_name_run(
            name=workflow_name,
            client=self._client,
            body=PostWorkflowsNameRunBody(
                start_event=PostWorkflowsNameRunBodyStartEvent.from_dict(
                    cast(dict, start_event) or {}
                ),
                context=PostWorkflowsNameRunBodyContext.from_dict(context or {}),
                kwargs=PostWorkflowsNameRunBodyKwargs.from_dict(kwargs),
            ),
        )
        if isinstance(response, Handler):
            return response
        else:
            raise ValueError("Response was not properly generated")

    async def run_workflow_nowait(
        self,
        workflow_name: str,
        start_event: Union[StartEvent, dict[str, Any], None] = None,
        context: Union[Context, dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Handler:
        if start_event and isinstance(start_event, StartEvent):
            start_event = start_event.model_dump()
        if context and isinstance(context, Context):
            try:
                context = context.to_dict()
            except Exception as e:
                raise ValueError(f"Impossible to serialize context because of: {e}")
        response = await post_workflows_name_run_nowait(
            name=workflow_name,
            client=self._client,
            body=PostWorkflowsNameRunNowaitBody(
                start_event=PostWorkflowsNameRunNowaitBodyStartEvent.from_dict(
                    cast(dict, start_event) or {}
                ),
                context=PostWorkflowsNameRunNowaitBodyContext.from_dict(context or {}),
                kwargs=PostWorkflowsNameRunNowaitBodyKwargs.from_dict(kwargs),
            ),
        )
        if isinstance(response, Handler):
            return response
        else:
            raise ValueError("Response was not properly generated")

    async def get_workflow_events(
        self,
        handler: Handler,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream events using newline-delimited JSON format
        """
        url = f"/events/{handler.handler_id}?sse=false"
        client = self._client.get_async_httpx_client()
        try:
            async with client.stream("GET", url) as response:
                # Handle different response codes
                if response.status_code == 404:
                    raise ValueError("Handler not found")
                elif response.status_code == 204:
                    # Handler completed, no more events
                    return

                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():  # Skip empty lines
                        try:
                            event = json.loads(line.replace("\n", ""))
                            yield event
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON: {e}, data: {line}")
                            continue

        except httpx.TimeoutException:
            raise TimeoutError(
                f"Timeout waiting for events from handler {handler.handler_id}"
            )
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to event stream: {e}")

    async def get_workflow_handlers(self) -> list[Handler]:
        response = await get_handlers(client=self._client)
        if response:
            return response.handlers
        else:
            raise ValueError("Response was not properly generated")

    async def get_workflow_result(self, handler: Handler) -> Any:
        response = await get_results_handler_id(
            handler_id=handler.handler_id, client=self._client
        )
        if isinstance(response, Handler):
            if response.status == HandlerStatus.COMPLETED:
                return response.result
            elif response.status == HandlerStatus.RUNNING:
                return response.status.value
            else:
                return response.error
        elif isinstance(response, str):
            return response
        else:
            raise ValueError("Response was not properly generated")

    async def send_workflow_event(
        self,
        handler: Handler,
        event: Union[Event, EventDict, str],
        step: Optional[str] = None,
    ) -> str:
        if isinstance(event, Event):
            try:
                event = JsonSerializer().serialize(event)
            except Exception as e:
                raise ValueError(
                    f"It was not possible to serialize the event you want to send because of: {e}"
                )
        elif event is EventDict:
            event.setdefault("__is_pydantic", True)
            try:
                event = json.dumps(event)
            except Exception as e:
                raise ValueError(
                    f"It was not possible to serialize the event you want to send because of: {e}"
                )
        if not step:
            step: Unset = UNSET
        response = await post_events_handler_id(
            handler_id=handler.handler_id,
            client=self._client,
            body=PostEventsHandlerIdBody(event=cast(str, event), step=step),
        )
        if isinstance(response, PostEventsHandlerIdResponse200):
            return response.status.value
        else:
            raise ValueError("Response was not properly generated")
