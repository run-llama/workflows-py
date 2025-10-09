import httpx
import json

from typing import Literal, Any, Union, AsyncGenerator, AsyncIterator, Optional
from contextlib import asynccontextmanager
from workflows.events import StartEvent, Event
from workflows import Context
from workflows.server.server import HandlerDict
from workflows.server.utils import serdes_event
from workflows.types import RunResultT


class WorkflowClient:
    def __init__(
        self,
        protocol: Optional[Literal["http", "https"]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
        httpx_kwargs: Optional[dict[str, Any]] = None,
    ):
        # TODO: middleware-related logic
        self.protocol = protocol or "http"
        self.host = host or "localhost"
        self.port = port or 8000
        self.timeout = timeout or 600
        self.httpx_kwargs = httpx_kwargs or {}
        # TODO: add some basic TLS/verification and auth features

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(
            base_url=self.protocol + "://" + self.host + ":" + str(self.port),
            timeout=self.timeout,
            **self.httpx_kwargs,
        ) as client:
            yield client

    async def is_healthy(self) -> bool:
        """
        Check whether the workflow server is helathy or not

        Returns:
            bool: True if the workflow server is healthy, false if not
        """
        async with self._get_client() as client:
            response = await client.get("/health")
            response.raise_for_status()
            return response.json().get("status", "") == "healthy"

    async def list_workflows(self) -> list[str]:
        """
        List workflows

        Returns:
            list: List of workflow names available through the server.
        """
        async with self._get_client() as client:
            response = await client.get("/workflows")

            response.raise_for_status()

            return response.json()["workflows"]

    async def run_workflow(
        self,
        workflow_name: str,
        handler_id: Optional[str] = None,
        start_event: Union[StartEvent, dict[str, Any], str, None] = None,
        context: Union[Context, dict[str, Any], None] = None,
    ) -> Any:
        """
        Run the workflow and wait until completion.

        Args:
            start_event (Union[StartEvent, dict[str, Any], None]): start event class or dictionary representation (optional, defaults to None and get passed as an empty dictionary if not provided).
            context: Context or serialized representation of it (optional, defaults to None if not provided)
            handler_id (Optional[str]): Workflow handler identifier to continue from a previous completed run.

        Returns:
            Any: Result of the workflow
        """
        if start_event is not None:
            try:
                start_event = serdes_event(start_event)
            except Exception as e:
                raise ValueError(
                    f"Impossible to serialize the start event because of: {e}"
                )
        if isinstance(context, Context):
            try:
                context = context.to_dict()
            except Exception as e:
                raise ValueError(f"Impossible to serialize the context because of: {e}")
        request_body = {
            "start_event": start_event or "",
            "context": context or {},
        }
        if handler_id:
            request_body["handler_id"] = handler_id
        async with self._get_client() as client:
            response = await client.post(
                f"/workflows/{workflow_name}/run", json=request_body
            )

            response.raise_for_status()

            return response.json()["result"]

    async def run_workflow_nowait(
        self,
        workflow_name: str,
        handler_id: Optional[str] = None,
        start_event: Union[StartEvent, dict[str, Any], None] = None,
        context: Union[Context, dict[str, Any], None] = None,
    ) -> dict[str, Any]:
        """
        Run the workflow in the background.

        Args:
            start_event (Union[StartEvent, dict[str, Any], None]): start event class or dictionary representation (optional, defaults to None and get passed as an empty dictionary if not provided).
            context: Context or serialized representation of it (optional, defaults to None if not provided)
            handler_id (Optional[str]): Workflow handler identifier to continue from a previous completed run.

        Returns:
            dict[str, Any]: JSON representation of the handler running the workflow
        """
        if start_event is not None:
            try:
                start_event = serdes_event(start_event)
            except Exception as e:
                raise ValueError(
                    f"Impossible to serialize the start event because of: {e}"
                )
        if isinstance(context, Context):
            try:
                context = context.to_dict()
            except Exception as e:
                raise ValueError(f"Impossible to serialize the context because of: {e}")
        request_body = {
            "start_event": start_event or "{}",
            "context": context or {},
        }
        if handler_id:
            request_body["handler_id"] = handler_id
        async with self._get_client() as client:
            response = await client.post(
                f"/workflows/{workflow_name}/run-nowait", json=request_body
            )

            response.raise_for_status()

            return response.json()

    async def get_workflow_events(
        self,
        handler_id: str,
        include_internal_events: bool = False,
        lock_timeout: float = 1,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream events as they are produced by the workflow.

        Args:
            handler_id (str): ID of the handler running the workflow
            include_internal_events (bool): Include internal workflow events. Defaults to False.
            lock_timeout (float): Timeout (in seconds) for acquiring the lock to iterate over the events.

        Returns:
            AsyncGenerator[dict[str, Any], None]: Generator for the events that are streamed in the form of dictionaries.
        """
        incl_inter = "true" if include_internal_events else "false"
        url = f"/events/{handler_id}"

        async with self._get_client() as client:
            try:
                async with client.stream(
                    "GET",
                    url,
                    params={
                        "sse": "false",
                        "include_internal": incl_inter,
                        "acquire_timeout": lock_timeout,
                    },
                ) as response:
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
                    f"Timeout waiting for events from handler {handler_id}"
                )
            except httpx.RequestError as e:
                raise ConnectionError(f"Failed to connect to event stream: {e}")

    async def send_event(
        self,
        handler_id: str,
        event: Union[Event, dict[str, Any], str],
        step: Optional[str] = None,
    ) -> bool:
        """
        Send an event to the workflow.

        Args:
            handler_id (str): ID of the handler of the running workflow to send the event to
            event (Event | dict[str, Any] | str): Event to send, represented as an Event object, a dictionary or a serialized string.
            step (Optional[str]): Step to send the event to (optional, defaults to None)

        Returns:
            bool: Success status of the send operation
        """
        try:
            event = serdes_event(event)
        except Exception as e:
            raise ValueError(f"Error while serializing the provided event: {e}")
        request_body = {"event": event}
        if step:
            request_body.update({"step": step})
        async with self._get_client() as client:
            response = await client.post(f"/events/{handler_id}", json=request_body)
            response.raise_for_status()

            return response.json()["status"] == "sent"

    async def get_result(
        self, handler_id: str, as_handler: bool = False
    ) -> Union[RunResultT, None, HandlerDict]:
        """
        Get the result of the workflow associated with the specified handler ID.

        Args:
            handler_id (str): ID of the handler running the workflow
            as_handler (bool): Return the workflow handler. Defaults to False.

        Returns:
            Any: Result of the workflow, if available, or workflow handler (when `as_handler` is set to `True`)
        """
        async with self._get_client() as client:
            response = await client.get(f"/results/{handler_id}")
            response.raise_for_status()

            if response.status_code == 202:
                return None

            if not as_handler:
                return response.json()["result"]
            else:
                return response.json()

    async def get_handlers(self) -> list[HandlerDict]:
        """
        Get all the workflow handlers.

        Returns:
            list[HandlerDict]: List of dictionaries representing workflow handlers.
        """
        async with self._get_client() as client:
            response = await client.get("/handlers")
            response.raise_for_status()

            return response.json()["handlers"]
