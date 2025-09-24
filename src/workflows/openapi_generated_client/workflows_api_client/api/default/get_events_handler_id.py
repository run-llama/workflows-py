from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_events_handler_id_response_200 import GetEventsHandlerIdResponse200
from ...types import UNSET, Response, Unset


def _get_kwargs(
    handler_id: str,
    *,
    sse: Union[Unset, bool] = True,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["sse"] = sse

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/events/{handler_id}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, GetEventsHandlerIdResponse200]]:
    if response.status_code == 200:
        response_200 = GetEventsHandlerIdResponse200.from_dict(response.text)

        return response_200

    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, GetEventsHandlerIdResponse200]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    sse: Union[Unset, bool] = True,
) -> Response[Union[Any, GetEventsHandlerIdResponse200]]:
    r"""Stream workflow events

     Streams events produced by a workflow execution. Events are emitted as
    newline-delimited JSON by default, or as Server-Sent Events when `sse=true`.
    Event data is formatted according to llama-index's json serializer. For
    pydantic serializable python types, it returns:
    {
      \"__is_pydantic\": True,
      \"value\": <pydantic serialized value>,
      \"qualified_name\": <python path to pydantic class>
    }

    Args:
        handler_id (str):
        sse (Union[Unset, bool]):  Default: True.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, GetEventsHandlerIdResponse200]]
    """

    kwargs = _get_kwargs(
        handler_id=handler_id,
        sse=sse,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    sse: Union[Unset, bool] = True,
) -> Optional[Union[Any, GetEventsHandlerIdResponse200]]:
    r"""Stream workflow events

     Streams events produced by a workflow execution. Events are emitted as
    newline-delimited JSON by default, or as Server-Sent Events when `sse=true`.
    Event data is formatted according to llama-index's json serializer. For
    pydantic serializable python types, it returns:
    {
      \"__is_pydantic\": True,
      \"value\": <pydantic serialized value>,
      \"qualified_name\": <python path to pydantic class>
    }

    Args:
        handler_id (str):
        sse (Union[Unset, bool]):  Default: True.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, GetEventsHandlerIdResponse200]
    """

    return sync_detailed(
        handler_id=handler_id,
        client=client,
        sse=sse,
    ).parsed


async def asyncio_detailed(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    sse: Union[Unset, bool] = True,
) -> Response[Union[Any, GetEventsHandlerIdResponse200]]:
    r"""Stream workflow events

     Streams events produced by a workflow execution. Events are emitted as
    newline-delimited JSON by default, or as Server-Sent Events when `sse=true`.
    Event data is formatted according to llama-index's json serializer. For
    pydantic serializable python types, it returns:
    {
      \"__is_pydantic\": True,
      \"value\": <pydantic serialized value>,
      \"qualified_name\": <python path to pydantic class>
    }

    Args:
        handler_id (str):
        sse (Union[Unset, bool]):  Default: True.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, GetEventsHandlerIdResponse200]]
    """

    kwargs = _get_kwargs(
        handler_id=handler_id,
        sse=sse,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    sse: Union[Unset, bool] = True,
) -> Optional[Union[Any, GetEventsHandlerIdResponse200]]:
    r"""Stream workflow events

     Streams events produced by a workflow execution. Events are emitted as
    newline-delimited JSON by default, or as Server-Sent Events when `sse=true`.
    Event data is formatted according to llama-index's json serializer. For
    pydantic serializable python types, it returns:
    {
      \"__is_pydantic\": True,
      \"value\": <pydantic serialized value>,
      \"qualified_name\": <python path to pydantic class>
    }

    Args:
        handler_id (str):
        sse (Union[Unset, bool]):  Default: True.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, GetEventsHandlerIdResponse200]
    """

    return (
        await asyncio_detailed(
            handler_id=handler_id,
            client=client,
            sse=sse,
        )
    ).parsed
