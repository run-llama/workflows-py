from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.post_events_handler_id_body import PostEventsHandlerIdBody
from ...models.post_events_handler_id_response_200 import PostEventsHandlerIdResponse200
from ...types import Response


def _get_kwargs(
    handler_id: str,
    *,
    body: PostEventsHandlerIdBody,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/events/{handler_id}",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, PostEventsHandlerIdResponse200]]:
    if response.status_code == 200:
        response_200 = PostEventsHandlerIdResponse200.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = cast(Any, None)
        return response_400

    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404

    if response.status_code == 409:
        response_409 = cast(Any, None)
        return response_409

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, PostEventsHandlerIdResponse200]]:
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
    body: PostEventsHandlerIdBody,
) -> Response[Union[Any, PostEventsHandlerIdResponse200]]:
    """Send event to workflow

     Sends an event to a running workflow's context.

    Args:
        handler_id (str):
        body (PostEventsHandlerIdBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, PostEventsHandlerIdResponse200]]
    """

    kwargs = _get_kwargs(
        handler_id=handler_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PostEventsHandlerIdBody,
) -> Optional[Union[Any, PostEventsHandlerIdResponse200]]:
    """Send event to workflow

     Sends an event to a running workflow's context.

    Args:
        handler_id (str):
        body (PostEventsHandlerIdBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, PostEventsHandlerIdResponse200]
    """

    return sync_detailed(
        handler_id=handler_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PostEventsHandlerIdBody,
) -> Response[Union[Any, PostEventsHandlerIdResponse200]]:
    """Send event to workflow

     Sends an event to a running workflow's context.

    Args:
        handler_id (str):
        body (PostEventsHandlerIdBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, PostEventsHandlerIdResponse200]]
    """

    kwargs = _get_kwargs(
        handler_id=handler_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PostEventsHandlerIdBody,
) -> Optional[Union[Any, PostEventsHandlerIdResponse200]]:
    """Send event to workflow

     Sends an event to a running workflow's context.

    Args:
        handler_id (str):
        body (PostEventsHandlerIdBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, PostEventsHandlerIdResponse200]
    """

    return (
        await asyncio_detailed(
            handler_id=handler_id,
            client=client,
            body=body,
        )
    ).parsed
