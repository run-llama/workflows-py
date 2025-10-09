# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import secrets
import string

from pydantic import BaseModel
from typing import Union, Any
from workflows.context.serializers import JsonSerializer

alphabet = string.ascii_letters + string.digits  # A-Z, a-z, 0-9


def nanoid(size: int = 10) -> str:
    """Returns a unique identifier with the format 'kY2xP9hTnQ'."""
    return "".join(secrets.choice(alphabet) for _ in range(size))


def serdes_event(
    event: Union[dict[str, Any], BaseModel, str], serialize: bool = True
) -> Any:
    """
    Serialize or deserialize a start event.

    Args:
        event (Union[dict[str, Any], BaseModel, str]): Input event
        serialize (bool): Serialize if true, deserialize if false.
    """
    serializer = JsonSerializer()
    if serialize:
        if isinstance(event, (BaseModel, dict)):
            event = serializer.serialize(event)
        else:
            event = serializer.serialize_value(event)
        return event
    else:
        if isinstance(event, str):
            event = serializer.deserialize(event)
        else:
            event = serializer.deserialize_value(event)
        return event
