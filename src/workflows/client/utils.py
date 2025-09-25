from pydantic import BaseModel, Field
from typing import TypedDict, Any, NotRequired


class AuthDetails(BaseModel):
    token: str = Field(description="Authentication token")
    prefix: str = Field(
        description="Prefix in the authentication header (defaults to `Bearer`)",
        default="Bearer",
    )
    auth_header_name: str = Field(
        description="Authentication header name (defaults to `Authentication`)",
        default="Authentication",
    )


class EventDict(TypedDict):
    __is_pydantic: NotRequired[bool]
    qualified_name: str
    value: dict[str, Any]
