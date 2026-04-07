from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class Auth:
    """Auth Profile configuration"""

    id: str
    name: str
    api_url: str
    project_id: str
    api_key: str | None = None
    # reference to the API key if we created it from device oauth, to be cleaned up
    # once de-authenticated
    api_key_id: str | None = None
    device_oidc: DeviceOIDC | None = None


class DeviceOIDC(BaseModel):
    """Device OIDC configuration"""

    # A name for this device, derived from the host. Used in API key name.
    device_name: str
    # A unique user ID to identify the user in the API. Prevents duplicate logins.
    user_id: str
    # email of the user
    email: str
    # OIDC client ID
    client_id: str
    # OIDC discovery URL
    discovery_url: str
    # usually 5m long JWT. For calling APIs.
    device_access_token: str
    # usually opaque, used to get new access tokens
    device_refresh_token: str | None = None
    # usually 1h long JWT. Contains user info (email, name, etc.)
    device_id_token: str | None = None


@dataclass
class Environment:
    """Environment configuration stored in SQLite.

    Note: `api_url`, `requires_auth`, and `min_llamactl_version` are persisted
    in the environments table.
    """

    api_url: str
    requires_auth: bool
    min_llamactl_version: str | None = None
    capabilities: list[str] = field(default_factory=list)


DEFAULT_ENVIRONMENT = Environment(
    api_url="https://api.cloud.llamaindex.ai",
    requires_auth=True,
    min_llamactl_version=None,
)
