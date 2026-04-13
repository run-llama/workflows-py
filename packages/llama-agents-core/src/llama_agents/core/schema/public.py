from typing import Literal

from .base import Base

Capability = Literal["code_push"] | str


class Capabilities:
    """Known capability identifiers advertised by the server."""

    CODE_PUSH: Capability = "code_push"
    ORGANIZATIONS: Capability = "organizations"


class VersionResponse(Base):
    version: str
    requires_auth: bool = False
    min_llamactl_version: str | None = None
    capabilities: list[Capability] = []
