from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

from rich import print as rprint

if TYPE_CHECKING:
    from llama_agents.core.client.manage_client import ControlPlaneClient, ProjectClient


def get_control_plane_client() -> ControlPlaneClient:
    from llama_agents.cli.config.env_service import service
    from llama_agents.core.client.manage_client import ControlPlaneClient

    auth_svc = service.current_auth_service()
    profile = service.current_auth_service().get_current_profile()
    if profile:
        resolved_base_url = profile.api_url.rstrip("/")
        resolved_api_key = profile.api_key
        return ControlPlaneClient(
            resolved_base_url, resolved_api_key, auth_svc.auth_middleware()
        )

    # Fallback: allow env-scoped client construction for env operations
    env = service.get_current_environment()
    resolved_base_url = env.api_url.rstrip("/")
    return ControlPlaneClient(resolved_base_url)


def get_project_client(project_id_override: str | None = None) -> ProjectClient:
    """Return a ProjectClient bound to the active profile.

    If ``project_id_override`` is provided, the client uses that project ID
    instead of the profile's default. Auth (API key + middleware) and base URL
    still come from the active profile — this mirrors ``kubectl -n <ns>``.
    """
    from llama_agents.cli.config.env_service import service
    from llama_agents.core.client.manage_client import ProjectClient

    auth_svc = service.current_auth_service()
    profile = auth_svc.get_current_profile()
    if not profile:
        rprint("\n[bold red]No profile configured![/bold red]")
        rprint("\nTo get started, create a profile with:")
        if auth_svc.env.requires_auth:
            rprint("[cyan]llamactl auth login[/cyan]")
        else:
            rprint("[cyan]llamactl auth token[/cyan]")
        raise SystemExit(1)
    project_id = project_id_override or profile.project_id
    return ProjectClient(
        profile.api_url, project_id, profile.api_key, auth_svc.auth_middleware()
    )


@asynccontextmanager
async def project_client_context(
    project_id_override: str | None = None,
) -> AsyncGenerator[ProjectClient, None]:
    client = get_project_client(project_id_override=project_id_override)
    try:
        yield client
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
