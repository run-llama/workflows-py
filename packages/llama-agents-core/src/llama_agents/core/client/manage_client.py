from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncIterator, Callable, List

import httpx
from httpx._types import PrimitiveData
from llama_agents.core.client.ssl_util import get_httpx_verify_param
from llama_agents.core.schema import LogEvent
from llama_agents.core.schema.backups import (
    BackupListResponse,
    BackupResponse,
    RestoreRequest,
    RestoreResponse,
)
from llama_agents.core.schema.deployments import (
    DeploymentCreate,
    DeploymentHistoryResponse,
    DeploymentResponse,
    DeploymentsListResponse,
    DeploymentUpdate,
    RollbackRequest,
)
from llama_agents.core.schema.git_validation import (
    RepositoryValidationRequest,
    RepositoryValidationResponse,
)
from llama_agents.core.schema.projects import (
    OrgsListResponse,
    OrgSummary,
    ProjectsListResponse,
    ProjectSummary,
)
from llama_agents.core.schema.public import VersionResponse


class BaseClient:
    def __init__(
        self, base_url: str, api_key: str | None = None, auth: httpx.Auth | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        verify = get_httpx_verify_param()
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            auth=auth,
            verify=verify,
        )
        self.hookless_client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, auth=auth, verify=verify
        )

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.hookless_client.aclose()


class ControlPlaneClient(BaseClient):
    """Unscoped client for non-project endpoints."""

    @classmethod
    @asynccontextmanager
    async def ctx(
        cls, base_url: str, api_key: str | None = None, auth: httpx.Auth | None = None
    ) -> AsyncIterator[ControlPlaneClient]:
        client = cls(base_url, api_key, auth)
        try:
            yield client
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    def __init__(
        self, base_url: str, api_key: str | None = None, auth: httpx.Auth | None = None
    ) -> None:
        super().__init__(base_url, api_key, auth)

    async def server_version(self) -> VersionResponse:
        response = await self.client.get("/api/v1beta1/deployments-public/version")
        _raise_for_status(response)
        return VersionResponse.model_validate(response.json())

    async def create_backup(self) -> BackupResponse:
        response = await self.client.post("/api/v1beta1/backups")
        _raise_for_status(response)
        return BackupResponse.model_validate(response.json())

    async def list_backups(self) -> BackupListResponse:
        response = await self.client.get("/api/v1beta1/backups")
        _raise_for_status(response)
        return BackupListResponse.model_validate(response.json())

    async def get_backup(self, backup_id: str) -> BackupResponse:
        response = await self.client.get(f"/api/v1beta1/backups/{backup_id}")
        _raise_for_status(response)
        return BackupResponse.model_validate(response.json())

    async def delete_backup(self, backup_id: str) -> BackupResponse:
        response = await self.client.delete(f"/api/v1beta1/backups/{backup_id}")
        _raise_for_status(response)
        return BackupResponse.model_validate(response.json())

    async def restore_backup(
        self, backup_id: str, request: RestoreRequest | None = None
    ) -> RestoreResponse:
        if request is None:
            request = RestoreRequest(backup_id=backup_id)
        response = await self.client.post(
            "/api/v1beta1/backups/restore",
            json=request.model_dump(),
        )
        _raise_for_status(response)
        return RestoreResponse.model_validate(response.json())

    async def list_orgs(self) -> List[OrgSummary]:
        response = await self.client.get("/api/v1beta1/deployments/list-orgs")
        _raise_for_status(response)
        orgs_response = OrgsListResponse.model_validate(response.json())
        return list(orgs_response.orgs)

    async def list_projects(self, org_id: str | None = None) -> List[ProjectSummary]:
        params = {}
        if org_id is not None:
            params["org_id"] = org_id
        response = await self.client.get(
            "/api/v1beta1/deployments/list-projects", params=params
        )
        _raise_for_status(response)
        projects_response = ProjectsListResponse.model_validate(response.json())
        return list(projects_response.projects)


def _raise_for_status(response: httpx.Response) -> None:
    """
    Custom raise for status that adds response body information to the error message, but still uses the httpx
    error classes
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = _response_body_snippet(response, limit=250)
        request_id = response.headers.get("x-request-id") or response.headers.get(
            "x-correlation-id"
        )
        rid = f" [request id: {request_id}]" if request_id else ""
        body_part = f" - {body}" if body else ""
        raise httpx.HTTPStatusError(
            f"HTTP {response.status_code} for url {response.url}{body_part}{rid}",
            request=e.request or response.request,
            response=e.response or response,
        )


def _response_body_snippet(response: httpx.Response, limit: int = 500) -> str:
    try:
        text = response.text
        if not text:
            # fallback attempt if body not read
            try:
                data = response.json()
            except Exception:
                data = None
            if data is not None:
                text = str(data)
        text = (text or "").strip()
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text
    except Exception:
        return ""


class ProjectClient(BaseClient):
    """Project-scoped client for deployment operations."""

    @classmethod
    @asynccontextmanager
    async def ctx(
        cls,
        base_url: str,
        project_id: str,
        api_key: str | None = None,
        auth: httpx.Auth | None = None,
    ) -> AsyncIterator[ProjectClient]:
        client = cls(base_url, project_id, api_key, auth)
        try:
            yield client
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    def __init__(
        self,
        base_url: str,
        project_id: str,
        api_key: str | None = None,
        auth: httpx.Auth | None = None,
    ) -> None:
        super().__init__(base_url, api_key, auth)
        self.project_id = project_id

    async def list_deployments(self) -> List[DeploymentResponse]:
        response = await self.client.get(
            "/api/v1beta1/deployments",
            params={"project_id": self.project_id},
        )
        _raise_for_status(response)
        deployments_response = DeploymentsListResponse.model_validate(response.json())
        return [deployment for deployment in deployments_response.deployments]

    async def get_deployment(
        self, deployment_id: str, include_events: bool = False
    ) -> DeploymentResponse:
        response = await self.client.get(
            f"/api/v1beta1/deployments/{deployment_id}",
            params={"project_id": self.project_id, "include_events": include_events},
        )
        _raise_for_status(response)
        return DeploymentResponse.model_validate(response.json())

    async def create_deployment(
        self, deployment_data: DeploymentCreate
    ) -> DeploymentResponse:
        response = await self.client.post(
            "/api/v1beta1/deployments",
            params={"project_id": self.project_id},
            json=deployment_data.model_dump(exclude_none=True),
        )
        _raise_for_status(response)
        return DeploymentResponse.model_validate(response.json())

    async def delete_deployment(self, deployment_id: str) -> None:
        response = await self.client.delete(
            f"/api/v1beta1/deployments/{deployment_id}",
            params={"project_id": self.project_id},
        )
        _raise_for_status(response)

    async def update_deployment(
        self,
        deployment_id: str,
        update_data: DeploymentUpdate,
    ) -> DeploymentResponse:
        response = await self.client.patch(
            f"/api/v1beta1/deployments/{deployment_id}",
            params={"project_id": self.project_id},
            json=update_data.model_dump(),
        )
        _raise_for_status(response)
        return DeploymentResponse.model_validate(response.json())

    async def get_deployment_history(
        self, deployment_id: str
    ) -> DeploymentHistoryResponse:
        response = await self.client.get(
            f"/api/v1beta1/deployments/{deployment_id}/history",
            params={"project_id": self.project_id},
        )
        _raise_for_status(response)
        return DeploymentHistoryResponse.model_validate(response.json())

    async def rollback_deployment(
        self, deployment_id: str, git_sha: str
    ) -> DeploymentResponse:
        response = await self.client.post(
            f"/api/v1beta1/deployments/{deployment_id}/rollback",
            params={"project_id": self.project_id},
            json=RollbackRequest(git_sha=git_sha).model_dump(),
        )
        _raise_for_status(response)
        return DeploymentResponse.model_validate(response.json())

    async def validate_repository(
        self,
        repo_url: str,
        deployment_id: str | None = None,
        pat: str | None = None,
    ) -> RepositoryValidationResponse:
        response = await self.client.post(
            "/api/v1beta1/deployments/validate-repository",
            params={"project_id": self.project_id},
            json=RepositoryValidationRequest(
                repository_url=repo_url,
                deployment_id=deployment_id,
                pat=pat,
            ).model_dump(),
        )
        _raise_for_status(response)
        return RepositoryValidationResponse.model_validate(response.json())

    async def stream_deployment_logs(
        self,
        deployment_id: str,
        *,
        include_init_containers: bool = False,
        since_seconds: int | None = None,
        tail_lines: int | None = None,
    ) -> AsyncGenerator[LogEvent, None]:
        """Stream logs as LogEvent items from the control plane using SSE.

        Yields `LogEvent` models until the stream ends (e.g., rollout completes).
        """
        params_dict: dict[str, PrimitiveData] = {
            "project_id": self.project_id,
            "include_init_containers": include_init_containers,
        }
        if since_seconds is not None:
            params_dict["since_seconds"] = since_seconds
        if tail_lines is not None:
            params_dict["tail_lines"] = tail_lines

        url = f"/api/v1beta1/deployments/{deployment_id}/logs"
        headers = {"Accept": "text/event-stream"}

        async with self.hookless_client.stream(
            "GET",
            url,
            params=httpx.QueryParams(params_dict),
            headers=headers,
            timeout=None,
        ) as response:
            _raise_for_status(response)

            event_name: str | None = None
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line is None:
                    continue
                line = line.decode() if isinstance(line, (bytes, bytearray)) else line
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].lstrip())
                elif line.strip() == "":
                    if event_name == "log" and data_lines:
                        data_str = "\n".join(data_lines)
                        try:
                            yield LogEvent.model_validate_json(data_str)
                        except Exception:
                            pass
                    event_name = None
                    data_lines = []


Closer = Callable[[], None]
