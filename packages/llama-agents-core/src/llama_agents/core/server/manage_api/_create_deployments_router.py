import logging
from collections.abc import AsyncGenerator
from typing import Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, Response, params
from fastapi.params import Header, Query
from fastapi.responses import StreamingResponse
from llama_agents.core import schema
from typing_extensions import Annotated

from ._abstract_deployments_service import (
    AbstractDeploymentsService,
    AbstractPublicDeploymentsService,
)
from ._exceptions import DeploymentNotFoundError, ReplicaSetNotFoundError

logger = logging.getLogger(__name__)


async def get_project_id(
    project_id: Annotated[str | None, Query()] = None,
    project_id_header: Annotated[str | None, Header(alias="project-id")] = None,
) -> str:
    resolved = project_id or project_id_header
    if not resolved:
        raise HTTPException(
            status_code=422,
            detail="project_id is required (query param or project-id header)",
        )
    return resolved


def create_v1beta1_deployments_router(
    deployments_service: AbstractDeploymentsService,
    public_service: AbstractPublicDeploymentsService,
    get_project_id: Callable[..., Awaitable[str]] = get_project_id,
    dependencies: list[params.Depends] | None = None,
    public_dependencies: list[params.Depends] | None = None,
    include_in_schema: bool = True,
) -> APIRouter:
    base_router = APIRouter(prefix="/api/v1beta1", include_in_schema=include_in_schema)
    public_router = APIRouter(
        tags=["v1beta1-deployments-public"],
        dependencies=public_dependencies,
        include_in_schema=include_in_schema,
    )
    router = APIRouter(
        tags=["v1beta1-deployments"],
        dependencies=dependencies,
        include_in_schema=include_in_schema,
    )

    @public_router.get("/version")
    async def get_version() -> schema.VersionResponse:
        return await public_service.get_version()

    @router.get("/organizations")
    async def get_organizations() -> schema.OrganizationsListResponse:
        """Get all organizations"""
        return await deployments_service.get_organizations()

    @router.get("/list-projects")
    async def get_projects(
        org_id: Annotated[str | None, Query()] = None,
    ) -> schema.ProjectsListResponse:
        """Get all unique projects with their deployment counts"""
        return await deployments_service.get_projects(org_id=org_id)

    @router.post("/validate-repository")
    async def validate_repository(
        project_id: Annotated[str, Depends(get_project_id)],
        request: schema.RepositoryValidationRequest,
    ) -> schema.RepositoryValidationResponse:
        """Validate repository access and return unified response."""
        return await deployments_service.validate_repository(
            project_id=project_id,
            request=request,
        )

    @router.post("", response_model=schema.DeploymentResponse)
    async def create_deployment(
        project_id: Annotated[str, Depends(get_project_id)],
        deployment_data: schema.DeploymentCreate,
    ) -> Response:
        deployment_response = await deployments_service.create_deployment(
            project_id=project_id,
            deployment_data=deployment_data,
        )
        # Return deployment response with warning header if there are git issues

        response = Response(
            content=deployment_response.model_dump_json(),
            status_code=201,
            media_type="application/json",
        )
        return response

    @router.get("")
    async def get_deployments(
        project_id: Annotated[str, Depends(get_project_id)],
    ) -> schema.DeploymentsListResponse:
        return await deployments_service.get_deployments(project_id=project_id)

    @router.get("/{deployment_id}")
    async def get_deployment(
        project_id: Annotated[str, Depends(get_project_id)],
        deployment_id: str,
        include_events: Annotated[bool, Query()] = False,
    ) -> schema.DeploymentResponse:
        try:
            return await deployments_service.get_deployment(
                project_id=project_id,
                deployment_id=deployment_id,
                include_events=include_events,
            )
        except DeploymentNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.get("/{deployment_id}/history")
    async def get_deployment_history(
        project_id: Annotated[str, Depends(get_project_id)],
        deployment_id: str,
    ) -> schema.DeploymentHistoryResponse:
        try:
            return await deployments_service.get_deployment_history(
                project_id=project_id, deployment_id=deployment_id
            )
        except DeploymentNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.post("/{deployment_id}/rollback")
    async def rollback_deployment(
        project_id: Annotated[str, Depends(get_project_id)],
        deployment_id: str,
        request: schema.RollbackRequest,
    ) -> schema.DeploymentResponse:
        try:
            return await deployments_service.rollback_deployment(
                project_id=project_id, deployment_id=deployment_id, request=request
            )
        except DeploymentNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.delete("/{deployment_id}")
    async def delete_deployment(
        project_id: Annotated[str, Depends(get_project_id)],
        deployment_id: str,
    ) -> None:
        try:
            await deployments_service.delete_deployment(
                project_id=project_id, deployment_id=deployment_id
            )
        except DeploymentNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.put("/{deployment_id}", response_model=schema.DeploymentResponse)
    async def apply_deployment(
        project_id: Annotated[str, Depends(get_project_id)],
        deployment_id: str,
        apply_data: schema.DeploymentApply,
    ) -> Response:
        """Declarative create-or-update for a deployment by stable id.

        Returns ``201`` when a new deployment was created and ``200`` when an
        existing one was updated.
        """
        deployment_response, created = await deployments_service.apply_deployment(
            project_id=project_id,
            deployment_id=deployment_id,
            apply_data=apply_data,
        )

        return Response(
            content=deployment_response.model_dump_json(),
            status_code=201 if created else 200,
            media_type="application/json",
        )

    @router.patch("/{deployment_id}", response_model=schema.DeploymentResponse)
    async def update_deployment(
        project_id: Annotated[str, Depends(get_project_id)],
        deployment_id: str,
        update_data: schema.DeploymentUpdate,
    ) -> Response:
        """Update an existing deployment with patch-style changes

        Args:
            project_id: The project ID
            deployment_id: The deployment ID to update
            update_data: The patch-style update data
        """

        try:
            deployment_response = await deployments_service.update_deployment(
                project_id=project_id,
                deployment_id=deployment_id,
                update_data=update_data,
            )
        except DeploymentNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        response = Response(
            content=deployment_response.model_dump_json(),
            status_code=200,
            media_type="application/json",
        )
        return response

    @router.get("/{deployment_id}/logs")
    async def stream_deployment_logs(
        request: Request,
        project_id: Annotated[str, Depends(get_project_id)],
        deployment_id: str,
        include_init_containers: Annotated[bool, Query()] = False,
        since_seconds: Annotated[int | None, Query()] = None,
        tail_lines: Annotated[int | None, Query()] = None,
        follow: Annotated[bool, Query()] = True,
    ) -> StreamingResponse:
        """Stream logs for a deployment.

        Build job logs (if any) are automatically merged before application logs.
        With ``follow=true`` (default) the stream continues until the latest
        ReplicaSet changes (e.g. a new rollout occurs). With ``follow=false``
        the server returns whatever logs are currently available and ends the
        SSE stream — useful for clients that want a bounded, "fetch and exit"
        response.
        """

        try:
            inner = deployments_service.stream_deployment_logs(
                project_id=project_id,
                deployment_id=deployment_id,
                include_init_containers=include_init_containers,
                since_seconds=since_seconds,
                tail_lines=tail_lines,
                follow=follow,
            )

            async def sse_lines() -> AsyncGenerator[str, None]:
                async for data in inner:
                    yield "event: log\n"
                    yield f"data: {data.model_dump_json()}\n\n"

            return StreamingResponse(
                sse_lines(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        except DeploymentNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ReplicaSetNotFoundError as e:
            # Deployment exists but hasn't created a ReplicaSet yet
            raise HTTPException(status_code=409, detail=str(e))

    @router.get("/{deployment_id}/git/{git_path:path}")
    @router.post("/{deployment_id}/git/{git_path:path}")
    async def git_request(
        request: Request,
        project_id: Annotated[str, Depends(get_project_id)],
        deployment_id: str,
        git_path: str,
    ) -> Response:
        """Handle git HTTP requests (info/refs, upload-pack, receive-pack)."""
        try:
            return await deployments_service.handle_git_request(
                request=request,
                project_id=project_id,
                deployment_id=deployment_id,
                git_path=git_path,
            )
        except DeploymentNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    base_router.include_router(public_router, prefix="/deployments-public")
    base_router.include_router(router, prefix="/deployments")
    return base_router
