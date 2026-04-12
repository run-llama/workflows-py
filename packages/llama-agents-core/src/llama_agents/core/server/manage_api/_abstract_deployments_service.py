from abc import ABC, abstractmethod
from typing import AsyncGenerator, cast

from fastapi import Request, Response
from llama_agents.core import schema
from llama_agents.core.schema import LogEvent
from llama_agents.core.schema.deployments import (
    DeploymentHistoryResponse,
    DeploymentResponse,
    RollbackRequest,
)


class AbstractPublicDeploymentsService(ABC):
    @abstractmethod
    async def get_version(self) -> schema.VersionResponse:
        """
        Get the version of the server
        """
        ...


class AbstractDeploymentsService(ABC):
    @abstractmethod
    async def get_organizations(self) -> schema.OrganizationsListResponse:
        """
        Get a list of organizations.
        """
        ...

    @abstractmethod
    async def get_projects(
        self, org_id: str | None = None
    ) -> schema.ProjectsListResponse:
        """
        Get a list of projects, optionally filtered by org_id.
        """
        ...

    @abstractmethod
    async def validate_repository(
        self,
        project_id: str,
        request: schema.RepositoryValidationRequest,
    ) -> schema.RepositoryValidationResponse:
        """
        Validate repository access and return unified response.
        """
        ...

    @abstractmethod
    async def create_deployment(
        self,
        project_id: str,
        deployment_data: schema.DeploymentCreate,
    ) -> DeploymentResponse:
        """
        Create a new deployment

        Args:
            project_id: The ID of the project to create the deployment in
            deployment_data: The data for the deployment

        Returns:
            The created deployment
        Raises:
            DeploymentNotFoundError: If the deployment ID is not found
        """
        ...

    @abstractmethod
    async def get_deployments(
        self,
        project_id: str,
    ) -> schema.DeploymentsListResponse:
        """
        Get a list of deployments for a project

        Args:
            project_id: The ID of the project to get the deployments for

        Returns:
            A list of deployments
        """
        ...

    @abstractmethod
    async def get_deployment(
        self,
        project_id: str,
        deployment_id: str,
        include_events: bool = False,
    ) -> schema.DeploymentResponse:
        """
        Get a deployment by ID

        Args:
            project_id: The ID of the project to get the deployment for
            deployment_id: The ID of the deployment to get
            include_events: Whether to include events in the response

        Returns:
            The deployment
        Raises:
            DeploymentNotFoundError: If the deployment ID is not found
        """
        ...

    @abstractmethod
    async def delete_deployment(
        self,
        project_id: str,
        deployment_id: str,
    ) -> None:
        """
        Delete a deployment

        Args:
            project_id: The ID of the project to delete the deployment from
            deployment_id: The ID of the deployment to delete

        Returns:
            None
        Raises:
            DeploymentNotFoundError: If the deployment ID is not found
        """
        ...

    @abstractmethod
    async def update_deployment(
        self,
        project_id: str,
        deployment_id: str,
        update_data: schema.DeploymentUpdate,
    ) -> DeploymentResponse:
        """
        Update a deployment

        Args:
            project_id: The ID of the project to update the deployment in
            deployment_id: The ID of the deployment to update
            update_data: The data to update the deployment with

        Returns:
            The updated deployment
        Raises:
            DeploymentNotFoundError: If the deployment ID is not found
        """
        ...

    @abstractmethod
    async def get_deployment_history(
        self, project_id: str, deployment_id: str
    ) -> DeploymentHistoryResponse:
        """
        Get the release history for a deployment.
        """
        ...

    @abstractmethod
    async def rollback_deployment(
        self, project_id: str, deployment_id: str, request: RollbackRequest
    ) -> DeploymentResponse:
        """
        Roll back a deployment to a previous git sha.
        """
        ...

    @abstractmethod
    async def stream_deployment_logs(
        self,
        project_id: str,
        deployment_id: str,
        include_init_containers: bool = False,
        since_seconds: int | None = None,
        tail_lines: int | None = None,
    ) -> AsyncGenerator[LogEvent, None]:
        """
        Stream the logs for a deployment.

        Build job logs (if any) are automatically merged into the stream
        before application logs.

        Args:
            project_id: The ID of the project to stream the logs for
            deployment_id: The ID of the deployment to stream the logs for
            include_init_containers: Whether to include init containers in the logs
            since_seconds: The number of seconds to stream the logs for
            tail_lines: The number of lines to stream the logs for

        Returns:
            A generator of log events
        Raises:
            DeploymentNotFoundError: If the deployment ID is not found
        """
        # This method is abstract. The following unreachable code ensures type
        # checkers treat it as an async generator, so call sites can `async for`.
        raise NotImplementedError
        if False:
            yield cast(LogEvent, None)

    @abstractmethod
    async def handle_git_request(
        self,
        request: Request,
        project_id: str,
        deployment_id: str,
        git_path: str,
    ) -> Response:
        """Handle a git HTTP request (info/refs, upload-pack, receive-pack).

        Args:
            request: The incoming HTTP request
            project_id: The project the deployment belongs to
            deployment_id: The deployment to serve git for
            git_path: The git sub-path (e.g. info/refs, git-upload-pack)

        Returns:
            The HTTP response from the git backend
        """
        raise NotImplementedError
