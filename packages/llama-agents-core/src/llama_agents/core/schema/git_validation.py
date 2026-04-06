from pathlib import Path

from pydantic import BaseModel, Field


class RepositoryValidationResponse(BaseModel):
    """
    Unified response for repository validation that works for any git repository.
    This is the primary schema that should be used for the /validate-repository endpoint.
    """

    accessible: bool = Field(
        ...,
        description="Whether the repository can be accessed by any means available to the server",
    )
    message: str = Field(..., description="Human-readable string explaining the status")
    pat_is_obsolete: bool = Field(
        default=False,
        description="True if validation succeeded via GitHub App for a deployment that previously used a PAT",
    )
    github_app_name: str | None = Field(
        default=None,
        description="Name of the GitHub App if repository is a private GitHub repo and server has GitHub App configured",
    )
    github_app_installation_url: str | None = Field(
        default=None,
        description="GitHub App installation/authorization URL for connecting the app to a repository owner",
    )
    github_app_settings_url: str | None = Field(
        default=None,
        description="GitHub App installation settings URL for managing repository access on an existing installation",
    )
    github_app_authorization_url: str | None = Field(
        default=None,
        description="Browser-openable URL that triggers GitHub OAuth for CLI clients needing to connect their GitHub account",
    )


class RepositoryValidationRequest(BaseModel):
    repository_url: str
    deployment_id: str | None = None
    pat: str | None = None


class GitApplicationValidationResponse(BaseModel):
    """
    After general repository validation, a model that describes further validation of configuration, such as the
    git reference, it's resolved SHA (if resolveable), and whether the deployment file is valid.
    """

    is_valid: bool
    error_message: str | None = None
    git_ref: str | None = None
    git_sha: str | None = None
    valid_deployment_file_path: str | None = None
    ui_build_output_path: Path | None = Field(
        default=None,
        description="Path to the UI build output, if the deployment's UI has a package.json with a build script; None if no UI is configured",
    )
