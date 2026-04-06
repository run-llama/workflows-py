import os
from pathlib import Path
from typing import Literal

from llama_agents.core.config import DEFAULT_DEPLOYMENT_FILE_PATH
from llama_agents.core.deployment_config import resolve_config_parent
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BootstrapSettings(BaseSettings):
    """
    Settings configurable via env vars for controlling how an application is
    created from a git repository.
    """

    model_config = SettingsConfigDict(env_prefix="LLAMA_DEPLOY_")
    repo_url: str | None = Field(
        default=None, description="The URL of the git repository to clone"
    )
    auth_token: str | None = Field(
        default=None, description="The token to use to clone the git repository"
    )
    git_ref: str | None = Field(
        default=None, description="The git reference to checkout"
    )
    git_sha: str | None = Field(default=None, description="The git SHA to checkout")
    deployment_file_path: str = Field(
        default=".",
        description="The path to the deployment file, relative to the root of the repository",
    )
    deployment_name: str | None = Field(
        default=None, description="The name of the deployment"
    )
    bootstrap_sdists: str | None = Field(
        default=None,
        description="A directory containing tar.gz sdists to install instead of installing the appserver",
    )
    build_id: str | None = None
    build_api_host: str | None = None
    appserver_version: str | None = Field(
        default=None,
        description="The pinned appserver version to install (from LLAMA_DEPLOY_APPSERVER_VERSION)",
    )


class ApiserverSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLAMA_DEPLOY_APISERVER_")

    host: str = Field(
        default="127.0.0.1",
        description="The host where to run the API Server",
    )
    port: int = Field(
        default=4501,
        description="The TCP port where to bind the API Server",
    )

    app_root: Path = Field(
        default=Path("."),
        description="The root of the application",
    )

    deployment_file_path: Path = Field(
        default=Path(DEFAULT_DEPLOYMENT_FILE_PATH),
        description="path, relative to the repository root, where the pyproject.toml file is located",
    )

    proxy_ui: bool = Field(
        default=False,
        description="If true, proxy a development UI server instead of serving built assets",
    )
    proxy_ui_port: int = Field(
        default=4502,
        description="The TCP port where to bind the UI proxy server",
    )

    reload: bool = Field(
        default=False,
        description="If true, reload the workflow modules, for use in a dev server environment",
    )

    persistence: Literal["memory", "local", "cloud"] | None = Field(
        default=None,
        description="The persistence mode to use for the workflow server",
    )
    local_persistence_path: str | None = Field(
        default=None,
        description="The path to the sqlite database to use for the workflow server",
    )
    cloud_persistence_name: str | None = Field(
        default=None,
        description="Agent Data deployment name to use for workflow persistence. May optionally include a `:` delimited collection name, e.g. 'my_agent:my_collection'. Leave none to use the current deployment name. Recommended to override with _public if running locally, and specify a collection name",
    )

    @property
    def resolved_config_parent(self) -> Path:
        return resolve_config_parent(self.app_root, self.deployment_file_path)


settings = ApiserverSettings()


def configure_settings(
    proxy_ui: bool | None = None,
    deployment_file_path: Path | None = None,
    app_root: Path | None = None,
    reload: bool | None = None,
    persistence: Literal["memory", "local", "cloud"] | None = None,
    local_persistence_path: str | None = None,
    cloud_persistence_name: str | None = None,
    host: str | None = None,
) -> None:
    if proxy_ui is not None:
        settings.proxy_ui = proxy_ui
        os.environ["LLAMA_DEPLOY_APISERVER_PROXY_UI"] = "true" if proxy_ui else "false"
    if deployment_file_path is not None:
        settings.deployment_file_path = deployment_file_path
        os.environ["LLAMA_DEPLOY_APISERVER_DEPLOYMENT_FILE_PATH"] = str(
            deployment_file_path
        )
    if app_root is not None:
        settings.app_root = app_root
        os.environ["LLAMA_DEPLOY_APISERVER_APP_ROOT"] = str(app_root)
    if reload is not None:
        settings.reload = reload
        os.environ["LLAMA_DEPLOY_APISERVER_RELOAD"] = "true" if reload else "false"
    if persistence is not None:
        settings.persistence = persistence
        os.environ["LLAMA_DEPLOY_APISERVER_PERSISTENCE"] = persistence
    if local_persistence_path is not None:
        settings.local_persistence_path = local_persistence_path
        os.environ["LLAMA_DEPLOY_APISERVER_LOCAL_PERSISTENCE_PATH"] = (
            local_persistence_path
        )
    if cloud_persistence_name is not None:
        settings.cloud_persistence_name = cloud_persistence_name
        os.environ["LLAMA_DEPLOY_APISERVER_CLOUD_PERSISTENCE_NAME"] = (
            cloud_persistence_name
        )
    if host is not None:
        settings.host = host
        os.environ["LLAMA_DEPLOY_APISERVER_HOST"] = host
