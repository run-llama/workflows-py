"""Deploy LlamaIndex Workflows to AWS Bedrock AgentCore Runtime.

Provides a high-level ``AgentCoreDeployer`` that handles the full lifecycle:
build a container via CodeBuild, push to ECR, create/update an AgentCore
Runtime, invoke it, and tear it down.

Example::

    import boto3
    from llama_agents.agentcore.deploy import AgentCoreDeployer

    deployer = AgentCoreDeployer(
        session=boto3.Session(region_name="us-east-1"),
        deployment_role="arn:aws:iam::123456789012:role/AgentCoreDeployRole",
        execution_role="arn:aws:iam::123456789012:role/AgentCoreExecutionRole",
    )

    # Build, push, and deploy
    runtime = deployer.deploy(project_dir=".")

    # Invoke
    result = deployer.invoke(runtime.arn, {"input": "Hello!"})

    # Clean up
    deployer.destroy(runtime.name)
"""

from __future__ import annotations

import json
import logging
import random
import re
import string
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError as e:
    raise ImportError(
        "boto3 is required for deployment. Install with: "
        "pip install 'llama-agents-agentcore[deploy]'"
    ) from e


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class DeployedRuntime:
    """Metadata for a deployed AgentCore Runtime."""

    name: str
    arn: str
    runtime_id: str
    ecr_image: str
    region: str
    account_id: str
    repository_name: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a dict for JSON persistence."""
        return {
            "name": self.name,
            "arn": self.arn,
            "runtime_id": self.runtime_id,
            "ecr_image": self.ecr_image,
            "region": self.region,
            "account_id": self.account_id,
            "repository_name": self.repository_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> DeployedRuntime:
        """Deserialize from a dict."""
        return cls(**data)


@dataclass
class DeployConfig:
    """Configuration for a deployment."""

    runtime_name: str | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    max_lifetime: int = 3600
    build_compute_type: str = "BUILD_GENERAL1_SMALL"
    source_files: list[str] | None = None


# ── Deployer ──────────────────────────────────────────────────────────────


class AgentCoreDeployer:
    """High-level deployer for LlamaIndex Workflows on AWS Bedrock AgentCore.

    Handles the full lifecycle: container build (CodeBuild), image push (ECR),
    runtime create/update, invocation, and teardown.

    Args:
        session: A configured ``boto3.Session``.
        deployment_role: IAM role ARN for CodeBuild to build/push images.
        execution_role: IAM role ARN for the AgentCore Runtime at runtime.
    """

    def __init__(
        self,
        session: boto3.Session,
        deployment_role: str,
        execution_role: str,
    ) -> None:
        self._session = session
        self._deployment_role = deployment_role
        self._execution_role = execution_role
        self._region = session.region_name or "us-east-1"
        self._account_id: str | None = None

    @property
    def account_id(self) -> str:
        """AWS account ID (resolved lazily)."""
        if self._account_id is None:
            self._account_id = str(
                self._session.client("sts").get_caller_identity()["Account"]
            )
        return self._account_id

    # ── Public API ────────────────────────────────────────────────────────

    def deploy(
        self,
        project_dir: str | Path = ".",
        config: DeployConfig | None = None,
    ) -> DeployedRuntime:
        """Build, push, and deploy a workflow project to AgentCore.

        Args:
            project_dir: Path to the project root (must contain pyproject.toml).
            config: Optional deployment configuration overrides.

        Returns:
            A ``DeployedRuntime`` with all metadata needed for invoke/destroy.
        """
        project_dir = Path(project_dir).resolve()
        config = config or DeployConfig()

        if not (project_dir / "pyproject.toml").exists():
            raise FileNotFoundError(
                f"No pyproject.toml found in {project_dir}. "
                "This file is required for workflow discovery."
            )

        # Derive names
        project_name = config.runtime_name or _project_name_from_pyproject(project_dir)
        safe_name = _sanitize_name(project_name)
        repository_name = f"bedrock-agentcore-{safe_name}"
        runtime_name = f"{safe_name}_runtime"

        logger.info("Deploying '%s' to AgentCore in %s", project_name, self._region)

        # Step 1: Build and push container image
        s3_bucket = f"llamactl-agentcore-{self.account_id}-{self._region}"
        self._ensure_s3_bucket(s3_bucket)

        ecr_image = self._build_and_push(
            project_dir=project_dir,
            repository_name=repository_name,
            s3_bucket=s3_bucket,
            source_files=config.source_files,
            build_compute_type=config.build_compute_type,
        )

        # Step 2: Merge env vars from DeploymentConfig (pyproject.toml) with
        # explicit overrides from DeployConfig so they're set at container level.
        deployment_env = _parse_deployment_env_vars(project_dir)
        env_vars = {
            "AWS_DEFAULT_REGION": self._region,
            **deployment_env,
            **config.env_vars,
        }
        runtime_id, runtime_arn = self._deploy_runtime(
            runtime_name=runtime_name,
            ecr_uri=ecr_image,
            env_vars=env_vars,
            max_lifetime=config.max_lifetime,
        )

        result = DeployedRuntime(
            name=runtime_name,
            arn=runtime_arn,
            runtime_id=runtime_id,
            ecr_image=ecr_image,
            region=self._region,
            account_id=self.account_id,
            repository_name=repository_name,
        )

        logger.info("Deployment complete: %s", runtime_arn)
        return result

    def invoke(
        self,
        runtime_arn: str,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke a deployed AgentCore Runtime.

        Args:
            runtime_arn: The Runtime ARN from ``deploy()``.
            payload: JSON-serializable payload. Typically includes
                ``{"input": "..."}`` and optionally ``{"workflow": "name"}``.
            session_id: Optional session ID for continuing a previous session.
                If not provided, a new random session ID is generated.

        Returns:
            The workflow result as a dict.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        client = self._session.client("bedrock-agentcore")

        response = client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            qualifier="DEFAULT",
            runtimeSessionId=session_id,
            payload=json.dumps(payload).encode(),
        )

        chunks: list[str] = []
        for chunk in response.get("response", []):
            chunks.append(chunk.decode("utf-8"))

        return json.loads("".join(chunks))

    def destroy(
        self,
        runtime_name: str,
        repository_name: str | None = None,
    ) -> None:
        """Delete an AgentCore Runtime and optionally its ECR repository.

        Args:
            runtime_name: The runtime name from ``deploy()``.
            repository_name: ECR repository to delete. If None, only the
                runtime is deleted.
        """
        control = self._session.client("bedrock-agentcore-control")

        runtime = self._find_runtime(control, runtime_name)
        if runtime:
            logger.info("Deleting runtime: %s", runtime_name)
            control.delete_agent_runtime(agentRuntimeId=runtime["agentRuntimeId"])
            self._wait_for_deletion(control, runtime["agentRuntimeId"])
        else:
            logger.info("Runtime '%s' not found (may already be deleted)", runtime_name)

        if repository_name:
            ecr = self._session.client("ecr")
            try:
                ecr.delete_repository(repositoryName=repository_name, force=True)
                logger.info("Deleted ECR repository: %s", repository_name)
            except ecr.exceptions.RepositoryNotFoundException:
                logger.info("ECR repository '%s' already deleted", repository_name)

    def destroy_from_metadata(self, runtime: DeployedRuntime) -> None:
        """Convenience: destroy using a ``DeployedRuntime`` object."""
        self.destroy(runtime.name, runtime.repository_name)

    # ── Container Build (CodeBuild + ECR) ─────────────────────────────────

    def _build_and_push(
        self,
        project_dir: Path,
        repository_name: str,
        s3_bucket: str,
        source_files: list[str] | None = None,
        build_compute_type: str = "BUILD_GENERAL1_SMALL",
    ) -> str:
        """Build an ARM64 container via CodeBuild and push to ECR.

        Returns the ECR image URI.
        """
        ecr = self._session.client("ecr")
        ecr_registry = f"{self.account_id}.dkr.ecr.{self._region}.amazonaws.com"
        ecr_image = f"{ecr_registry}/{repository_name}:latest"

        # Ensure ECR repo
        try:
            ecr.create_repository(repositoryName=repository_name)
            logger.info("Created ECR repository: %s", repository_name)
        except ecr.exceptions.RepositoryAlreadyExistsException:
            logger.info("Using existing ECR repository: %s", repository_name)

        # Create source zip
        suffix = "".join(random.choices(string.ascii_letters, k=16))
        s3_key = f"codebuild-{suffix}.zip"

        with tempfile.TemporaryFile() as tmp:
            self._create_source_zip(
                tmp,
                project_dir,
                ecr_image,
                repository_name,
                ecr_registry,
                source_files,
            )
            tmp.seek(0)
            self._session.client("s3").upload_fileobj(tmp, s3_bucket, s3_key)

        logger.info("Uploaded source to s3://%s/%s", s3_bucket, s3_key)

        # Create and run CodeBuild project
        project_name = f"llama-build-{suffix}"
        codebuild = self._session.client("codebuild")

        codebuild.create_project(
            name=project_name,
            source={"type": "S3", "location": f"{s3_bucket}/{s3_key}"},
            artifacts={"type": "NO_ARTIFACTS"},
            environment={
                "type": "ARM_CONTAINER",
                "image": "aws/codebuild/amazonlinux2-aarch64-standard:3.0",
                "computeType": build_compute_type,
                "privilegedMode": True,
            },
            serviceRole=self._deployment_role,
        )

        build_id = codebuild.start_build(projectName=project_name)["build"]["id"]
        logger.info("Started CodeBuild: %s", project_name)

        status = self._wait_for_build(build_id)

        # Clean up temporary build resources
        codebuild.delete_project(name=project_name)
        self._session.client("s3").delete_object(Bucket=s3_bucket, Key=s3_key)

        if status != "SUCCEEDED":
            raise RuntimeError(f"CodeBuild failed with status: {status}")

        logger.info("Image pushed: %s", ecr_image)
        return ecr_image

    def _create_source_zip(
        self,
        fileobj: Any,
        project_dir: Path,
        ecr_image: str,
        repository_name: str,
        ecr_registry: str,
        source_files: list[str] | None,
    ) -> None:
        """Create a zip with source code, Dockerfile, and buildspec."""
        with zipfile.ZipFile(fileobj, "w") as zf:
            # Dockerfile
            zf.writestr("Dockerfile", _generate_dockerfile())

            # requirements.txt from pyproject.toml
            zf.writestr("requirements.txt", _generate_requirements(project_dir))

            # Source files — include everything except venv, __pycache__, .git
            if source_files:
                for name in source_files:
                    path = project_dir / name
                    if path.exists():
                        zf.write(path, name)
            else:
                for path in project_dir.rglob("*"):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(project_dir)
                    parts = rel.parts
                    if any(
                        p
                        in (
                            "__pycache__",
                            ".venv",
                            ".git",
                            ".agentcore",
                            "node_modules",
                        )
                        for p in parts
                    ):
                        continue
                    zf.write(path, str(rel))

            # buildspec.yml
            zf.writestr(
                "buildspec.yml",
                _generate_buildspec(
                    ecr_registry,
                    repository_name,
                    ecr_image,
                    self._region,
                ),
            )

    def _wait_for_build(self, build_id: str, poll_interval: int = 10) -> str:
        """Wait for CodeBuild to complete, streaming log output."""
        codebuild = self._session.client("codebuild")
        logs_client = self._session.client(
            "logs", config=BotoConfig(retries={"max_attempts": 15})
        )

        next_token: str | None = None

        while True:
            info = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
            status = info["buildStatus"]
            log_group = info["logs"].get("groupName")
            stream_name = info["logs"].get("streamName")

            # Stream available logs
            if log_group and stream_name:
                try:
                    kwargs: dict[str, Any] = {
                        "logGroupName": log_group,
                        "logStreamName": stream_name,
                        "startFromHead": True,
                    }
                    if next_token:
                        kwargs["nextToken"] = next_token
                    resp = logs_client.get_log_events(**kwargs)
                    for event in resp["events"]:
                        logger.info("[build] %s", event["message"].rstrip())
                    next_token = resp["nextForwardToken"]
                except Exception:
                    pass

            if status != "IN_PROGRESS":
                return status

            time.sleep(poll_interval)

    # ── AgentCore Runtime CRUD ────────────────────────────────────────────

    def _deploy_runtime(
        self,
        runtime_name: str,
        ecr_uri: str,
        env_vars: dict[str, str],
        max_lifetime: int = 3600,
    ) -> tuple[str, str]:
        """Create or update an AgentCore Runtime. Returns (runtime_id, runtime_arn)."""
        client = self._session.client("bedrock-agentcore-control")

        runtime_config: dict[str, Any] = {
            "roleArn": self._execution_role,
            "agentRuntimeArtifact": {
                "containerConfiguration": {"containerUri": ecr_uri}
            },
            "networkConfiguration": {"networkMode": "PUBLIC"},
            "environmentVariables": env_vars,
            "lifecycleConfiguration": {"maxLifetime": max_lifetime},
            "filesystemConfigurations": [
                {"mountPoint": "/mnt/workspace", "filesystemType": "WORKSPACE"}
            ],
        }

        existing = self._find_runtime(client, runtime_name)

        if existing:
            logger.info("Updating existing runtime: %s", runtime_name)
            client.update_agent_runtime(
                agentRuntimeId=existing["agentRuntimeId"],
                **runtime_config,
            )
            runtime_id = existing["agentRuntimeId"]
            runtime_arn = existing["agentRuntimeArn"]
        else:
            logger.info("Creating new runtime: %s", runtime_name)
            resp = client.create_agent_runtime(
                agentRuntimeName=runtime_name,
                **runtime_config,
            )
            runtime_id = resp["agentRuntimeId"]
            runtime_arn = resp["agentRuntimeArn"]

        self._wait_for_ready(client, runtime_id)
        return runtime_id, runtime_arn

    @staticmethod
    def _find_runtime(client: Any, runtime_name: str) -> dict[str, Any] | None:
        """Find an existing runtime by name."""
        try:
            for rt in client.list_agent_runtimes().get("agentRuntimes", []):
                if rt["agentRuntimeName"] == runtime_name:
                    return rt
        except Exception as e:
            logger.warning("Could not list runtimes: %s", e)
        return None

    @staticmethod
    def _wait_for_ready(client: Any, runtime_id: str) -> None:
        """Poll until the runtime reaches READY status."""
        logger.info("Waiting for runtime to be ready...")
        while True:
            resp = client.get_agent_runtime(agentRuntimeId=runtime_id)
            status = resp["status"]
            if status == "READY":
                logger.info("Runtime is ready!")
                return
            if status in ("CREATE_FAILED", "UPDATE_FAILED", "DELETE_FAILED"):
                raise RuntimeError(f"Runtime failed with status: {status}")
            logger.info("  Status: %s", status)
            time.sleep(10)

    @staticmethod
    def _wait_for_deletion(client: Any, runtime_id: str) -> None:
        """Poll until the runtime is deleted."""
        while True:
            try:
                resp = client.get_agent_runtime(agentRuntimeId=runtime_id)
                status = resp["status"]
                if status == "DELETE_FAILED":
                    logger.warning("Deletion failed with status: %s", status)
                    return
                logger.info("  Status: %s", status)
                time.sleep(5)
            except client.exceptions.ResourceNotFoundException:
                logger.info("Runtime deleted.")
                return

    # ── S3 bucket ─────────────────────────────────────────────────────────

    def _ensure_s3_bucket(self, bucket_name: str) -> None:
        """Create S3 bucket for CodeBuild artifacts if it doesn't exist."""
        s3 = self._session.client("s3")
        try:
            s3.head_bucket(Bucket=bucket_name)
        except ClientError:
            kwargs: dict[str, Any] = {"Bucket": bucket_name}
            if self._region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._region
                }
            s3.create_bucket(**kwargs)
            logger.info("Created S3 bucket: %s", bucket_name)


# ── Helpers ───────────────────────────────────────────────────────────────


def _sanitize_name(name: str) -> str:
    """Sanitize for AgentCore (underscores only, no hyphens)."""
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def _project_name_from_pyproject(project_dir: Path) -> str:
    """Read the project name from pyproject.toml."""
    try:
        import tomllib  # type: ignore
    except ModuleNotFoundError:
        import tomli as tomllib

    text = (project_dir / "pyproject.toml").read_text()
    data = tomllib.loads(text)

    # Try [tool.llamadeploy].name first, then [project].name
    name = data.get("tool", {}).get("llamadeploy", {}).get("name") or data.get(
        "project", {}
    ).get("name")
    if not name:
        raise ValueError("Could not determine project name from pyproject.toml")
    return name


def _generate_dockerfile() -> str:
    """Generate a minimal Dockerfile for AgentCore (ARM64)."""
    return """\
FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src:/app

# llama-agents-agentcore discovers workflows from pyproject.toml at startup
CMD ["python", "-m", "llama_agents.agentcore.main", "--run"]
"""


def _generate_requirements(project_dir: Path) -> str:
    """Extract dependencies from pyproject.toml for the container."""
    try:
        import tomllib  # type: ignore
    except ModuleNotFoundError:
        import tomli as tomllib

    text = (project_dir / "pyproject.toml").read_text()
    data = tomllib.loads(text)
    deps = data.get("project", {}).get("dependencies", [])

    # Always include the agentcore package itself
    lines = list(deps)
    if not any("llama-agents-agentcore" in d for d in lines):
        lines.append("llama-agents-agentcore>=0.7.0")

    return "\n".join(lines) + "\n"


def _parse_deployment_env_vars(project_dir: Path) -> dict[str, str]:
    """Read env / env_files from the project's DeploymentConfig (pyproject.toml)."""
    try:
        from llama_agents.appserver.workflow_loader import parse_environment_variables
        from llama_agents.core.deployment_config import (
            read_deployment_config_from_git_root_or_cwd,
        )

        config = read_deployment_config_from_git_root_or_cwd(project_dir, project_dir)
        return parse_environment_variables(config, project_dir)
    except Exception:
        return {}


def _generate_buildspec(
    ecr_registry: str,
    repository_name: str,
    ecr_image: str,
    region: str,
) -> str:
    """Generate CodeBuild buildspec.yml."""
    return f"""\
version: 0.2
phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {ecr_registry}
  build:
    commands:
      - echo Building the Docker image...
      - docker build -t {repository_name}:latest .
      - docker tag {repository_name}:latest {ecr_image}
  post_build:
    commands:
      - echo Pushing the Docker image...
      - docker push {ecr_image}
"""
