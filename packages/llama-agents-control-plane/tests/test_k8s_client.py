"""Unit tests for k8s_client.py"""

import base64
import sys
from collections.abc import Iterator
from typing import Generator, TypedDict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from kubernetes.client import (
    V1Container,
    V1ObjectMeta,
    V1OwnerReference,
    V1Pod,
    V1PodSpec,
    V1ReplicaSet,
)
from kubernetes.client.exceptions import ApiException
from llama_agents.control_plane import k8s_client
from llama_agents.control_plane.k8s_client import (
    LogLine,
    _append_random_suffix,
    get_replicaset_pods_for_deployment,
    stream_container_logs,
)
from llama_agents.core.schema.deployments import (
    DeploymentResponse,
    DeploymentUpdate,
    LlamaDeploymentCRD,
)

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


@pytest.fixture
def mock_k8s() -> Generator[MagicMock, None, None]:
    with patch("llama_agents.control_plane.k8s_client._k8s_client") as mock_k8s:
        yield mock_k8s


@pytest.fixture
def mock_validate() -> Generator[MagicMock, None, None]:
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate_deployment_id:
        yield mock_validate_deployment_id


class DeploymentMockParams(TypedDict, total=False):
    name: str
    namespace: str
    deployment_id: str
    project_id: str
    repo_url: str
    git_ref: str
    deployment_file_path: str
    secret_name: str | None
    status: str
    auth_token: str | None


def create_deployment_mock(
    name: str = "test-deploy",
    namespace: str = "llama-agents",
    deployment_id: str = "test-deploy",
    project_id: str = "test-project",
    repo_url: str = "https://github.com/user/repo.git",
    git_ref: str = "main",
    deployment_file_path: str = "llama_deploy.yaml",
    secret_name: str | None = None,
    status: str = "Running",
    auth_token: str | None = None,
) -> dict[str, object]:
    return {
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "name": deployment_id,
            "projectId": project_id,
            "repoUrl": repo_url,
            "gitRef": git_ref,
            "deploymentFilePath": deployment_file_path,
            "secretName": secret_name,
        },
        "status": {"phase": status, "authToken": auth_token},
    }


def create_deployment_mock_crd(
    **kwargs: Unpack[DeploymentMockParams],
) -> LlamaDeploymentCRD:
    return LlamaDeploymentCRD.model_validate(create_deployment_mock(**kwargs))


def test_append_random_suffix() -> None:
    """Test random suffix generation"""
    result = _append_random_suffix("test", 20)
    assert result.startswith("test-")
    assert len(result) == 10  # "test-" + 5 chars

    # Test with empty string
    result = _append_random_suffix("", 10)
    assert len(result) == 5  # Just the random part

    # Test truncation - takes max_length - 5 - 1 chars, then dash, then 5 hex chars
    long_name = "a" * 50
    result = _append_random_suffix(long_name, 20)
    assert len(result) == 20
    assert result == "aaaaaaaaaaaaaa-" + result[-5:]  # 14 a's + dash + 5 hex chars
    assert len(result.split("-")[-1]) == 5  # Last part is 5 hex chars


# Integration-style tests for the main functions
@pytest.mark.asyncio
async def test_validate_deployment_id_available(mock_k8s: MagicMock) -> None:
    """Test deployment ID validation when available"""
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.side_effect = ApiException(
        status=404
    )

    result = await k8s_client.validate_deployment_id("test-deploy")
    assert result is True


@pytest.mark.asyncio
async def test_validate_deployment_id_taken(mock_k8s: MagicMock) -> None:
    """Test deployment ID validation when taken"""
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        create_deployment_mock(
            name="test",
            deployment_id="test",
        )
    )

    result = await k8s_client.validate_deployment_id("test-deploy")
    assert result is False


@pytest.mark.asyncio
async def test_find_deployment_id_first_try(mock_validate: MagicMock) -> None:
    """Test deployment ID generation when first attempt works"""
    mock_validate.return_value = True

    result = await k8s_client.find_deployment_id("My Service")
    assert result == "my-service"
    mock_validate.assert_called_once_with("my-service")


@pytest.mark.asyncio
async def test_find_deployment_id_with_collision(mock_validate: MagicMock) -> None:
    """Test deployment ID generation with name collision"""
    # First call returns False (taken), second returns True (available)
    mock_validate.side_effect = [False, True]

    result = await k8s_client.find_deployment_id("test")
    # Should get a random suffix since "test" was taken
    assert result.startswith("test-")
    assert len(result) == 10  # "test-" + 5 random chars
    assert mock_validate.call_count == 2


@pytest.mark.asyncio
async def test_validate_deployment_token_success(mock_k8s: MagicMock) -> None:
    """Test successful token validation"""
    mock_deployment = create_deployment_mock(
        name="test-deployment",
        auth_token="valid-token-123",
    )
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        mock_deployment
    )

    result = await k8s_client.validate_deployment_token(
        "test-deployment", "valid-token-123"
    )
    assert result is not None
    assert result.status.authToken == "valid-token-123"
    assert result.metadata.name == "test-deployment"


@pytest.mark.asyncio
async def test_validate_deployment_token_wrong_token(mock_k8s: MagicMock) -> None:
    """Test token validation with wrong token"""
    mock_deployment = create_deployment_mock(
        name="test-deployment",
        auth_token="different-token",
    )
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        mock_deployment
    )

    result = await k8s_client.validate_deployment_token(
        "test-deployment", "wrong-token"
    )
    assert result is None


@pytest.mark.asyncio
async def test_validate_deployment_token_no_token(mock_k8s: MagicMock) -> None:
    """Test token validation when deployment has no token"""
    mock_deployment = create_deployment_mock(
        name="test-deployment",
        auth_token=None,
    )
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        mock_deployment
    )

    result = await k8s_client.validate_deployment_token("test-deployment", "any-token")
    assert result is None


@pytest.mark.asyncio
async def test_validate_deployment_token_deployment_not_found(
    mock_k8s: MagicMock,
) -> None:
    """Test token validation when deployment doesn't exist"""
    from kubernetes.client.exceptions import ApiException

    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.side_effect = ApiException(
        status=404
    )

    result = await k8s_client.validate_deployment_token(
        "nonexistent-deployment", "any-token"
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_secret_names_success(mock_k8s: MagicMock) -> None:
    """Test successful retrieval of secret names"""
    mock_secret = Mock()
    mock_secret.data = {
        "API_KEY": "value1",
        "DATABASE_URL": "value2",
        "GITHUB_PAT": "token",
    }
    mock_k8s.k8s_core_v1.read_namespaced_secret.return_value = mock_secret

    result = await k8s_client.get_secret_names("test-secret")
    assert set(result) == {"API_KEY", "DATABASE_URL", "GITHUB_PAT"}


@pytest.mark.asyncio
async def test_get_secret_names_not_found(mock_k8s: MagicMock) -> None:
    """Test handling of secret not found"""
    mock_k8s.k8s_core_v1.read_namespaced_secret.side_effect = ApiException(404)

    result = await k8s_client.get_secret_names("nonexistent-secret")
    assert result == []


@pytest.mark.asyncio
async def test_get_secret_names_empty_secret(mock_k8s: MagicMock) -> None:
    """Test handling of secret with no data"""
    mock_secret = Mock()
    mock_secret.data = None
    mock_k8s.k8s_core_v1.read_namespaced_secret.return_value = mock_secret

    result = await k8s_client.get_secret_names("empty-secret")
    assert result == []


@patch("llama_agents.control_plane.k8s_client.get_secret_names")
@pytest.mark.asyncio
async def test_get_secret_names_batch(mock_get_secret_names: MagicMock) -> None:
    """Test batch retrieval of secret names"""
    # Mock individual calls
    mock_get_secret_names.side_effect = [
        ["API_KEY", "DATABASE_URL"],
        ["GITHUB_PAT"],
        None,
    ]

    result = await k8s_client.get_secret_names_batch(["secret1", "secret2", "secret3"])

    expected = {
        "secret1": ["API_KEY", "DATABASE_URL"],
        "secret2": ["GITHUB_PAT"],
        "secret3": None,
    }
    assert result == expected


def test_llamadeployment_crd_missing_status(mock_k8s: MagicMock) -> None:
    """CRDs without a status field (not yet reconciled) should default to Pending phase"""
    raw = create_deployment_mock(name="new-deploy", deployment_id="new-deploy")
    del raw["status"]

    crd = LlamaDeploymentCRD.model_validate(raw)
    assert crd.status.phase is None

    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "default"
    response = k8s_client._llamadeployment_to_response(crd)
    assert response.status == "Pending"


def test_llamadeployment_to_response_basic(mock_k8s: MagicMock) -> None:
    """Test basic conversion from LlamaDeployment to DeploymentResponse"""
    llamadeployment = create_deployment_mock_crd(
        name="test-deployment",
        deployment_id="test-deployment",
        project_id="test-project",
        repo_url="https://github.com/test/repo.git",
        git_ref="abc123",
        deployment_file_path="deploy.yml",
        secret_name="test-secret",
    )

    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "default"

    result = k8s_client._llamadeployment_to_response(llamadeployment)

    assert isinstance(result, DeploymentResponse)
    assert result.id == "test-deployment"
    assert result.name == "test-deployment"
    assert result.repo_url == "https://github.com/test/repo.git"
    assert result.git_ref == "abc123"
    assert result.has_personal_access_token is False
    assert result.secret_names is None


def test_llamadeployment_to_response_with_secrets(mock_k8s: MagicMock) -> None:
    """Test conversion with secret names including GITHUB_PAT filtering"""
    llamadeployment = create_deployment_mock_crd(
        name="test-deployment",
        project_id="test-project",
        repo_url="https://github.com/test/repo.git",
        git_ref="abc123",
        deployment_file_path="deploy.yml",
        secret_name="test-secret",
    )

    secret_names = ["API_KEY", "DATABASE_URL", "GITHUB_PAT"]

    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "default"

    result = k8s_client._llamadeployment_to_response(llamadeployment, secret_names)

    assert result.has_personal_access_token is True  # GITHUB_PAT was present
    assert result.secret_names == [
        "API_KEY",
        "DATABASE_URL",
    ]


def test_llamadeployment_to_response_only_github_pat(mock_k8s: MagicMock) -> None:
    """Test conversion when only GITHUB_PAT is in secrets"""
    llamadeployment = create_deployment_mock_crd(
        name="test-deployment",
        project_id="test-project",
        repo_url="https://github.com/test/repo.git",
        git_ref="abc123",
        deployment_file_path="deploy.yml",
        secret_name="test-secret",
    )

    secret_names = ["GITHUB_PAT"]

    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "default"

    result = k8s_client._llamadeployment_to_response(llamadeployment, secret_names)

    assert result.has_personal_access_token is True
    assert result.secret_names is None  # No secrets left after filtering GITHUB_PAT


def test_llamadeployment_to_response_with_ingress(mock_k8s: MagicMock) -> None:
    """Test conversion with ingress enabled"""
    llamadeployment = create_deployment_mock_crd(
        name="test-deployment",
        git_ref="abc123",
    )

    mock_k8s.enable_ingress = True
    mock_k8s.domain = "127.0.0.1.nip.io"

    result = k8s_client._llamadeployment_to_response(llamadeployment)

    assert result.apiserver_url is not None
    assert "test-deployment.127.0.0.1.nip.io:8090" in str(result.apiserver_url)


def test_llamadeployment_to_response_building_mapped_to_pending(
    mock_k8s: MagicMock,
) -> None:
    """Building phase is mapped to Pending for backward-compatible clients."""
    llamadeployment = create_deployment_mock_crd(
        name="build-deploy",
        status="Building",
    )
    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "default"

    result = k8s_client._llamadeployment_to_response(llamadeployment)

    assert result.status == "Pending"
    assert result.warning is not None
    assert "Building" in result.warning


def test_llamadeployment_to_response_buildfailed_mapped_to_failed(
    mock_k8s: MagicMock,
) -> None:
    """BuildFailed phase is mapped to Failed for backward-compatible clients."""
    llamadeployment = create_deployment_mock_crd(
        name="fail-deploy",
        status="BuildFailed",
    )
    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "default"

    result = k8s_client._llamadeployment_to_response(llamadeployment)

    assert result.status == "Failed"
    assert result.warning is not None
    assert "BuildFailed" in result.warning


def test_llamadeployment_to_response_running_not_mapped(
    mock_k8s: MagicMock,
) -> None:
    """Non-build phases pass through unchanged with no warning."""
    llamadeployment = create_deployment_mock_crd(
        name="run-deploy",
        status="Running",
    )
    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "default"

    result = k8s_client._llamadeployment_to_response(llamadeployment)

    assert result.status == "Running"
    assert result.warning is None


def test_llamadeployment_to_response_awaitingcode_mapped_to_pending(
    mock_k8s: MagicMock,
) -> None:
    """AwaitingCode phase is mapped to Pending for backward-compatible clients."""
    llamadeployment = create_deployment_mock_crd(
        name="awaiting-deploy",
        status="AwaitingCode",
    )
    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "default"

    result = k8s_client._llamadeployment_to_response(llamadeployment)

    assert result.status == "Pending"
    assert result.warning is not None
    assert "Waiting for code push" in result.warning


@patch("llama_agents.control_plane.k8s_client.get_deployment")
@patch("llama_agents.control_plane.k8s_client._create_k8s_secret")
@pytest.mark.asyncio
async def test_update_deployment_basic_fields(
    mock_create_secret: MagicMock,
    mock_get_deployment: MagicMock,
    mock_k8s: MagicMock,
) -> None:
    """Test updating basic deployment fields without secrets"""

    # Mock existing deployment
    existing_deployment = create_deployment_mock(
        name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/old-repo.git",
        deployment_file_path="old_deploy.yml",
    )
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        existing_deployment
    )

    # Mock final result
    mock_get_deployment.return_value = DeploymentResponse(
        id="test-deploy",
        display_name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/new-repo.git",  # updated
        git_ref="",
        deployment_file_path="new_deploy.yml",  # updated
        has_personal_access_token=False,
        secret_names=None,
        apiserver_url=None,
        status="Running",
    )

    update = DeploymentUpdate(
        repo_url="https://github.com/user/new-repo.git",
        deployment_file_path="new_deploy.yml",
    )

    result = await k8s_client.update_deployment("test-deploy", update)

    # Verify the deployment was fetched
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.assert_called_once_with(
        group="deploy.llamaindex.ai",
        version="v1",
        namespace=mock_k8s.namespace,
        plural="llamadeployments",
        name="test-deploy",
    )

    # Verify the deployment was updated with new spec
    mock_k8s.k8s_custom_objects.replace_namespaced_custom_object.assert_called_once()
    call_args = mock_k8s.k8s_custom_objects.replace_namespaced_custom_object.call_args
    updated_body = call_args[1]["body"]

    # Verify Kubernetes metadata is present
    assert updated_body["apiVersion"] == "deploy.llamaindex.ai/v1"
    assert updated_body["kind"] == "LlamaDeployment"

    # Verify spec fields
    assert updated_body["spec"]["repoUrl"] == "https://github.com/user/new-repo.git"
    assert updated_body["spec"]["deploymentFilePath"] == "new_deploy.yml"
    assert updated_body["spec"]["projectId"] == "test-project"  # unchanged

    # No secret operations should have been called
    mock_create_secret.assert_not_called()

    # Final deployment should be returned
    assert result is not None
    assert result.repo_url == "https://github.com/user/new-repo.git"


@patch(
    "llama_agents.control_plane.k8s_client.get_deployment",
    new_callable=AsyncMock,
)
@patch(
    "llama_agents.control_plane.k8s_client._create_k8s_secret",
    new_callable=AsyncMock,
)
@pytest.mark.asyncio
async def test_update_deployment_with_secrets(
    mock_create_secret: MagicMock,
    mock_get_deployment: MagicMock,
    mock_k8s: MagicMock,
) -> None:
    """Test updating deployment with secret changes"""

    # Mock existing deployment
    existing_deployment = create_deployment_mock(
        name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/repo.git",
        deployment_file_path="deploy.yml",
        secret_name="test-deploy-secrets",
    )
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        existing_deployment
    )

    # Mock existing secret
    mock_secret = Mock()
    mock_secret.data = {
        "EXISTING_SECRET": base64.b64encode(b"old_value").decode(),
        "REMOVE_ME": base64.b64encode(b"will_be_removed").decode(),
    }
    mock_k8s.k8s_core_v1.read_namespaced_secret.return_value = mock_secret

    # Mock final result
    mock_get_deployment.return_value = DeploymentResponse(
        id="test-deploy",
        display_name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/repo.git",
        git_ref="",
        deployment_file_path="deploy.yml",
        has_personal_access_token=True,  # PAT was added
        secret_names=["EXISTING_SECRET", "NEW_SECRET"],  # updated secrets
        apiserver_url=None,
        status="Running",
    )

    update = DeploymentUpdate(
        personal_access_token="ghp_newtoken",
        secrets={
            "NEW_SECRET": "new_value",
            "REMOVE_ME": None,  # remove this
        },
    )

    result = await k8s_client.update_deployment("test-deploy", update)

    # Verify secret was read
    mock_k8s.k8s_core_v1.read_namespaced_secret.assert_called_once_with(
        name="test-deploy-secrets",
        namespace=mock_k8s.namespace,
    )

    # Verify secret was updated with correct changes
    mock_create_secret.assert_called_once()
    secret_call_args = mock_create_secret.call_args
    updated_secrets = secret_call_args[0][1]  # Second argument

    assert updated_secrets["GITHUB_PAT"] == "ghp_newtoken"  # PAT added
    assert updated_secrets["NEW_SECRET"] == "new_value"  # new secret added
    assert updated_secrets["EXISTING_SECRET"] == "old_value"  # existing preserved
    assert "REMOVE_ME" not in updated_secrets  # removed

    # Verify deployment was updated
    mock_k8s.k8s_custom_objects.replace_namespaced_custom_object.assert_called_once()

    assert result is not None
    assert result.has_personal_access_token is True


@pytest.mark.asyncio
async def test_update_deployment_not_found(mock_k8s: MagicMock) -> None:
    """Test updating non-existent deployment"""

    # Mock deployment not found
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.side_effect = ApiException(
        status=404
    )

    update = DeploymentUpdate(repo_url="https://github.com/user/new-repo.git")

    result = await k8s_client.update_deployment("nonexistent", update)
    assert result is None


@patch("llama_agents.control_plane.k8s_client.get_deployment")
@patch("llama_agents.control_plane.k8s_client._create_k8s_secret")
@pytest.mark.asyncio
async def test_update_deployment_secret_required_but_missing(
    mock_create_secret: MagicMock,
    mock_get_deployment: MagicMock,
    mock_k8s: MagicMock,
) -> None:
    """Test updating with secret changes but no existing secret - should create lazily"""

    # Mock existing deployment without secret
    existing_deployment = create_deployment_mock(
        name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/repo.git",
        deployment_file_path="deploy.yml",
    )
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        existing_deployment
    )

    # Mock final result
    mock_get_deployment.return_value = DeploymentResponse(
        id="test-deploy",
        display_name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/repo.git",
        git_ref="",
        deployment_file_path="deploy.yml",
        has_personal_access_token=True,
        secret_names=None,
        apiserver_url=None,
        status="Running",
    )

    update = DeploymentUpdate(personal_access_token="ghp_token")

    result = await k8s_client.update_deployment("test-deploy", update)

    # Verify secret was created with just the PAT
    mock_create_secret.assert_called_once()
    secret_call_args = mock_create_secret.call_args
    secret_name = secret_call_args[0][0]
    updated_secrets = secret_call_args[0][1]

    assert secret_name == "test-deploy-secrets"
    assert updated_secrets["GITHUB_PAT"] == "ghp_token"

    # Verify deployment was updated with secret name
    mock_k8s.k8s_custom_objects.replace_namespaced_custom_object.assert_called_once()
    call_args = mock_k8s.k8s_custom_objects.replace_namespaced_custom_object.call_args
    updated_body = call_args[1]["body"]

    # Verify Kubernetes metadata is present
    assert updated_body["apiVersion"] == "deploy.llamaindex.ai/v1"
    assert updated_body["kind"] == "LlamaDeployment"

    # Verify spec fields
    assert updated_body["spec"]["secretName"] == "test-deploy-secrets"

    assert result is not None


@patch("llama_agents.control_plane.k8s_client.get_deployment")
@patch("llama_agents.control_plane.k8s_client._create_k8s_secret")
@pytest.mark.asyncio
async def test_update_deployment_secret_not_found(
    mock_create_secret: MagicMock,
    mock_get_deployment: MagicMock,
    mock_k8s: MagicMock,
) -> None:
    """Test updating when secret is referenced but doesn't exist - should create it"""

    # Mock existing deployment with secret reference
    existing_deployment = create_deployment_mock(
        name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/repo.git",
        deployment_file_path="deploy.yml",
        secret_name="missing-secret",
    )
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        existing_deployment
    )

    # Mock secret not found
    mock_k8s.k8s_core_v1.read_namespaced_secret.side_effect = ApiException(status=404)

    # Mock final result
    mock_get_deployment.return_value = DeploymentResponse(
        id="test-deploy",
        display_name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/repo.git",
        git_ref="",
        deployment_file_path="deploy.yml",
        has_personal_access_token=True,
        secret_names=None,
        apiserver_url=None,
        status="Running",
    )

    update = DeploymentUpdate(personal_access_token="ghp_token")

    result = await k8s_client.update_deployment("test-deploy", update)

    # Verify secret was created with just the PAT (starting from empty)
    mock_create_secret.assert_called_once()
    secret_call_args = mock_create_secret.call_args
    secret_name = secret_call_args[0][0]
    updated_secrets = secret_call_args[0][1]

    assert secret_name == "missing-secret"
    assert updated_secrets["GITHUB_PAT"] == "ghp_token"

    # Verify deployment was updated
    mock_k8s.k8s_custom_objects.replace_namespaced_custom_object.assert_called_once()

    assert result is not None


@pytest.mark.asyncio
async def test_update_deployment_api_error(mock_k8s: MagicMock) -> None:
    """Test API error during deployment update"""
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.side_effect = ApiException(
        status=500, reason="Internal Server Error"
    )

    update = DeploymentUpdate(repo_url="https://github.com/user/new-repo.git")

    with pytest.raises(ApiException):
        await k8s_client.update_deployment("test-deploy", update)


# Tests for create_deployment with git_ref
@patch("llama_agents.control_plane.k8s_client.find_deployment_id")
@pytest.mark.asyncio
async def test_create_deployment_with_git_ref(
    mock_find_deployment_id: MagicMock,
    mock_k8s: MagicMock,
) -> None:
    """Test create_deployment with git_ref parameter"""
    # Mock the dependencies
    mock_find_deployment_id.return_value = "test-deploy"

    # Mock the k8s client calls
    mock_k8s.k8s_custom_objects.create_namespaced_custom_object.return_value = {}
    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "test-namespace"

    result = await k8s_client.create_deployment(
        project_id="test-project",
        display_name="Test Deploy",
        repo_url="https://github.com/user/repo.git",
        git_ref="main",
        deployment_file_path="deploy.yml",
    )

    # Verify the deployment was created
    assert result.git_ref == "main"
    assert result.display_name == "Test Deploy"


@patch("llama_agents.control_plane.k8s_client.find_deployment_id")
@patch("llama_agents.control_plane.k8s_client._create_k8s_secret")
@pytest.mark.asyncio
async def test_create_deployment_with_git_ref_and_pat(
    mock_create_secret: MagicMock,
    mock_find_deployment_id: MagicMock,
    mock_k8s: MagicMock,
) -> None:
    """Test create_deployment with git_ref and PAT"""
    mock_find_deployment_id.return_value = "test-deploy"
    mock_create_secret.return_value = None

    # Mock the k8s client calls
    mock_k8s.k8s_custom_objects.create_namespaced_custom_object.return_value = {}
    mock_k8s.enable_ingress = False
    mock_k8s.namespace = "test-namespace"

    result = await k8s_client.create_deployment(
        project_id="test-project",
        display_name="Test Deploy",
        repo_url="https://github.com/user/repo.git",
        git_ref="feature-branch",
        pat="ghp_token123",
    )

    assert result.git_ref == "feature-branch"
    assert result.has_personal_access_token is True


# Tests for update_deployment with git_ref validation
@pytest.mark.asyncio
async def test_update_deployment_git_ref_validation_error(mock_k8s: MagicMock) -> None:
    """Test update_deployment with invalid git_ref continues with warning"""
    # Mock existing deployment
    existing_deployment = create_deployment_mock(
        name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/repo.git",
        git_ref="main",
    )
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        existing_deployment
    )

    # Mock updated deployment for get_deployment call
    mock_k8s.k8s_custom_objects.replace_namespaced_custom_object.return_value = (
        existing_deployment
    )
    mock_k8s.domain = "127.0.0.1.nip.io"
    mock_k8s.enable_ingress = True

    update = DeploymentUpdate(git_ref="invalid")

    # Should succeed with warning, not raise exception
    result = await k8s_client.update_deployment("test-deploy", update)

    # Verify the update was applied
    assert result is not None


@patch("llama_agents.control_plane.k8s_client.get_deployment")
@pytest.mark.asyncio
async def test_update_deployment_with_git_ref_success(
    mock_get_deployment: MagicMock, mock_k8s: MagicMock
) -> None:
    """Test successful update_deployment with git_ref"""
    # Mock existing deployment
    existing_deployment = create_deployment_mock(
        name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/repo.git",
        git_ref="main",
    )
    mock_k8s.k8s_custom_objects.get_namespaced_custom_object.return_value = (
        existing_deployment
    )

    # Mock final result
    mock_get_deployment.return_value = DeploymentResponse(
        id="test-deploy",
        display_name="test-deploy",
        project_id="test-project",
        repo_url="https://github.com/user/repo.git",
        git_ref="feature-branch",  # updated
        deployment_file_path="deploy.yml",
        status="Running",
        has_personal_access_token=False,
        secret_names=None,
        apiserver_url=None,
    )

    update = DeploymentUpdate(git_ref="feature-branch")
    result = await k8s_client.update_deployment("test-deploy", update)

    # Verify the deployment was updated
    mock_k8s.k8s_custom_objects.replace_namespaced_custom_object.assert_called_once()

    assert result is not None
    assert result.git_ref == "feature-branch"


@pytest.mark.asyncio
async def test_get_latest_replicaset_for_deployment_not_found(
    mock_k8s: MagicMock,
) -> None:
    mock_k8s.k8s_apps_v1.read_namespaced_deployment.side_effect = ApiException(
        status=404
    )
    result = await k8s_client.get_latest_replicaset_for_deployment("nope")
    assert result is None


@pytest.mark.asyncio
async def test_get_latest_replicaset_for_deployment_picks_highest_revision(
    mock_k8s: MagicMock,
) -> None:
    # Mock deployment
    dep = Mock()
    dep.metadata = Mock(uid="dep-uid")
    mock_k8s.k8s_apps_v1.read_namespaced_deployment.return_value = dep

    # Two ReplicaSets with revisions 1 and 2, both owned by dep-uid
    rs1 = V1ReplicaSet(
        metadata=V1ObjectMeta(
            name="rs-1",
            uid="rs-uid-1",
            annotations={"deployment.kubernetes.io/revision": "1"},
            owner_references=[
                V1OwnerReference(
                    api_version="apps/v1", kind="Deployment", uid="dep-uid", name="d"
                )
            ],
        )
    )
    rs2 = V1ReplicaSet(
        metadata=V1ObjectMeta(
            name="rs-2",
            uid="rs-uid-2",
            annotations={"deployment.kubernetes.io/revision": "2"},
            owner_references=[
                V1OwnerReference(
                    api_version="apps/v1", kind="Deployment", uid="dep-uid", name="d"
                )
            ],
        )
    )
    rs_list = Mock(items=[rs1, rs2])
    mock_k8s.k8s_apps_v1.list_namespaced_replica_set.return_value = rs_list

    result = await k8s_client.get_latest_replicaset_for_deployment("dep")
    assert result is not None
    assert result.metadata is not None
    assert result.metadata.name == "rs-2"


@pytest.mark.asyncio
async def test_stream_replicaset_logs_follow(mock_k8s: MagicMock) -> None:
    # Patch latest replicaset helper to provide a fixed RS uid
    with patch(
        "llama_agents.control_plane.k8s_client.get_latest_replicaset_for_deployment",
        return_value=V1ReplicaSet(metadata=V1ObjectMeta(uid="rs-uid")),
    ):
        # One pod owned by this RS with one container
        pod = V1Pod(
            metadata=V1ObjectMeta(
                name="pod-1",
                owner_references=[
                    V1OwnerReference(
                        api_version="apps/v1",
                        kind="ReplicaSet",
                        uid="rs-uid",
                        name="rs",
                    )
                ],
            ),
            spec=V1PodSpec(containers=[V1Container(name="app")]),
        )
        mock_k8s.k8s_core_v1.list_namespaced_pod.return_value = Mock(items=[pod])

        class FakeStreamResponse:
            def stream(
                self, amt: int | None = None, decode_content: bool = False
            ) -> Iterator[bytes]:
                yield b"hello\n"
                yield b"world\n"

        # Streamed content for follow=True path
        mock_k8s.k8s_core_v1.read_namespaced_pod_log.return_value = FakeStreamResponse()

        gen = k8s_client.stream_replicaset_logs("dep")
        # Pull a couple lines then stop
        first = await anext(gen)
        second = await anext(gen)

        assert first == LogLine(pod="pod-1", container="app", text="hello")
        assert second == LogLine(pod="pod-1", container="app", text="world")


@pytest.mark.asyncio
async def test_get_replicaset_pods_for_deployment_filters_by_owner(
    mock_k8s: MagicMock,
) -> None:
    # Latest RS uid
    with patch(
        "llama_agents.control_plane.k8s_client.get_latest_replicaset_for_deployment",
        return_value=V1ReplicaSet(metadata=V1ObjectMeta(uid="rs-uid")),
    ):
        # Two pods: one owned, one not
        owned = V1Pod(
            metadata=V1ObjectMeta(
                name="p1",
                owner_references=[
                    V1OwnerReference(
                        api_version="apps/v1",
                        kind="ReplicaSet",
                        uid="rs-uid",
                        name="rs",
                    )
                ],
            ),
            spec=V1PodSpec(containers=[V1Container(name="c")]),
        )
        other = V1Pod(
            metadata=V1ObjectMeta(name="p2"),
            spec=V1PodSpec(containers=[V1Container(name="c")]),
        )
        mock_k8s.k8s_core_v1.list_namespaced_pod.return_value = Mock(
            items=[owned, other]
        )

        pods = await get_replicaset_pods_for_deployment("dep")
        assert [p.metadata.name for p in pods if p.metadata is not None] == ["p1"]


@pytest.mark.asyncio
async def test_stream_container_logs_single_pod_multi_lines(
    mock_k8s: MagicMock,
) -> None:
    class FakeStreamResponse:
        def stream(
            self, amt: int | None = None, decode_content: bool = False
        ) -> Iterator[bytes]:
            yield b"a\n"
            yield b"b\n"

    mock_k8s.k8s_core_v1.read_namespaced_pod_log.return_value = FakeStreamResponse()

    cancel, gen = await stream_container_logs("pod-1", "app")
    assert await anext(gen) == "a"
    assert await anext(gen) == "b"
