"""Unit tests to validate find_deployment_id behavior and expose potential bugs"""

from unittest.mock import patch

import pytest
from kubernetes.client.exceptions import ApiException
from llama_agents.control_plane import k8s_client


@pytest.mark.asyncio
async def test_find_deployment_id_no_suffix_when_available() -> None:
    """Test that no suffix is added when base deployment ID is available"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = True  # Base ID is available

        result = await k8s_client.find_deployment_id("my-service")

        # Should return the cleaned name without any suffix
        assert result == "my-service"
        # Should only call validate once with the base ID
        mock_validate.assert_called_once_with("my-service")


@pytest.mark.asyncio
async def test_find_deployment_id_suffix_when_base_taken() -> None:
    """Test that suffix is added when base deployment ID is taken"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        # First call (base ID) returns False (taken), second call returns True (available)
        mock_validate.side_effect = [False, True]

        result = await k8s_client.find_deployment_id("my-service")

        # Should return the base name with a suffix
        assert result.startswith("my-service-")
        assert len(result) == len("my-service-") + 5  # 5 char hex suffix
        # Should call validate twice - once for base, once for suffixed version
        assert mock_validate.call_count == 2

        # First call should be with base ID
        assert mock_validate.call_args_list[0][0][0] == "my-service"
        # Second call should be with suffixed version
        assert mock_validate.call_args_list[1][0][0].startswith("my-service-")


@pytest.mark.asyncio
async def test_find_deployment_id_short_name_gets_suffix() -> None:
    """Test that names shorter than 3 characters always get a suffix"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = True  # Available

        result = await k8s_client.find_deployment_id("ab")

        # Should have a suffix added since name is < 3 chars
        assert len(result) == 8  # "ab-" + 5 char hex suffix
        assert result.startswith("ab-")


@pytest.mark.asyncio
async def test_find_deployment_id_three_char_name_no_suffix() -> None:
    """Test that names with 3 characters don't get a suffix"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = True  # Available

        result = await k8s_client.find_deployment_id("abc")

        # Should NOT have a suffix added since name is >= 3 chars
        assert result == "abc"
        mock_validate.assert_called_once_with("abc")


@pytest.mark.asyncio
async def test_find_deployment_id_name_cleaning() -> None:
    """Test that special characters are properly cleaned"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = True  # Available

        result = await k8s_client.find_deployment_id("My Service!@#$%")

        # Should clean the name and return without suffix if available
        assert result == "my-service"
        mock_validate.assert_called_once_with("my-service")


@pytest.mark.asyncio
async def test_find_deployment_id_real_validation() -> None:
    """Test find_deployment_id with real validation to check for bugs"""
    # Use real validate_deployment_id but mock the K8s client
    with patch("llama_agents.control_plane.k8s_client._k8s_client") as mock_k8s_client:
        # Mock a 404 response (deployment doesn't exist, so ID is available)

        mock_k8s_client.k8s_custom_objects.get_namespaced_custom_object.side_effect = (
            ApiException(status=404)
        )
        mock_k8s_client.namespace = "test-namespace"

        result = await k8s_client.find_deployment_id("my-service")

        # If validation works correctly, should return base name without suffix
        assert result == "my-service"

        # Should have called get_namespaced_custom_object once with base name
        mock_k8s_client.k8s_custom_objects.get_namespaced_custom_object.assert_called_once()
        call_args = (
            mock_k8s_client.k8s_custom_objects.get_namespaced_custom_object.call_args
        )
        assert call_args[1]["name"] == "my-service"


@pytest.mark.asyncio
async def test_create_deployment_id_behavior() -> None:
    """Test create_deployment to verify deployment ID behavior end-to-end"""

    # Mock all the dependencies
    with (
        patch("llama_agents.core.git.git_util.clone_repo") as mock_clone,
        patch(
            "llama_agents.control_plane.k8s_client.validate_deployment_id"
        ) as mock_validate,
        patch("llama_agents.control_plane.k8s_client._k8s_client") as mock_k8s_client,
        patch("tempfile.mkdtemp") as mock_temp,
    ):
        # Setup mocks
        mock_clone.return_value = "abc123"
        mock_validate.return_value = True  # Base deployment ID is available
        mock_temp.return_value = "/tmp/test"

        # Mock K8s client methods
        mock_k8s_client.namespace = "test-namespace"
        mock_k8s_client.enable_ingress = False
        mock_k8s_client.k8s_custom_objects.create_namespaced_custom_object.return_value = {}

        # Call create_deployment
        result = await k8s_client.create_deployment(
            project_id="test-project",
            display_name="Test Service",
            repo_url="https://github.com/test/repo.git",
        )

        # The deployment ID should be the cleaned name without suffix
        assert result.id == "test-service"

        # Verify validate_deployment_id was called with the clean base name
        mock_validate.assert_called_once_with("test-service")

        # Verify the K8s object was created with the correct name
        create_call = mock_k8s_client.k8s_custom_objects.create_namespaced_custom_object
        create_call.assert_called_once()
        created_object = create_call.call_args[1]["body"]
        assert created_object["metadata"]["name"] == "test-service"


@pytest.mark.asyncio
async def test_create_deployment_with_collision() -> None:
    """Test create_deployment when there's a name collision"""

    with (
        patch("llama_agents.core.git.git_util.clone_repo") as mock_clone,
        patch(
            "llama_agents.control_plane.k8s_client.validate_deployment_id"
        ) as mock_validate,
        patch("llama_agents.control_plane.k8s_client._k8s_client") as mock_k8s_client,
        patch("tempfile.mkdtemp") as mock_temp,
    ):
        # Setup mocks
        mock_clone.return_value = "abc123"
        # First call (base ID) returns False, second call returns True
        mock_validate.side_effect = [False, True]
        mock_temp.return_value = "/tmp/test"

        # Mock K8s client methods
        mock_k8s_client.namespace = "test-namespace"
        mock_k8s_client.enable_ingress = False
        mock_k8s_client.k8s_custom_objects.create_namespaced_custom_object.return_value = {}

        # Call create_deployment
        result = await k8s_client.create_deployment(
            project_id="test-project",
            display_name="Test Service",
            repo_url="https://github.com/test/repo.git",
        )

        # The deployment ID should have a suffix since base was taken
        assert result.id.startswith("test-service-")
        assert len(result.id) == len("test-service-") + 5  # 5 char hex suffix

        # Verify validate_deployment_id was called twice
        assert mock_validate.call_count == 2
        # First call with base name
        assert mock_validate.call_args_list[0][0][0] == "test-service"
        # Second call with suffixed name
        assert mock_validate.call_args_list[1][0][0].startswith("test-service-")


@pytest.mark.asyncio
async def test_find_deployment_id_numeric_name_gets_prefix() -> None:
    """Test that all-numeric names get a 'd-' prefix for DNS-1035 compliance"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = True

        result = await k8s_client.find_deployment_id("10101010")

        assert result == "d-10101010"
        mock_validate.assert_called_once_with("d-10101010")


@pytest.mark.asyncio
async def test_find_deployment_id_digit_start_gets_prefix() -> None:
    """Test that names starting with a digit get a 'd-' prefix"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = True

        result = await k8s_client.find_deployment_id("123-service")

        assert result == "d-123-service"
        mock_validate.assert_called_once_with("d-123-service")


@pytest.mark.asyncio
async def test_find_deployment_id_alpha_start_no_prefix() -> None:
    """Test that names already starting with a letter don't get a prefix"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = True

        result = await k8s_client.find_deployment_id("service-123")

        assert result == "service-123"
        mock_validate.assert_called_once_with("service-123")


@pytest.mark.asyncio
async def test_find_deployment_id_empty_name_suffix_starts_with_letter() -> None:
    """Test that empty names (all special chars) produce DNS-compliant suffixes"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = True

        result = await k8s_client.find_deployment_id("!!!")

        # Name becomes empty after sanitization, so gets a random suffix
        # The suffix must start with a letter for DNS-1035 compliance
        assert len(result) == 5
        assert result[0].isalpha()


@pytest.mark.asyncio
async def test_find_deployment_id_max_length_63() -> None:
    """Test that deployment IDs are truncated to 63 chars (DNS-1035 max)"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = True

        long_name = "a" * 100
        result = await k8s_client.find_deployment_id(long_name)

        assert len(result) <= 63


@pytest.mark.asyncio
async def test_find_deployment_id_exhaustion() -> None:
    """Test behavior when all deployment IDs are taken"""
    with patch(
        "llama_agents.control_plane.k8s_client.validate_deployment_id"
    ) as mock_validate:
        mock_validate.return_value = False  # All IDs are taken

        with pytest.raises(ValueError) as exc_info:
            await k8s_client.find_deployment_id("test")

        assert "already in use" in str(exc_info.value)
        # Should call validate 99 times (range(1, 100))
        assert mock_validate.call_count == 99
