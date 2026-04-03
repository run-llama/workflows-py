"""Tests for client.py - Business logic only"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from llama_agents.cli.client import get_project_client


def test_deployment_project_resolution() -> None:
    """Test that get_project_client uses profile's project by default"""
    profile = SimpleNamespace(
        api_url="http://test:8011",
        project_id="default-project",
        api_key=None,
    )
    with patch("llama_agents.cli.config.env_service.service") as mock_service:
        mock_auth_svc = MagicMock()
        mock_auth_svc.get_current_profile.return_value = profile
        mock_service.current_auth_service.return_value = mock_auth_svc
        client = get_project_client()
        assert client.base_url == "http://test:8011"
        assert client.project_id == "default-project"


def test_client_requires_profile_with_project() -> None:
    """Test that client works when profile has a project (project_id is required)"""
    profile = SimpleNamespace(
        api_url="http://test:8011", project_id="test-project", api_key=None
    )
    with patch("llama_agents.cli.config.env_service.service") as mock_service:
        mock_auth_svc = MagicMock()
        mock_auth_svc.get_current_profile.return_value = profile
        mock_service.current_auth_service.return_value = mock_auth_svc
        client = get_project_client()
        assert client.project_id == "test-project"


def test_client_requires_valid_profile() -> None:
    """Test that client fails when no profile is configured"""
    with patch("llama_agents.cli.config.env_service.service") as mock_service:
        mock_auth_svc = MagicMock()
        mock_auth_svc.get_current_profile.return_value = None
        mock_service.current_auth_service.return_value = mock_auth_svc
        with pytest.raises(SystemExit):
            get_project_client()
