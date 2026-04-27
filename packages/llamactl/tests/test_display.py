# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for the CLI ``DeploymentDisplay`` projection model."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import make_deployment
from llama_agents.cli.display import (
    SECRET_MASK,
    DeploymentDisplay,
    DeploymentSpec,
    DeploymentStatus,
)
from pydantic import ValidationError


def test_from_response_translates_spec_fields() -> None:
    response = make_deployment(
        "my-app",
        display_name="My App",
        repo_url="https://github.com/example/repo",
        deployment_file_path="llama_deploy.yaml",
        git_ref="main",
        appserver_version="0.4.2",
        suspended=True,
    )
    display = DeploymentDisplay.from_response(response)

    # ``name`` is the stable id, NOT the deprecated ``r.name`` alias.
    assert display.name == "my-app"
    assert isinstance(display.spec, DeploymentSpec)
    assert display.spec.display_name == "My App"
    assert display.spec.repo_url == "https://github.com/example/repo"
    assert display.spec.deployment_file_path == "llama_deploy.yaml"
    assert display.spec.git_ref == "main"
    assert display.spec.appserver_version == "0.4.2"
    assert display.spec.suspended is True


def test_from_response_status_block() -> None:
    response = make_deployment(
        "my-app",
        git_sha="abc123",
        apiserver_url="http://my-app.svc.cluster.local/",
        warning="upgrade me",
    )
    display = DeploymentDisplay.from_response(response)

    assert isinstance(display.status, DeploymentStatus)
    assert display.status.phase == "Running"
    assert display.status.git_sha == "abc123"
    assert display.status.apiserver_url == "http://my-app.svc.cluster.local/"
    assert display.status.project_id == "proj_default"
    assert display.status.warning == "upgrade me"


def test_from_response_secrets_masked() -> None:
    response = make_deployment(
        "my-app", secret_names=["LLAMA_CLOUD_API_KEY", "OPENAI_API_KEY"]
    )
    display = DeploymentDisplay.from_response(response)

    assert display.spec.secrets == {
        "LLAMA_CLOUD_API_KEY": SECRET_MASK,
        "OPENAI_API_KEY": SECRET_MASK,
    }


def test_from_response_no_secrets_is_none() -> None:
    response = make_deployment("my-app", secret_names=[])
    display = DeploymentDisplay.from_response(response)
    assert display.spec.secrets is None

    response = make_deployment("my-app", secret_names=None)
    display = DeploymentDisplay.from_response(response)
    assert display.spec.secrets is None


def test_from_response_pat_masking() -> None:
    response = make_deployment("my-app", has_personal_access_token=True)
    display = DeploymentDisplay.from_response(response)
    assert display.spec.personal_access_token == SECRET_MASK

    response = make_deployment("my-app", has_personal_access_token=False)
    display = DeploymentDisplay.from_response(response)
    assert display.spec.personal_access_token is None


def test_to_output_dict_omits_empty_spec_fields() -> None:
    response = make_deployment("my-app")  # no secrets, no PAT
    data = DeploymentDisplay.from_response(response).to_output_dict()

    spec = data["spec"]
    assert "secrets" not in spec
    assert "personal_access_token" not in spec
    # Editable defaults still surface inside spec so apply round-trips cleanly.
    assert data["name"] == "my-app"
    assert spec["suspended"] is False


def test_to_output_dict_keeps_explicit_status_warning_null() -> None:
    response = make_deployment("my-app", warning=None)
    data = DeploymentDisplay.from_response(response).to_output_dict()
    assert data["status"]["warning"] is None


def test_to_output_dict_includes_secrets_and_pat_when_set() -> None:
    response = make_deployment(
        "my-app",
        secret_names=["KEY"],
        has_personal_access_token=True,
    )
    data = DeploymentDisplay.from_response(response).to_output_dict()
    assert data["spec"]["secrets"] == {"KEY": SECRET_MASK}
    assert data["spec"]["personal_access_token"] == SECRET_MASK


def test_display_model_forbids_extra_fields() -> None:
    """Adding an unknown wire field should fail loudly during translation."""
    with pytest.raises(ValidationError):
        DeploymentDisplay.model_validate(
            {
                "name": "x",
                "spec": {
                    "display_name": "x",
                    "repo_url": "https://github.com/x/y",
                    "deployment_file_path": ".",
                },
                "novel_field": "leak",
            }
        )


def test_spec_model_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DeploymentSpec.model_validate(
            {
                "display_name": "x",
                "repo_url": "https://github.com/x/y",
                "deployment_file_path": ".",
                "novel_field": "leak",
            }
        )


def test_status_model_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DeploymentStatus.model_validate(
            {"phase": "Running", "project_id": "proj", "extra_field": "x"}
        )


def test_no_legacy_aliases_in_output(monkeypatch: Any) -> None:
    """Sanity: deprecated wire aliases (``id``, ``llama_deploy_version``,
    ``has_personal_access_token``, ``secret_names``) must not leak."""
    response = make_deployment(
        "my-app",
        secret_names=["KEY"],
        has_personal_access_token=True,
        appserver_version="0.4.2",
    )
    data = DeploymentDisplay.from_response(response).to_output_dict()
    for forbidden in (
        "id",
        "llama_deploy_version",
        "has_personal_access_token",
        "secret_names",
    ):
        assert forbidden not in data
