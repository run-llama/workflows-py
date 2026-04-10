# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

from dataclasses import dataclass

import click
import llama_agents.cli.config.env_service as es
import llama_agents.cli.param_types as pt
import pytest
from click.shell_completion import CompletionItem
from llama_agents.cli.param_types import (
    DeploymentType,
    EnvironmentType,
    GitShaType,
    ProfileType,
    ProjectType,
    TemplateType,
)


@pytest.fixture()
def ctx() -> click.Context:
    """A minimal Click context for shell_complete calls."""
    cmd = click.Command("test")
    return click.Context(cmd)


@pytest.fixture()
def param() -> click.Parameter:
    return click.Argument(["test_arg"])


def test_deployment_type_returns_fetched_items(
    ctx: click.Context, param: click.Parameter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pt,
        "_fetch_deployments",
        lambda: [
            CompletionItem("my-app"),
            CompletionItem("staging"),
        ],
    )

    dt = DeploymentType()
    items = dt.shell_complete(ctx, param, "")
    assert len(items) == 2
    assert items[0].value == "my-app"

    # Filter by prefix
    items = dt.shell_complete(ctx, param, "my")
    assert len(items) == 1
    assert items[0].value == "my-app"


def test_deployment_type_fetch_failure(
    ctx: click.Context, param: click.Parameter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> list[CompletionItem]:
        raise RuntimeError("API down")

    monkeypatch.setattr(pt, "_fetch_deployments", _boom)

    dt = DeploymentType()
    items = dt.shell_complete(ctx, param, "")
    assert items == []


def test_profile_type_returns_profiles(
    ctx: click.Context, param: click.Parameter, monkeypatch: pytest.MonkeyPatch
) -> None:
    @dataclass
    class FakeAuth:
        name: str
        api_url: str

    class FakeAuthService:
        def list_profiles(self) -> list[FakeAuth]:
            return [
                FakeAuth(name="prod", api_url="https://api.prod.example.com"),
                FakeAuth(name="dev", api_url="https://api.dev.example.com"),
            ]

    class FakeService:
        def current_auth_service(self) -> FakeAuthService:
            return FakeAuthService()

    monkeypatch.setattr(es, "service", FakeService())

    prof = ProfileType()
    items = prof.shell_complete(ctx, param, "")
    assert len(items) == 2
    assert items[0].value == "prod"
    assert items[0].help == "https://api.prod.example.com"

    items = prof.shell_complete(ctx, param, "dev")
    assert len(items) == 1
    assert items[0].value == "dev"


def test_project_type_returns_fetched_items(
    ctx: click.Context, param: click.Parameter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pt,
        "_fetch_projects",
        lambda: [
            CompletionItem("proj_abc", help="My Project (3 deployments)"),
            CompletionItem("proj_def", help="Staging (1 deployment)"),
        ],
    )

    proj = ProjectType()
    items = proj.shell_complete(ctx, param, "")
    assert len(items) == 2
    assert items[0].value == "proj_abc"


def test_environment_type_returns_environments(
    ctx: click.Context, param: click.Parameter, monkeypatch: pytest.MonkeyPatch
) -> None:
    @dataclass
    class FakeEnv:
        api_url: str
        requires_auth: bool = True

    class FakeService:
        def list_environments(self) -> list[FakeEnv]:
            return [
                FakeEnv(api_url="https://api.prod.example.com"),
                FakeEnv(api_url="https://api.dev.example.com"),
            ]

        def get_current_environment(self) -> FakeEnv:
            return FakeEnv(api_url="https://api.prod.example.com")

    monkeypatch.setattr(es, "service", FakeService())

    et = EnvironmentType()
    items = et.shell_complete(ctx, param, "")
    assert len(items) == 2
    # Current env should have "(current)" help
    assert items[0].help == "(current)"
    assert items[1].help == ""


def test_template_type_returns_all_templates(
    ctx: click.Context, param: click.Parameter
) -> None:
    tt = TemplateType()
    items = tt.shell_complete(ctx, param, "")
    assert len(items) == 12  # 6 UI + 6 headless

    # Filter
    items = tt.shell_complete(ctx, param, "basic")
    assert len(items) == 2  # basic-ui and basic


def test_template_type_case_insensitive(
    ctx: click.Context, param: click.Parameter
) -> None:
    tt = TemplateType()
    items = tt.shell_complete(ctx, param, "RAG")
    assert len(items) == 1
    assert items[0].value == "rag"


def test_git_sha_type_no_deployment_id(
    ctx: click.Context, param: click.Parameter
) -> None:
    ctx.params = {}
    gt = GitShaType()
    items = gt.shell_complete(ctx, param, "")
    assert items == []


def test_git_sha_type_with_deployment_id(
    ctx: click.Context, param: click.Parameter, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx.params = {"deployment_id": "my-deploy"}
    monkeypatch.setattr(
        pt,
        "_fetch_deployment_history",
        lambda dep_id: [
            CompletionItem("abc1234", help="2026-01-01T00:00:00"),
            CompletionItem("def5678", help="2026-01-02T00:00:00"),
        ],
    )

    gt = GitShaType()
    items = gt.shell_complete(ctx, param, "")
    assert len(items) == 2

    items = gt.shell_complete(ctx, param, "abc")
    assert len(items) == 1
    assert items[0].value == "abc1234"
