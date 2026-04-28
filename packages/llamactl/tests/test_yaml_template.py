# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ``cli.yaml_template.render``."""

from __future__ import annotations

import yaml as pyyaml
from llama_agents.cli.display import (
    DeploymentDisplay,
    DeploymentSpec,
    DeploymentStatus,
)
from llama_agents.cli.yaml_template import load_commented, render

import pytest


def _full_display() -> DeploymentDisplay:
    return DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(
            display_name="My App",
            repo_url="https://github.com/example/repo",
            deployment_file_path="llama_deploy.yaml",
            git_ref="main",
            appserver_version="0.5.0",
            suspended=False,
            secrets={"OPENAI_API_KEY": "sk-x"},
            personal_access_token=None,
        ),
        status=DeploymentStatus(phase="Running", project_id="proj_default"),
    )


def test_render_omits_status_unconditionally() -> None:
    out = render(_full_display())
    assert "status:" not in out
    assert "Running" not in out
    assert "phase" not in out


def test_render_emits_name_and_spec_in_declaration_order() -> None:
    out = render(_full_display())
    # `name:` is at the top.
    name_idx = out.index("name:")
    spec_idx = out.index("spec:")
    assert name_idx < spec_idx
    # Within spec, declaration order = display_name, repo_url, deployment_file_path...
    body = out[spec_idx:]
    fields_in_order = [
        "display_name:",
        "repo_url:",
        "deployment_file_path:",
        "git_ref:",
        "appserver_version:",
        "suspended:",
        "secrets:",
    ]
    last = -1
    for f in fields_in_order:
        idx = body.index(f)
        assert idx > last, f"{f} out of declaration order in:\n{body}"
        last = idx


def test_render_attaches_doc_marker_text_above_each_set_field() -> None:
    out = render(_full_display())
    # A few representative Docs.
    assert "#! Stable id for the deployment" in out
    assert "#! Human-readable name shown in the UI" in out
    assert "#! Branch, tag, or commit SHA to deploy" in out


def test_render_overrides_replace_doc_text_for_one_render() -> None:
    out = render(
        _full_display(),
        field_overrides={"repo_url": "Empty = push mode."},
    )
    assert "#! Empty = push mode." in out
    # Original Doc text must not appear for overridden field.
    assert "Empty string = push mode" not in out


def test_render_unknown_override_key_raises() -> None:
    with pytest.raises(ValueError, match="repo_uurl"):
        render(_full_display(), field_overrides={"repo_uurl": "bogus"})


def test_render_partial_spec_omits_unset_fields() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(display_name="My App"),
    )
    out = render(display)
    assert "display_name:" in out
    # Unset → not rendered.
    assert "repo_url:" not in out
    assert "git_ref:" not in out
    assert "secrets:" not in out


def test_render_scaffold_unset_appends_optional_block() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(display_name="My App", repo_url=""),
    )
    out = render(display, scaffold_unset=True)
    # Header line for the tail block.
    assert "#! Optional fields (uncomment to use):" in out
    # Each unset field with a Doc should appear in the tail.
    assert "# git_ref:" in out
    assert "# secrets:" in out
    assert "# appserver_version:" in out


def test_render_scaffold_unset_skips_set_fields() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(
            display_name="My App",
            repo_url="https://github.com/example/repo",
        ),
    )
    out = render(display, scaffold_unset=True)
    # repo_url is set → must NOT appear in the scaffold tail.
    after_set = out.split("#! Optional fields", 1)
    if len(after_set) == 2:
        assert "# repo_url:" not in after_set[1]


def test_render_head_lines_emit_at_top_with_prefix() -> None:
    out = render(
        _full_display(),
        head=("Edit, then run: llamactl deployments apply -f <file>",),
    )
    first = out.splitlines()[0]
    assert first.startswith("#! ")
    assert "Edit, then run" in first


def test_render_secret_comments_attach_inside_secrets_block() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(
            display_name="My App",
            secrets={"API_KEY": "${API_KEY}", "OTHER": "x"},
        ),
    )
    out = render(display, secret_comments={"API_KEY": "from your .env"})
    # Comment appears above the API_KEY entry, indented under secrets.
    secrets_idx = out.index("secrets:")
    block = out[secrets_idx:]
    assert "#! from your .env" in block
    api_key_idx = block.index("API_KEY")
    note_idx = block.index("from your .env")
    assert note_idx < api_key_idx


def test_render_empty_string_repo_url_is_double_quoted() -> None:
    """Empty repo_url is meaningful (push-mode signal). Render explicit ``""``."""
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(display_name="My App", repo_url=""),
    )
    out = render(display)
    assert 'repo_url: ""' in out


def test_render_output_is_yaml_safe_loadable() -> None:
    """Ignoring the `#!` lines, the result must parse as YAML."""
    out = render(_full_display())
    parsed = pyyaml.safe_load(out)
    assert parsed["name"] == "my-app"
    assert parsed["spec"]["display_name"] == "My App"
    assert "status" not in parsed


def test_render_scaffold_output_is_yaml_safe_loadable() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(display_name="My App", repo_url=""),
    )
    out = render(display, scaffold_unset=True, head=("hint",))
    parsed = pyyaml.safe_load(out)
    assert parsed["name"] == "my-app"
    assert parsed["spec"]["repo_url"] == ""


def test_render_idempotent_for_same_input() -> None:
    a = render(_full_display())
    b = render(_full_display())
    assert a == b


def test_load_commented_returns_commentedmap_round_trip() -> None:
    out = render(_full_display())
    loaded = load_commented(out)
    assert loaded["name"] == "my-app"
    assert loaded["spec"]["display_name"] == "My App"
