# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ``cli.yaml_template.render``."""

from __future__ import annotations

import yaml as pyyaml
from llama_agents.cli.display import (
    SECRET_MASK,
    DeploymentDisplay,
    DeploymentSpec,
    DeploymentStatus,
)
from llama_agents.cli.yaml_template import render


def _full_display() -> DeploymentDisplay:
    return DeploymentDisplay(
        name="my-app",
        generate_name="My App",
        spec=DeploymentSpec(
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
    name_idx = out.index("name:")
    spec_idx = out.index("spec:")
    assert name_idx < spec_idx
    body = out[spec_idx:]
    fields_in_order = [
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
    assert "## Stable id for the deployment" in out
    assert "## Branch, tag, or commit SHA to deploy" in out


def test_render_omits_generate_name_when_not_scaffolded() -> None:
    """``scaffold_generate_name=False`` (default) skips the field entirely —
    no comment block, no commented-out line."""
    out = render(_full_display())
    assert "generate_name" not in out
    assert "name takes precedence" not in out


def test_render_emits_generate_name_commented_when_scaffolded() -> None:
    """``scaffold_generate_name=True`` emits a commented identity-tier line
    with the ``Doc`` block above; the value is the model field's value."""
    out = render(_full_display(), scaffold_generate_name=True)
    assert "## name takes precedence" in out
    assert "# generate_name: My App" in out
    # Commented-out (not authoritative).
    assert "\ngenerate_name:" not in out


def test_render_uses_name_example_for_unset_top_level_keys() -> None:
    """``name=None`` and (with scaffold) ``generate_name=None`` both fall back
    to the ``name_example`` argument."""
    out = render(
        DeploymentDisplay(name=None, spec=DeploymentSpec()),
        name_example="cwd-name",
        scaffold_generate_name=True,
    )
    assert "# name: cwd-name" in out
    assert "# generate_name: cwd-name" in out


def test_render_partial_spec_emits_unset_fields_as_commented_examples() -> None:
    """Unset (not required) spec fields render as commented-out one-liners
    with a doc above, in declaration order inside the spec block."""
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(),
    )
    out = render(display)
    assert "  # repo_url:" in out
    assert "  # git_ref: main" in out
    assert "  # secrets:" in out
    assert "    # MY_SECRET: ${MY_SECRET}" in out
    git_ref_idx = out.index("  # git_ref:")
    assert "## Branch, tag, or commit SHA to deploy." in out[:git_ref_idx]


def test_render_required_unset_emits_tilde_with_required_line() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(),
    )
    out = render(display, required=("repo_url",))
    assert "  repo_url: ~" in out
    repo_idx = out.index("  repo_url: ~")
    assert "## Required — set before `apply`." in out[:repo_idx]


def test_render_unset_top_level_name_is_commented() -> None:
    """``DeploymentDisplay.name=None`` renders the top-level key commented-out
    (the server slugifies an id when ``name`` is unset on apply)."""
    display = DeploymentDisplay(
        name=None,
        spec=DeploymentSpec(),
    )
    out = render(display)
    assert "\n# name: my-app" in out or out.startswith("# name: my-app")
    # No bare ``name:`` line at the top level.
    assert "\nname:" not in out
    assert not out.startswith("name:")


def test_render_alternatives_emit_commented_line_under_set_field() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(repo_url=""),
    )
    out = render(
        display,
        field_alternatives={
            "repo_url": (
                "https://github.com/owner/repo",
                "auto-detected from your git remotes",
            )
        },
    )
    repo_idx = out.index('  repo_url: ""')
    after = out[repo_idx:]
    next_line = after.splitlines()[1]
    assert (
        next_line
        == "  # repo_url: https://github.com/owner/repo  # auto-detected from your git remotes"
    )
    parsed = pyyaml.safe_load(out)
    assert parsed["spec"]["repo_url"] == ""


def test_render_alternatives_ignored_for_unset_field() -> None:
    """An alternative on a field that's not set is silently ignored — the
    alternative is conceptually 'a different value than the one you have',
    not 'an example'."""
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(),
    )
    out = render(
        display,
        field_alternatives={
            "repo_url": ("https://github.com/owner/repo", "detected"),
        },
    )
    assert "# detected" not in out


def test_render_multi_line_doc_emits_one_comment_per_chunk() -> None:
    """``Doc`` text containing ``\\n`` becomes one ``## `` line per chunk.

    Uses the real ``repo_url`` Doc, which is the schema's only multi-line
    Doc — the rendered output above ``repo_url:`` should list each line as
    its own marker comment.
    """
    out = render(_full_display())
    head = out.split("repo_url:", 1)[0]
    expected_lines = [
        "## Git repository URL. Supported shapes:",
        '## - "" = push mode (the CLI pushes your working tree on apply).',
        "## - https://github.com/<owner>/<repo> = GitHub HTTPS",
        "## - https://gitlab.com/<owner>/<repo> = GitLab HTTPS",
    ]
    last = -1
    for marker in expected_lines:
        idx = head.find(marker)
        assert idx > last, f"missing or out-of-order: {marker!r}\nin:\n{head}"
        last = idx
    for marker in expected_lines:
        assert f"  {marker}" in head


def test_render_head_lines_emit_at_top_with_prefix() -> None:
    out = render(
        _full_display(),
        head=("Edit, then run: llamactl deployments apply -f <file>",),
    )
    first = out.splitlines()[0]
    assert first.startswith("## ")
    assert "Edit, then run" in first


def test_render_head_blank_line_emits_bare_marker() -> None:
    out = render(
        _full_display(),
        head=("first", "", "third"),
    )
    lines = out.splitlines()
    assert lines[0] == "## first"
    assert lines[1] == "##"
    assert lines[2] == "## third"


def test_render_secret_comments_attach_inside_secrets_block() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(
            secrets={"API_KEY": "${API_KEY}", "OTHER": "x"},
        ),
    )
    out = render(display, secret_comments={"API_KEY": "from your .env"})
    secrets_idx = out.index("secrets:")
    block = out[secrets_idx:]
    assert "## from your .env" in block
    api_key_idx = block.index("API_KEY")
    note_idx = block.index("from your .env")
    assert note_idx < api_key_idx


def test_render_empty_string_repo_url_is_double_quoted() -> None:
    """Empty repo_url is meaningful (push-mode signal). Render explicit ``""``."""
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(repo_url=""),
    )
    out = render(display)
    assert 'repo_url: ""' in out


def test_render_deployment_file_path_dot_is_quoted() -> None:
    """Bare-dot path renders as ``"."`` so the value reads as a path string."""
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(deployment_file_path="."),
    )
    out = render(display)
    assert 'deployment_file_path: "."' in out


def test_render_strips_secret_mask_sentinels() -> None:
    """``SECRET_MASK`` values inside ``secrets`` are dropped before render —
    the mask must never round-trip into apply input."""
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(
            secrets={"REAL": "${REAL}", "MASKED": SECRET_MASK},
        ),
    )
    out = render(display)
    assert SECRET_MASK not in out
    assert "MASKED" not in out
    # The remaining real secret is still rendered.
    assert "REAL: ${REAL}" in out


def test_render_drops_secrets_block_when_only_masks() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(
            secrets={"ONLY_MASKED": SECRET_MASK},
        ),
    )
    out = render(display)
    # No uncommented secrets block — falls into the unset/example branch.
    assert "  secrets:\n    ONLY_MASKED" not in out
    assert SECRET_MASK not in out
    # Commented-out example is what surfaces.
    assert "  # secrets:" in out


def test_render_strips_personal_access_token_mask() -> None:
    display = DeploymentDisplay(
        name="my-app",
        spec=DeploymentSpec(personal_access_token=SECRET_MASK),
    )
    out = render(display)
    assert SECRET_MASK not in out
    # Falls through to the commented example.
    assert "  # personal_access_token:" in out


def test_render_output_is_yaml_safe_loadable() -> None:
    out = render(_full_display())
    parsed = pyyaml.safe_load(out)
    assert parsed["name"] == "my-app"
    assert "display_name" not in parsed["spec"]
    assert "generate_name" not in parsed["spec"]
    assert "status" not in parsed


def test_render_output_with_required_and_alternatives_is_yaml_safe_loadable() -> None:
    display = DeploymentDisplay(
        name=None,
        spec=DeploymentSpec(repo_url=""),
    )
    out = render(
        display,
        head=("hint",),
        required=("git_ref",),
        field_alternatives={"repo_url": ("https://example.com/x", "detected")},
    )
    parsed = pyyaml.safe_load(out)
    assert "name" not in parsed
    assert parsed["spec"]["git_ref"] is None
    assert parsed["spec"]["repo_url"] == ""


def test_render_idempotent_for_same_input() -> None:
    a = render(_full_display())
    b = render(_full_display())
    assert a == b
