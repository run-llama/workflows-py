# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ``cli.yaml_format.parse_apply_yaml`` and helpers."""

from __future__ import annotations

import pytest
from llama_agents.cli.yaml_format import (
    ApplyYamlError,
    UnresolvedVarsError,
    parse_apply_yaml,
    resolve_env_vars,
)


def test_parse_minimal_with_name() -> None:
    parsed = parse_apply_yaml(
        """
        name: my-app
        display_name: My App
        repo_url: https://github.com/example/repo
        git_ref: main
        """,
        env={},
    )
    assert parsed.name == "my-app"
    assert parsed.apply.display_name == "My App"
    assert parsed.apply.repo_url == "https://github.com/example/repo"
    assert parsed.apply.git_ref == "main"


def test_parse_without_name_returns_none() -> None:
    parsed = parse_apply_yaml(
        """
        display_name: My App
        """,
        env={},
    )
    assert parsed.name is None
    assert parsed.apply.display_name == "My App"


def test_var_resolution_in_secrets_and_scalars() -> None:
    parsed = parse_apply_yaml(
        """
        name: my-app
        git_ref: ${BRANCH}
        secrets:
          OPENAI_API_KEY: ${OPENAI_KEY}
          OTHER: literal
        """,
        env={"BRANCH": "main", "OPENAI_KEY": "sk-x"},
    )
    assert parsed.apply.git_ref == "main"
    assert parsed.apply.secrets == {"OPENAI_API_KEY": "sk-x", "OTHER": "literal"}


def test_unresolved_vars_strict_raises_with_all_names() -> None:
    with pytest.raises(UnresolvedVarsError) as exc_info:
        parse_apply_yaml(
            """
            name: my-app
            git_ref: ${BRANCH}
            secrets:
              A: ${MISSING_A}
              B: ${MISSING_B}
            """,
            env={},
        )
    assert "BRANCH" in exc_info.value.names
    assert "MISSING_A" in exc_info.value.names
    assert "MISSING_B" in exc_info.value.names


def test_unresolved_vars_non_strict_keeps_literal() -> None:
    resolved, missing = resolve_env_vars(
        {"git_ref": "${BRANCH}"}, env={}, strict=False
    )
    assert resolved == {"git_ref": "${BRANCH}"}
    assert missing == ["BRANCH"]


def test_secret_mask_passthrough_is_dropped() -> None:
    parsed = parse_apply_yaml(
        """
        name: my-app
        secrets:
          KEEP_ME: real-value
          MASKED: ${MASK}
        """,
        # Resolves to the literal mask token, mimicking a get -o yaml round-trip.
        env={"MASK": "********"},
    )
    assert parsed.apply.secrets == {"KEEP_ME": "real-value"}


def test_null_secret_preserved_for_deletion() -> None:
    parsed = parse_apply_yaml(
        """
        name: my-app
        secrets:
          DROP_ME: null
          KEEP_ME: real
        """,
        env={},
    )
    assert parsed.apply.secrets == {"DROP_ME": None, "KEEP_ME": "real"}


def test_empty_repo_url_preserved_for_push_mode() -> None:
    parsed = parse_apply_yaml(
        """
        name: my-app
        display_name: My App
        repo_url: ""
        """,
        env={},
    )
    # Empty string is the documented push-mode sentinel — must not become None.
    assert parsed.apply.repo_url == ""


def test_schema_validation_error_includes_field_path() -> None:
    with pytest.raises(ApplyYamlError) as exc_info:
        parse_apply_yaml(
            """
            name: my-app
            suspended: not-a-bool
            """,
            env={},
        )
    assert "suspended" in str(exc_info.value)


def test_yaml_syntax_error_surfaces_as_apply_yaml_error() -> None:
    with pytest.raises(ApplyYamlError) as exc_info:
        parse_apply_yaml("name: my-app\n  bad: indent\n", env={})
    assert "YAML parse error" in str(exc_info.value)


def test_non_mapping_top_level_errors() -> None:
    with pytest.raises(ApplyYamlError) as exc_info:
        parse_apply_yaml("- not\n- a\n- mapping\n", env={})
    assert "mapping" in str(exc_info.value)


def test_empty_document_errors() -> None:
    with pytest.raises(ApplyYamlError):
        parse_apply_yaml("", env={})


def test_source_appears_in_error_messages() -> None:
    with pytest.raises(ApplyYamlError) as exc_info:
        parse_apply_yaml("- not a mapping\n", env={}, source="deployment.yaml")
    assert "deployment.yaml" in str(exc_info.value)


def test_non_string_name_errors() -> None:
    with pytest.raises(ApplyYamlError) as exc_info:
        parse_apply_yaml("name: 123\n", env={})
    assert "name" in str(exc_info.value)
