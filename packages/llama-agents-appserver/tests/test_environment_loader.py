from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest
from llama_agents.appserver.workflow_loader import (
    load_environment_variables,
    validate_required_env_vars,
)
from llama_agents.core.deployment_config import DeploymentConfig


@pytest.fixture(autouse=True)
def _cleanup_env() -> Iterator[None]:
    # Ensure test-set env vars are cleaned between tests
    before = set(os.environ.keys())
    yield
    for key in list(os.environ.keys()):
        if key not in before:
            os.environ.pop(key, None)


def test_env_loader_sets_from_env_dict_only(tmp_path: Path) -> None:
    cfg = DeploymentConfig.model_validate(
        {
            "env": {"FOO": "bar", "HELLO": "world"},
        }
    )
    load_environment_variables(cfg, tmp_path)
    assert os.environ.get("FOO") == "bar"
    assert os.environ.get("HELLO") == "world"


def test_env_loader_env_files_override_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=baz\nNEW_KEY=123\n")

    cfg = DeploymentConfig.model_validate(
        {
            "name": "n",
            "env": {"FOO": "bar"},
            "env_files": [".env"],
        }
    )
    load_environment_variables(cfg, tmp_path)
    # .env should override the inline env value
    assert os.environ.get("FOO") == "baz"
    # .env should add new keys
    assert os.environ.get("NEW_KEY") == "123"


def test_env_loader_multiple_env_files_last_wins(tmp_path: Path) -> None:
    (tmp_path / "a.env").write_text("X=1\nY=from_a\n")
    (tmp_path / "b.env").write_text("X=2\n")

    cfg = DeploymentConfig.model_validate(
        {
            "name": "n",
            "env": {"Y": "inline"},
            "env_files": ["a.env", "b.env"],
        }
    )
    load_environment_variables(cfg, tmp_path)
    # Later env file overrides earlier
    assert os.environ.get("X") == "2"
    # First env file overrides inline
    assert os.environ.get("Y") == "from_a"


def test_env_loader_missing_env_file_is_ignored(tmp_path: Path) -> None:
    cfg = DeploymentConfig.model_validate(
        {
            "env": {"FOO": "bar"},
            "env_files": ["does_not_exist.env"],
        }
    )
    load_environment_variables(cfg, tmp_path)
    # Should still set from inline env
    assert os.environ.get("FOO") == "bar"


def test_env_loader_skips_empty_values(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("EMPTY=\n")

    cfg = DeploymentConfig.model_validate(
        {
            "name": "n",
            "env": {"ALSO_EMPTY": ""},
            "env_files": [".env"],
        }
    )
    load_environment_variables(cfg, tmp_path)
    # Falsy values should not be set by the loader
    assert "EMPTY" not in os.environ or os.environ["EMPTY"]
    assert "ALSO_EMPTY" not in os.environ or os.environ["ALSO_EMPTY"]


def test_validate_required_env_vars_raises_for_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure a clean environment for this test
    monkeypatch.delenv("REQ_ONE", raising=False)
    monkeypatch.delenv("REQ_TWO", raising=False)

    cfg = DeploymentConfig.model_validate(
        {
            "name": "n",
            "required_env_vars": ["REQ_ONE", "REQ_TWO"],
        }
    )
    with pytest.raises(RuntimeError) as exc:
        validate_required_env_vars(cfg)
    assert "REQ_ONE" in str(exc.value) and "REQ_TWO" in str(exc.value)


def test_validate_required_env_vars_passes_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REQ_ONE", "1")
    monkeypatch.setenv("REQ_TWO", "2")
    cfg = DeploymentConfig.model_validate(
        {
            "name": "n",
            "required_env_vars": ["REQ_ONE", "REQ_TWO"],
        }
    )
    # Should not raise
    validate_required_env_vars(cfg)


def test_validate_required_env_vars_fill_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure env vars are not set
    monkeypatch.delenv("FILL_ONE", raising=False)
    monkeypatch.delenv("FILL_TWO", raising=False)

    cfg = DeploymentConfig.model_validate(
        {
            "name": "n",
            "required_env_vars": ["FILL_ONE", "FILL_TWO"],
        }
    )
    # Should not raise when fill_missing=True
    validate_required_env_vars(cfg, fill_missing=True)

    # Check that placeholder values were set
    assert os.environ.get("FILL_ONE") == "__PLACEHOLDER_FILL_ONE__"
    assert os.environ.get("FILL_TWO") == "__PLACEHOLDER_FILL_TWO__"


def test_validate_required_env_vars_fill_missing_only_fills_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Set one env var, leave the other unset
    monkeypatch.setenv("PARTIAL_ONE", "real_value")
    monkeypatch.delenv("PARTIAL_TWO", raising=False)

    cfg = DeploymentConfig.model_validate(
        {
            "name": "n",
            "required_env_vars": ["PARTIAL_ONE", "PARTIAL_TWO"],
        }
    )
    validate_required_env_vars(cfg, fill_missing=True)

    # The set variable should keep its value
    assert os.environ.get("PARTIAL_ONE") == "real_value"
    # The unset variable should get a placeholder
    assert os.environ.get("PARTIAL_TWO") == "__PLACEHOLDER_PARTIAL_TWO__"
