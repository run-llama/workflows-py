import json
from pathlib import Path
from unittest.mock import patch

import pytest
from llama_agents.control_plane.git import git_service
from llama_agents.control_plane.git._git_service import GitRepository
from llama_agents.core.git.git_util import GitCloneResult


@pytest.mark.asyncio
async def test_validate_git_application_ui_path_for_pyproject_at_examples_basic_ui(
    tmp_path: Path,
) -> None:
    # Arrange: mock clone_repo to populate a temp repo layout with pyproject at examples/basic_ui and UI under examples/basic_ui/ui
    async def _mock_clone_repo(
        repository_url: str,
        git_ref: str | None = None,
        basic_auth: str | None = None,
        dest_dir: Path | str | None = None,
    ) -> GitCloneResult:
        # Simulate repo_root
        if dest_dir is None:
            raise AssertionError("dest_dir must be provided for _mock_clone_repo")
        repo_root = Path(dest_dir)
        # examples/basic_ui/pyproject.toml with tool.llamadeploy
        config_dir = repo_root / "examples" / "basic_ui"
        ui_dir = config_dir / "ui"
        config_dir.mkdir(parents=True, exist_ok=True)
        ui_dir.mkdir(parents=True, exist_ok=True)

        # minimal pyproject.toml with name so config validates after adding a workflow
        (config_dir / "pyproject.toml").write_text(
            """
[project]
name = "basic-ui"

[tool.llamadeploy]
name = "basic-ui"
workflows = { default = "quick_start.workflow:app" }
[tool.llamadeploy.ui]
directory = "ui"
""".strip()
        )

        # minimal package.json with a build script so ui_build_output_path is detected
        (ui_dir / "package.json").write_text(
            json.dumps(
                {"name": "ui", "version": "0.0.0", "scripts": {"build": "vite build"}}
            )
        )

        return GitCloneResult(git_sha="deadbeef", git_ref=git_ref or "main")

    with (
        patch(
            "llama_agents.control_plane.git._git_service.clone_repo",
            new=_mock_clone_repo,
        ),
        # Pretend repo access is fine to reach the config parsing logic
        patch(
            "llama_agents.control_plane.git._git_service.GitService.obtain_basic_auth_token",
            return_value=(
                None,
                GitRepository(url="https://example.com/repo.git", access_token=None),
            ),
        ),
    ):
        # Act
        result = await git_service.validate_git_application(
            repository_url="https://example.com/repo.git",
            git_ref=None,
            deployment_file_path="examples/basic_ui",
            deployment_id=None,
            pat=None,
        )

    # Assert
    assert result.is_valid is True
    # Expect static assets path relative to repo root
    assert result.ui_build_output_path == Path("examples/basic_ui/ui/dist")
