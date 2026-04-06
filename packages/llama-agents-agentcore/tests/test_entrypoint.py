from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest
from llama_agents.agentcore.entrypoint import (
    WorkflowResult,
    _load_workflows,
    _parse_and_validate_payload,
    invoke,
)
from workflows import Workflow
from workflows.events import StartEvent

from .conftest import (
    DummyFileWorkflow,
    DummyMetadataWorkflow,
    DummyWorkflow,
    DummyWorkflowWithError,
    FileEvent,
    MockAgentCoreService,
    MockContext,
)


@pytest.fixture(scope="module")
def loaded_workflows() -> dict[str, Workflow]:
    return {
        "default": DummyWorkflow(),
        "metadata": DummyMetadataWorkflow(),
        "process-file": DummyFileWorkflow(),
    }


def test_load_workflows_with_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded_workflows: dict[str, Workflow],
) -> None:
    with patch(
        "llama_agents.agentcore.entrypoint.read_deployment_config_from_git_root_or_cwd",
        new_callable=Mock,
    ) as mock_read:
        with patch(
            "llama_agents.agentcore.entrypoint.load_workflows", new_callable=Mock
        ) as mock_load:
            monkeypatch.chdir(tmp_path)

            (tmp_path / "pyproject.toml").touch()

            mock_read.return_value = {"config": {}}
            mock_load.return_value = loaded_workflows

            workflows, default_workflow, file_workflow = _load_workflows()
            _load_workflows.cache_clear()
            for key in workflows:
                assert key in loaded_workflows
                assert isinstance(workflows[key], type(loaded_workflows[key]))
            assert default_workflow == "default"
            assert file_workflow == "process-file"
            mock_read.assert_has_calls([call(Path.cwd(), Path.cwd())])
            mock_load.assert_called_once_with({"config": {}})


def test_load_workflows_without_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded_workflows: dict[str, Workflow],
) -> None:
    with patch(
        "llama_agents.agentcore.entrypoint.read_deployment_config_from_git_root_or_cwd",
        new_callable=Mock,
    ) as mock_read:
        with patch(
            "llama_agents.agentcore.entrypoint.load_workflows", new_callable=Mock
        ) as mock_load:
            loaded_workflows_cp = loaded_workflows.copy()

            loaded_workflows_cp.pop("default")

            monkeypatch.chdir(tmp_path)

            (tmp_path / "pyproject.toml").touch()

            mock_read.return_value = {"config": {}}
            mock_load.return_value = loaded_workflows_cp

            workflows, default_workflow, file_workflow = _load_workflows()
            _load_workflows.cache_clear()
            for key in workflows:
                assert key in loaded_workflows_cp
                assert isinstance(workflows[key], type(loaded_workflows_cp[key]))
            assert default_workflow == "metadata"
            assert file_workflow == "process-file"
            mock_read.assert_has_calls([call(Path.cwd(), Path.cwd())])
            mock_load.assert_called_once_with({"config": {}})


def test_load_workflows_without_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded_workflows: dict[str, Workflow],
) -> None:
    with patch(
        "llama_agents.agentcore.entrypoint.read_deployment_config_from_git_root_or_cwd",
        new_callable=Mock,
    ) as mock_read:
        with patch(
            "llama_agents.agentcore.entrypoint.load_workflows", new_callable=Mock
        ) as mock_load:
            loaded_workflows_cp = loaded_workflows.copy()

            loaded_workflows_cp.pop("process-file")

            monkeypatch.chdir(tmp_path)

            (tmp_path / "pyproject.toml").touch()

            mock_read.return_value = {"config": {}}
            mock_load.return_value = loaded_workflows_cp

            workflows, default_workflow, file_workflow = _load_workflows()
            _load_workflows.cache_clear()
            for key in workflows:
                assert key in loaded_workflows_cp
                assert isinstance(workflows[key], type(loaded_workflows_cp[key]))
            assert default_workflow == "default"
            assert file_workflow is None
            mock_read.assert_has_calls([call(Path.cwd(), Path.cwd())])
            mock_load.assert_called_once_with({"config": {}})


def test_parse_and_validate_payload_success(
    loaded_workflows: dict[str, Workflow],
) -> None:
    parsed = _parse_and_validate_payload(
        workflows=loaded_workflows,
        default_workflow="default",
        file_workflow="process-file",
        payload={"workflow": "default", "start_event": {}},
    )
    assert len(parsed) == 2
    assert isinstance(parsed[0], str)
    assert isinstance(parsed[1], StartEvent)
    assert parsed[0] == "default"
    assert parsed[1].model_dump() == {}


def test_parse_and_validate_payload_file_id(
    loaded_workflows: dict[str, Workflow],
) -> None:
    parsed = _parse_and_validate_payload(
        workflows=loaded_workflows,
        default_workflow="default",
        file_workflow="process-file",
        payload={"file_id": "1"},
    )
    assert len(parsed) == 2
    assert isinstance(parsed[0], str)
    assert isinstance(parsed[1], FileEvent)
    assert parsed[0] == "process-file"
    assert parsed[1].file_id == "1"


def test_parse_and_validate_payload_defaults(
    loaded_workflows: dict[str, Workflow],
) -> None:
    parsed = _parse_and_validate_payload(
        workflows=loaded_workflows,
        default_workflow="default",
        file_workflow="process-file",
        payload={},
    )
    assert len(parsed) == 2
    assert isinstance(parsed[0], str)
    assert isinstance(parsed[1], StartEvent)
    assert parsed[0] == "default"
    assert parsed[1].model_dump() == {}


def test_parse_and_validate_payload_workflow_not_found(
    loaded_workflows: dict[str, Workflow],
) -> None:
    parsed = _parse_and_validate_payload(
        workflows=loaded_workflows,
        default_workflow="default",
        file_workflow="process-file",
        payload={"workflow": "notfound", "start_event": {}},
    )
    assert len(parsed) == 2
    assert isinstance(parsed[0], str)
    assert isinstance(parsed[1], str)
    assert parsed[0] == "notfound"
    assert parsed[1] == "Workflow not found: notfound"


def test_parse_and_validate_payload_invalid_event(
    loaded_workflows: dict[str, Workflow],
) -> None:
    parsed = _parse_and_validate_payload(
        workflows=loaded_workflows,
        default_workflow="default",
        file_workflow="process-file",
        payload={"workflow": "process-file", "start_event": {"file_id": 1}},
    )
    assert len(parsed) == 2
    assert isinstance(parsed[0], str)
    assert isinstance(parsed[1], str)
    assert parsed[0] == "process-file"
    assert parsed[1].startswith("Invalid input data: ")


@pytest.mark.asyncio
async def test_invoke_success_default_action(
    loaded_workflows: dict[str, Workflow],
) -> None:
    """When no action is specified, invoke runs the workflow synchronously."""
    with (
        patch(
            "llama_agents.agentcore.entrypoint._load_workflows", new_callable=Mock
        ) as mock_load,
        patch(
            "llama_agents.agentcore.entrypoint.get_agentcore_service",
            new_callable=Mock,
        ) as mock_get_service,
    ):
        mock_service = MockAgentCoreService()
        mock_get_service.return_value = mock_service
        mock_load.return_value = (loaded_workflows, "default", "process-file")
        result = await invoke(
            {"workflow": "process-file", "start_event": {"file_id": "1"}},
            MockContext(),
        )
        assert isinstance(result, dict)
        assert result["session_id"] == "session-1"
        assert result["handler_id"] == "session-1"  # session_id used as handler_id
        assert result["workflow_name"] == "process-file"
        assert result["status"] == "completed"
        # run_workflow_with_session should have been called
        assert len(mock_service.call_args) == 1
        assert mock_service.call_args[0]["workflow_name"] == "process-file"
        assert mock_service.call_args[0]["handler_id"] == "session-1"
        assert mock_service.call_args[0]["start_event"] == FileEvent(file_id="1")


@pytest.mark.asyncio
async def test_invoke_with_explicit_handler_id(
    loaded_workflows: dict[str, Workflow],
) -> None:
    """Explicit handler_id in payload overrides session_id."""
    with (
        patch(
            "llama_agents.agentcore.entrypoint._load_workflows", new_callable=Mock
        ) as mock_load,
        patch(
            "llama_agents.agentcore.entrypoint.get_agentcore_service",
            new_callable=Mock,
        ) as mock_get_service,
    ):
        mock_service = MockAgentCoreService()
        mock_get_service.return_value = mock_service
        mock_load.return_value = (loaded_workflows, "default", "process-file")
        result = await invoke(
            {
                "workflow": "default",
                "start_event": {},
                "handler_id": "custom-handler-id",
            },
            MockContext(),
        )
        assert isinstance(result, dict)
        assert result["handler_id"] == "custom-handler-id"
        assert result["session_id"] == "session-1"
        assert mock_service.call_args[0]["handler_id"] == "custom-handler-id"


@pytest.mark.asyncio
async def test_invoke_error(loaded_workflows: dict[str, Workflow]) -> None:
    with (
        patch(
            "llama_agents.agentcore.entrypoint._load_workflows", new_callable=Mock
        ) as mock_load,
        patch(
            "llama_agents.agentcore.entrypoint.get_agentcore_service",
            new_callable=Mock,
        ) as mock_get_service,
    ):
        mock_service = MockAgentCoreService()
        mock_get_service.return_value = mock_service
        loaded_workflows_cp = loaded_workflows.copy()
        loaded_workflows_cp["with_error"] = DummyWorkflowWithError()
        mock_load.return_value = (loaded_workflows_cp, "default", "process-file")
        result = await invoke(
            {"workflow": "with_error", "start_event": {}},
            MockContext(),
        )
        assert isinstance(result, dict)
        validated = WorkflowResult.model_validate(result)
        assert validated.error is not None
        assert (
            "Workflow failed: " in validated.error
            and "You shall not pass!" in validated.error
        )
        assert validated.result is None
        assert validated.status == "failed"
        assert validated.session_id == "session-1"
        assert validated.workflow == "with_error"


@pytest.mark.asyncio
async def test_invoke_handler_with_error(loaded_workflows: dict[str, Workflow]) -> None:
    with (
        patch(
            "llama_agents.agentcore.entrypoint._load_workflows", new_callable=Mock
        ) as mock_load,
        patch(
            "llama_agents.agentcore.entrypoint.get_agentcore_service",
            new_callable=Mock,
        ) as mock_get_service,
    ):
        mock_service = MockAgentCoreService(with_error=True)
        mock_get_service.return_value = mock_service
        mock_load.return_value = (loaded_workflows, "default", "process-file")
        result = await invoke(
            {"workflow": "default", "start_event": {}},
            MockContext(),
        )
        assert isinstance(result, dict)
        assert result["error"] == "Some fancy error"
        assert result["status"] == "completed"  # handler completed but with error
        assert result["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_invoke_validation_error(loaded_workflows: dict[str, Workflow]) -> None:
    """Workflow not found returns a WorkflowResult error."""
    with (
        patch(
            "llama_agents.agentcore.entrypoint._load_workflows", new_callable=Mock
        ) as mock_load,
        patch(
            "llama_agents.agentcore.entrypoint.get_agentcore_service",
            new_callable=Mock,
        ) as mock_get_service,
    ):
        mock_service = MockAgentCoreService()
        mock_get_service.return_value = mock_service
        mock_load.return_value = (loaded_workflows, "default", "process-file")
        result = await invoke(
            {"workflow": "notfound", "start_event": {}},
            MockContext(),
        )
        assert isinstance(result, dict)
        validated = WorkflowResult.model_validate(result)
        assert validated.error == "Workflow not found: notfound"
        assert validated.status == "failed"
        assert validated.session_id == "session-1"


@pytest.mark.asyncio
async def test_invoke_run_nowait(loaded_workflows: dict[str, Workflow]) -> None:
    """action=run_nowait starts workflow without waiting."""
    with (
        patch(
            "llama_agents.agentcore.entrypoint._load_workflows", new_callable=Mock
        ) as mock_load,
        patch(
            "llama_agents.agentcore.entrypoint.get_agentcore_service",
            new_callable=Mock,
        ) as mock_get_service,
    ):
        mock_service = MockAgentCoreService()
        mock_get_service.return_value = mock_service
        mock_load.return_value = (loaded_workflows, "default", "process-file")
        result = await invoke(
            {"action": "run_nowait", "workflow": "default", "start_event": {}},
            MockContext(),
        )
        assert isinstance(result, dict)
        assert result["session_id"] == "session-1"
        assert mock_service.call_args[0]["nowait"] is True


@pytest.mark.asyncio
async def test_invoke_list_workflows(loaded_workflows: dict[str, Workflow]) -> None:
    """action=list_workflows returns workflow names."""
    with (
        patch(
            "llama_agents.agentcore.entrypoint._load_workflows", new_callable=Mock
        ) as mock_load,
        patch(
            "llama_agents.agentcore.entrypoint.get_agentcore_service",
            new_callable=Mock,
        ) as mock_get_service,
    ):
        mock_service = MockAgentCoreService()
        mock_get_service.return_value = mock_service
        mock_load.return_value = (loaded_workflows, "default", "process-file")
        result = await invoke(
            {"action": "list_workflows"},
            MockContext(),
        )
        assert isinstance(result, dict)
        assert "workflows" in result
        assert "session_id" in result


@pytest.mark.asyncio
async def test_invoke_unknown_action(loaded_workflows: dict[str, Workflow]) -> None:
    """Unknown action returns error with available actions."""
    with (
        patch(
            "llama_agents.agentcore.entrypoint._load_workflows", new_callable=Mock
        ) as mock_load,
        patch(
            "llama_agents.agentcore.entrypoint.get_agentcore_service",
            new_callable=Mock,
        ) as mock_get_service,
    ):
        mock_service = MockAgentCoreService()
        mock_get_service.return_value = mock_service
        mock_load.return_value = (loaded_workflows, "default", "process-file")
        result = await invoke(
            {"action": "invalid_action"},
            MockContext(),
        )
        assert isinstance(result, dict)
        assert "error" in result
        assert "available" in result
        assert "run" in result["available"]
