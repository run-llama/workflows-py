from pathlib import Path

import pytest
from llama_agents.agentcore.export import (
    agentcore_dir,
    export_generated_entrypoint_code,
)


def test_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    export_generated_entrypoint_code()
    assert (tmp_path / agentcore_dir).is_dir()
    assert (tmp_path / agentcore_dir / "entrypoint.py").is_file()
    content = (tmp_path / agentcore_dir / "entrypoint.py").read_text()
    assert content.startswith("# SPDX-License-Identifier: MIT")
    assert content.endswith(
        'return {"error": str(e), "action": action, "session_id": session_id}\n'
    )
